#!/usr/bin/env python3
"""Train a learnable head on top of cached MiewID features (M3 Pro path).

This is the cheap fine-tuning variant: MiewID stays frozen and we train a small
projection head (linear or MLP) on the cached 2152-dim vectors. The head learns
an orca-specific re-weighting of MiewID's representation.

Full backbone fine-tuning lives at BYU GPU lab (see docs/byu-gpu-finetuning-plan.md).

Usage:

    python scripts/finetune_miewid_head.py \\
        --features artifacts/miewid_features/orca_all.npz \\
        --splits-dir data/manifests/orca_all_splits \\
        --head mlp \\
        --hidden-dim 512 \\
        --embed-dim 256 \\
        --epochs 50 \\
        --run-name miewid-mlp-head-001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.miewid_finetune import (  # noqa: E402
    MIEWID_FEATURE_DIM,
    CachedFeatureDataset,
    HeadCheckpointMetadata,
    build_head,
    load_cached_features,
    project_cached_features,
    save_head_checkpoint,
)
from fluke_model.orca_data import OrcaManifestRow, manifest_stats, read_jsonl_manifest  # noqa: E402
from fluke_model.retrieval_eval import evaluate_retrieval  # noqa: E402
from fluke_model.trainable import BalancedBatchSampler, batch_hard_triplet_loss  # noqa: E402
from fluke_model.training_utils import build_scheduler, select_device  # noqa: E402


def evaluate_head(
    head: torch.nn.Module,
    cache_features: np.ndarray,
    cache_paths: list[str],
    train_rows: list[OrcaManifestRow],
    eval_rows: list[OrcaManifestRow],
    *,
    device: torch.device,
    embedder_name: str,
) -> dict | None:
    """Project cached features through the head, then run retrieval eval."""
    if not eval_rows:
        return None

    path_to_idx = {p: i for i, p in enumerate(cache_paths)}
    train_idx = np.array([path_to_idx[r.path] for r in train_rows], dtype=np.int64)
    eval_idx = np.array([path_to_idx[r.path] for r in eval_rows], dtype=np.int64)

    train_proj = project_cached_features(head, cache_features[train_idx], device=device)
    eval_proj = project_cached_features(head, cache_features[eval_idx], device=device)
    return evaluate_retrieval(
        train_proj,
        train_rows,
        eval_proj,
        eval_rows,
        embedder_name=f"{embedder_name}-head",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a head on cached MiewID features.")
    parser.add_argument(
        "--features", default=str(REPO_ROOT / "artifacts/miewid_features/orca_all.npz")
    )
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "data/manifests/orca_all_splits"))
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-bn", action="store_true", help="Disable BatchNorm in MLP head.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--identities-per-batch", type=int, default=8)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/heads/miewid"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results/orca"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cache = load_cached_features(args.features)
    if cache.features.shape[1] != MIEWID_FEATURE_DIM:
        print(
            f"  warning: cache dim {cache.features.shape[1]} != expected {MIEWID_FEATURE_DIM}",
            file=sys.stderr,
        )

    splits_dir = Path(args.splits_dir)
    train_rows = read_jsonl_manifest(splits_dir / "train.jsonl")
    val_path = splits_dir / "val.jsonl"
    val_rows = read_jsonl_manifest(val_path) if val_path.exists() else []
    test_path = splits_dir / "test.jsonl"
    test_rows = read_jsonl_manifest(test_path) if test_path.exists() else []

    if not train_rows:
        print("No training rows found in splits dir.", file=sys.stderr)
        return 2

    device = select_device(args.device)
    print(f"Device: {device}")
    print(
        f"Train rows: {len(train_rows)} | val rows: {len(val_rows)} | test rows: {len(test_rows)}"
    )

    head_kwargs: dict = {"in_features": cache.features.shape[1], "embed_dim": args.embed_dim}
    if args.head == "mlp":
        head_kwargs.update(
            {"hidden_dim": args.hidden_dim, "dropout": args.dropout, "use_bn": not args.no_bn}
        )
    head = build_head(args.head, **head_kwargs).to(device)

    dataset = CachedFeatureDataset(cache, train_rows)
    labels = [dataset.label_to_idx[row.individual_id] for row in dataset.rows]
    try:
        sampler = BalancedBatchSampler(
            labels,
            identities_per_batch=args.identities_per_batch,
            images_per_identity=args.images_per_identity,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"Cannot build balanced batches: {exc}", file=sys.stderr)
        return 3

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        kind=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

    started = time.time()
    history: list[dict] = []
    best_top1 = -1.0
    epochs_since_best = 0
    run_name = (
        args.run_name
        or f"miewid-{args.head}-head-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    checkpoint_path = Path(args.out_dir) / run_name / "head.pt"

    metadata = HeadCheckpointMetadata(
        embedder_name=cache.embedder_name,
        embedder_dim=cache.features.shape[1],
        head_kind=args.head,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim if args.head == "mlp" else None,
        dropout=args.dropout if args.head == "mlp" else None,
        use_bn=(not args.no_bn) if args.head == "mlp" else None,
        image_size=cache.image_size,
    )

    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses: list[float] = []
        for features, labels_batch, _paths in loader:
            features = features.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = head(features)
            loss = batch_hard_triplet_loss(embeddings, labels_batch, margin=args.margin)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        val_report = evaluate_head(
            head,
            cache.features,
            cache.paths,
            train_rows,
            val_rows,
            device=device,
            embedder_name=cache.embedder_name,
        )
        metrics = val_report["metrics"] if val_report else {}
        epoch_report = {
            "epoch": epoch,
            "loss": sum(losses) / max(len(losses), 1),
            "lr": current_lr,
            "val_metrics": metrics,
        }
        history.append(epoch_report)
        top1 = metrics.get("top_1", -1.0)
        if top1 > best_top1:
            best_top1 = top1
            epochs_since_best = 0
            save_head_checkpoint(checkpoint_path, head, metadata, epoch=epoch, metrics=metrics)
        else:
            epochs_since_best += 1
        print(json.dumps(epoch_report))

        if (
            val_rows
            and args.early_stop_patience > 0
            and epochs_since_best >= args.early_stop_patience
        ):
            print(
                f"Early stop: val top_1 has not improved for {args.early_stop_patience} epochs "
                f"(best={best_top1:.4f} at epoch {epoch - epochs_since_best})."
            )
            stopped_early = True
            break

    test_metrics: dict = {}
    if test_rows:
        # Reload only tensors and primitive metadata from the locally created checkpoint.
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
            head.load_state_dict(payload["head_state"])
        test_report = evaluate_head(
            head,
            cache.features,
            cache.paths,
            train_rows,
            test_rows,
            device=device,
            embedder_name=cache.embedder_name,
        )
        if test_report is not None:
            test_metrics = test_report["metrics"]

    report = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "embedder": cache.embedder_name,
        "head": args.head,
        "head_kwargs": head_kwargs,
        "splits_dir": str(splits_dir),
        "split_seed": args.seed,
        "train_stats": manifest_stats(train_rows),
        "val_stats": manifest_stats(val_rows),
        "test_stats": manifest_stats(test_rows),
        "history": history,
        "best_val_top_1": best_top1,
        "test_metrics": test_metrics,
        "stopped_early": stopped_early,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "early_stop_patience": args.early_stop_patience,
        "wall_clock_seconds": time.time() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    results_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote head checkpoint: {checkpoint_path}")
    print(f"Wrote results: {results_path}")
    if test_metrics:
        print(
            f"TEST: top-1={test_metrics.get('top_1', 0):.3f} "
            f"top-3={test_metrics.get('top_3', 0):.3f} "
            f"mrr={test_metrics.get('mrr', 0):.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
