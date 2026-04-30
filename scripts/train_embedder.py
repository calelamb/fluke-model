#!/usr/bin/env python3
"""Train a beginner-friendly metric-learning orca embedder.

This is deliberately plain PyTorch: dataset -> dataloader -> forward pass ->
triplet loss -> optimizer -> validation retrieval metrics. It is meant to be
readable before it is clever.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.orca_data import OrcaManifestRow, manifest_stats, read_jsonl_manifest  # noqa: E402
from fluke_model.retrieval_eval import evaluate_retrieval  # noqa: E402
from fluke_model.trainable import (  # noqa: E402
    BalancedBatchSampler,
    CheckpointMetadata,
    EmbedderNet,
    OrcaImageDataset,
    batch_hard_triplet_loss,
    embed_rows,
    save_checkpoint,
)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tiny_overfit_subset(rows: list[OrcaManifestRow], individuals: int = 3) -> list[OrcaManifestRow]:
    by_id: dict[str, list[OrcaManifestRow]] = {}
    for row in rows:
        by_id.setdefault(row.individual_id, []).append(row)
    selected: list[OrcaManifestRow] = []
    for _individual_id, group in sorted(by_id.items(), key=lambda item: (-len(item[1]), item[0]))[:individuals]:
        selected.extend(group[: min(4, len(group))])
    return selected


def run_validation(
    model: EmbedderNet,
    train_rows: list[OrcaManifestRow],
    val_rows: list[OrcaManifestRow],
    *,
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> dict | None:
    if not val_rows:
        return None
    model.eval()
    reference = embed_rows(model, train_rows, image_size=image_size, device=device, batch_size=batch_size)
    queries = embed_rows(model, val_rows, image_size=image_size, device=device, batch_size=batch_size)
    return evaluate_retrieval(
        reference,
        train_rows,
        queries,
        val_rows,
        embedder_name=f"trained-{model.backbone_name}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an orca metric-learning embedder.")
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "data/manifests/happywhale_orca_splits"))
    parser.add_argument("--backbone", default="resnet50", help="Any timm image model, e.g. resnet50 or convnext_tiny.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--identities-per-batch", type=int, default=4)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--overfit-tiny", action="store_true", help="Use a tiny repeated-ID subset for a learning smoke test.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/models/orca"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results/orca"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    splits_dir = Path(args.splits_dir)
    train_rows = read_jsonl_manifest(splits_dir / "train.jsonl")
    val_rows = read_jsonl_manifest(splits_dir / "val.jsonl") if (splits_dir / "val.jsonl").exists() else []
    if args.overfit_tiny:
        train_rows = tiny_overfit_subset(train_rows)
        val_rows = []
        print(f"Overfit-tiny mode: {len(train_rows)} train images across {manifest_stats(train_rows)['individuals']} IDs")

    if not train_rows:
        print("No training rows found. Build and split the Happywhale orca manifest first.", file=sys.stderr)
        return 2

    device = select_device(args.device)
    print(f"Device: {device}")
    print(f"Train rows: {len(train_rows)} | val rows: {len(val_rows)}")
    model = EmbedderNet(
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        pretrained=not args.no_pretrained,
    ).to(device)

    dataset = OrcaImageDataset(train_rows, image_size=args.image_size, train=True)
    labels = [dataset.label_to_idx[row.individual_id] for row in dataset.rows]
    try:
        sampler = BalancedBatchSampler(
            labels,
            identities_per_batch=args.identities_per_batch,
            images_per_identity=args.images_per_identity,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"Cannot build balanced batches: {e}", file=sys.stderr)
        print("Try lowering --identities-per-batch or rebuild splits with more repeated images.", file=sys.stderr)
        return 3

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    started = time.time()
    history: list[dict] = []
    best_top1 = -1.0
    run_name = args.run_name or f"{args.backbone}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    checkpoint_path = Path(args.out_dir) / run_name / "model.pt"

    metadata = CheckpointMetadata(
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        image_size=args.image_size,
        source_dataset="happywhale_orca",
        split_seed=args.seed,
        num_train_images=len(train_rows),
        num_train_individuals=manifest_stats(train_rows)["individuals"],
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, labels_batch, _paths in pbar:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = batch_hard_triplet_loss(embeddings, labels_batch, margin=args.margin)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

        val_report = run_validation(
            model,
            train_rows,
            val_rows,
            image_size=args.image_size,
            device=device,
            batch_size=args.identities_per_batch * args.images_per_identity,
        )
        metrics = val_report["metrics"] if val_report else {}
        epoch_report = {
            "epoch": epoch,
            "loss": sum(losses) / max(len(losses), 1),
            "val_metrics": metrics,
        }
        history.append(epoch_report)
        top1 = metrics.get("top_1", -1.0)
        if top1 >= best_top1:
            best_top1 = top1
            save_checkpoint(checkpoint_path, model, metadata, epoch=epoch, metrics=metrics)
        print(json.dumps(epoch_report, indent=2))

    report = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "backbone": args.backbone,
        "embed_dim": args.embed_dim,
        "image_size": args.image_size,
        "dataset": "happywhale_orca",
        "split_seed": args.seed,
        "train_stats": manifest_stats(train_rows),
        "val_stats": manifest_stats(val_rows),
        "history": history,
        "wall_clock_seconds": time.time() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_name}.json"
    results_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote results: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
