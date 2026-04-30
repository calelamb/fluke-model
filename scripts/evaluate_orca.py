#!/usr/bin/env python3
"""Evaluate MiewID and/or a trained Fluke embedder on the same orca split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.embedders import EmbedderUnavailable, load_embedder  # noqa: E402
from fluke_model.orca_data import manifest_stats, read_jsonl_manifest  # noqa: E402
from fluke_model.retrieval_eval import evaluate_retrieval  # noqa: E402
from fluke_model.trainable import embed_rows, load_checkpoint  # noqa: E402


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def embed_frozen(embedder_name: str, rows, batch_size: int) -> tuple[np.ndarray, dict]:
    embedder = load_embedder(embedder_name)
    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(rows), batch_size), desc=f"embed {embedder_name}"):
        chunk = rows[start : start + batch_size]
        vectors.append(embedder.embed_fn(load_images([row.path for row in chunk])))
    return np.concatenate(vectors, axis=0).astype(np.float32), {
        "embedder": embedder.name,
        "embed_dim": embedder.embed_dim,
    }


def append_summary(report: dict, summary_md: Path) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    if not summary_md.exists():
        summary_md.write_text(
            "# Public Orca Evaluation Summary\n\n"
            "| Model | Reference Images | Query Images | Individuals | Top-1 | Top-3 | Top-5 | MRR |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
    with summary_md.open("a") as f:
        for result in report["results"]:
            if result.get("status") == "unavailable":
                continue
            m = result["metrics"]
            f.write(
                f"| {result['model']} "
                f"| {result['n_reference_images']} "
                f"| {result['n_query_images']} "
                f"| {result['n_query_individuals']} "
                f"| {m['top_1']:.1%} "
                f"| {m['top_3']:.1%} "
                f"| {m['top_5']:.1%} "
                f"| {m['mrr']:.3f} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate public-orca identification models.")
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "data/manifests/happywhale_orca_splits"))
    parser.add_argument("--baseline", action="append", default=["miewid-msv3"], help="Frozen embedder baseline; repeatable.")
    parser.add_argument("--skip-baseline", action="store_true", help="Evaluate only the trained checkpoint.")
    parser.add_argument("--include-per-query", action="store_true", help="Persist per-query match records.")
    parser.add_argument("--trained-checkpoint", default="", help="Path to a train_embedder.py checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(REPO_ROOT / "results/orca/evaluation.json"))
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    train_rows = read_jsonl_manifest(splits_dir / "train.jsonl")
    test_rows = read_jsonl_manifest(splits_dir / "test.jsonl")
    if not train_rows or not test_rows:
        print("Train/test splits are empty. Run scripts/split_manifest.py first.", file=sys.stderr)
        return 2

    results: list[dict] = []
    baselines = [] if args.skip_baseline else (args.baseline or [])
    for baseline in baselines:
        try:
            ref, meta = embed_frozen(baseline, train_rows, args.batch_size)
            qry, _ = embed_frozen(baseline, test_rows, args.batch_size)
            result = evaluate_retrieval(
                ref,
                train_rows,
                qry,
                test_rows,
                embedder_name=baseline,
            )
            if not args.include_per_query:
                result.pop("per_query", None)
            result.update({"model": baseline, "metadata": meta})
            results.append(result)
        except EmbedderUnavailable as e:
            results.append({"model": baseline, "status": "unavailable", "error": str(e)})
            print(f"Baseline unavailable: {baseline}: {e}", file=sys.stderr)

    if args.trained_checkpoint:
        device = select_device(args.device)
        model, metadata = load_checkpoint(args.trained_checkpoint, device=device)
        image_size = int(metadata["image_size"])
        ref = embed_rows(model, train_rows, image_size=image_size, device=device, batch_size=args.batch_size)
        qry = embed_rows(model, test_rows, image_size=image_size, device=device, batch_size=args.batch_size)
        result = evaluate_retrieval(
            ref,
            train_rows,
            qry,
            test_rows,
            embedder_name=f"trained-{metadata['backbone']}",
        )
        if not args.include_per_query:
            result.pop("per_query", None)
        result.update({"model": f"trained-{metadata['backbone']}", "metadata": metadata})
        results.append(result)

    report = {
        "dataset": "happywhale_orca",
        "splits_dir": str(splits_dir),
        "train_stats": manifest_stats(train_rows),
        "test_stats": manifest_stats(test_rows),
        "results": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    append_summary(report, out_path.parent / "summary.md")
    print(f"Wrote {out_path}")
    for result in results:
        if result.get("status") == "unavailable":
            print(f"{result['model']}: unavailable")
            continue
        m = result["metrics"]
        print(
            f"{result['model']}: top-1={m['top_1']:.1%} "
            f"top-3={m['top_3']:.1%} top-5={m['top_5']:.1%} mrr={m['mrr']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
