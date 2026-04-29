#!/usr/bin/env python3
"""Leave-one-out evaluation of a frozen embedder on a re-ID manifest.

For every photo in the manifest:
  1. Embed it.
  2. Query the index with that vector, asking for k+1 neighbors.
  3. Drop the self-hit (matched by exact path).
  4. Aggregate the remaining neighbors into per-individual scores (mean-of-top-3).
  5. Record whether the true individual is at rank 1, in top-3, and the reciprocal rank.

Outputs:
  - results/<embedder>-eval.json: full per-query record + summary metrics.
  - Appends to results/summary.md: a single Markdown row.

Subset / synthetic fallbacks:
  --subset N            : evaluate only N photos (sampled deterministically).
  --max-individuals M   : restrict to M individuals (prefers ones with the most photos).
  --synthetic           : skip the manifest entirely and generate a tiny synthetic
                          set of random RGB blobs grouped into 5 individuals. This
                          exists so the pipeline still produces numbers when the
                          real download fails.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.embedders import load_embedder, EmbedderUnavailable  # noqa: E402
from fluke_model.index import build_index, search, aggregate_per_individual  # noqa: E402
from fluke_model.io import load_image, read_manifest, write_json, ManifestRow  # noqa: E402
from fluke_model.metrics import top_k_accuracy, mean_reciprocal_rank  # noqa: E402


def deterministic_subset(rows: list[ManifestRow], n: int, seed: int = 42, max_individuals: int = 0) -> list[ManifestRow]:
    """Pick a deterministic, identity-stratified subset of size ~n.

    Prefers individuals with at least 2 photos so leave-one-out is meaningful.
    """
    by_id: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        by_id[r.individual_id].append(r)
    eligible = {ind: photos for ind, photos in by_id.items() if len(photos) >= 2}
    if not eligible:
        return rows[:n]

    inds = sorted(eligible.keys(), key=lambda i: (-len(eligible[i]), i))
    if max_individuals > 0:
        inds = inds[:max_individuals]

    rng = random.Random(seed)
    selected: list[ManifestRow] = []
    # Round-robin sample one photo per individual until we have n
    pools = {i: list(eligible[i]) for i in inds}
    for pool in pools.values():
        rng.shuffle(pool)
    while len(selected) < n and any(pools.values()):
        for i in list(pools.keys()):
            if not pools[i]:
                continue
            selected.append(pools[i].pop())
            if len(selected) >= n:
                break
    return selected


def synthetic_manifest(n_individuals: int = 5, photos_per_id: int = 8, image_size: int = 224) -> tuple[list[ManifestRow], dict[str, Image.Image]]:
    """Generate a deterministic synthetic dataset for pipeline smoke-tests.

    Each individual gets a unique base color plus random per-photo perturbation;
    embedders should still cluster photos of the same individual together.
    """
    rng = np.random.default_rng(seed=0)
    img_cache: dict[str, Image.Image] = {}
    rows: list[ManifestRow] = []
    for i in range(n_individuals):
        base = rng.integers(low=20, high=235, size=3)
        for j in range(photos_per_id):
            noise = rng.integers(low=-15, high=15, size=3)
            color = np.clip(base + noise, 0, 255).astype(np.uint8)
            arr = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            arr[:, :] = color
            # Add an off-center texture patch unique to the individual
            patch_size = 40
            cy = (image_size // 2) + (i * 5)
            cx = (image_size // 2) + ((i * 13) % 20)
            patch_color = np.clip(255 - color, 0, 255).astype(np.uint8)
            arr[cy : cy + patch_size, cx : cx + patch_size] = patch_color
            img = Image.fromarray(arr)
            key = f"synthetic://individual-{i}/photo-{j}.png"
            img_cache[key] = img
            rows.append(ManifestRow(path=key, individual_id=f"synth_{i}"))
    return rows, img_cache


def evaluate(
    embedder_name: str,
    rows: list[ManifestRow],
    image_loader,
    batch_size: int = 8,
) -> dict:
    embedder = load_embedder(embedder_name)
    print(f"Loaded {embedder_name} (dim={embedder.embed_dim}). Embedding {len(rows)} images...")

    started = time.time()
    all_vecs: list[np.ndarray] = []
    paths: list[str] = []
    individuals: list[str] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="embed"):
        chunk = rows[start : start + batch_size]
        images = [image_loader(r.path) for r in chunk]
        vecs = embedder.embed_fn(images)
        all_vecs.append(vecs)
        for r in chunk:
            paths.append(r.path)
            individuals.append(r.individual_id)

    embeddings = np.concatenate(all_vecs, axis=0).astype(np.float32)
    embed_seconds = time.time() - started
    print(f"  embedding wall-clock: {embed_seconds:.1f}s ({embed_seconds / max(len(rows), 1) * 1000:.0f} ms/img)")

    metadata = [{"path": p, "individual_id": i} for p, i in zip(paths, individuals)]
    bundle = build_index(embeddings, metadata, embedder_name=embedder_name)

    eval_started = time.time()
    predictions: list[list[str]] = []
    truths: list[str] = []
    per_query: list[dict] = []
    n = embeddings.shape[0]
    # Query each row leave-one-out: ask for n neighbors, then drop self.
    k_neighbors = min(n, 50)
    for i in tqdm(range(n), desc="evaluate"):
        q = embeddings[i : i + 1]
        hits = search(bundle, q, k=k_neighbors)
        # Drop self-hit by exact path match
        hits = [(s, m) for (s, m) in hits if m["path"] != paths[i]]
        aggregated = aggregate_per_individual(hits, top_n=3)
        pred_ids = [ind for ind, _ in aggregated]
        predictions.append(pred_ids)
        truths.append(individuals[i])
        per_query.append(
            {
                "path": paths[i],
                "truth": individuals[i],
                "top5": [{"individual_id": ind, "score": s} for ind, s in aggregated[:5]],
            }
        )
    eval_seconds = time.time() - eval_started

    top1 = top_k_accuracy(predictions, truths, k=1)
    top3 = top_k_accuracy(predictions, truths, k=3)
    top5 = top_k_accuracy(predictions, truths, k=5)
    mrr = mean_reciprocal_rank(predictions, truths)

    return {
        "embedder": embedder_name,
        "embed_dim": embedder.embed_dim,
        "n_photos": n,
        "n_individuals": len(set(individuals)),
        "metrics": {
            "top_1": top1,
            "top_3": top3,
            "top_5": top5,
            "mrr": mrr,
        },
        "wall_clock_seconds": {
            "embed": embed_seconds,
            "evaluate": eval_seconds,
            "total": embed_seconds + eval_seconds,
        },
        "per_query": per_query,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def status_for_top3(top3: float) -> str:
    if top3 >= 0.80:
        return "ship-V1"
    if top3 >= 0.60:
        return "ship-V0"
    return "retrain"


def append_summary(report: dict, summary_md: Path) -> None:
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    if not summary_md.exists():
        summary_md.write_text(
            "# M-Model-0 Evaluation Summary\n\n"
            "First numbers for the Fluke zero-shot photo-ID prototype. See "
            "`fluke/docs/specs/m-model-0-prototype.md` for the spec.\n\n"
            "| Embedder | Photos | Individuals | Top-1 | Top-3 | Top-5 | MRR | Wall-clock | Status |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
    m = report["metrics"]
    line = (
        f"| {report['embedder']} "
        f"| {report['n_photos']} "
        f"| {report['n_individuals']} "
        f"| {m['top_1']:.1%} "
        f"| {m['top_3']:.1%} "
        f"| {m['top_5']:.1%} "
        f"| {m['mrr']:.3f} "
        f"| {report['wall_clock_seconds']['total']:.1f}s "
        f"| {status_for_top3(m['top_3'])} |\n"
    )
    with summary_md.open("a") as f:
        f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation of a frozen embedder.")
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data/beluga-id-2022/manifest.csv"))
    parser.add_argument("--subset", type=int, default=100, help="Photos to evaluate (0 = full).")
    parser.add_argument("--max-individuals", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic dataset.")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    image_loader = load_image

    if args.synthetic:
        rows, cache = synthetic_manifest(n_individuals=5, photos_per_id=8)
        image_loader = lambda key: cache[key]  # noqa: E731
        print(f"Using synthetic manifest: {len(rows)} rows, {len(set(r.individual_id for r in rows))} individuals.")
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}", file=sys.stderr)
            print("Falling back to --synthetic.", file=sys.stderr)
            rows, cache = synthetic_manifest(n_individuals=5, photos_per_id=8)
            image_loader = lambda key: cache[key]  # noqa: E731
        else:
            all_rows = read_manifest(manifest_path)
            print(f"Manifest: {manifest_path} ({len(all_rows)} rows)")
            target_n = args.subset if args.subset > 0 else len(all_rows)
            rows = deterministic_subset(all_rows, n=target_n, seed=args.seed, max_individuals=args.max_individuals)
            print(f"Evaluating {len(rows)} photos across {len(set(r.individual_id for r in rows))} individuals.")

    try:
        report = evaluate(
            embedder_name=args.embedder,
            rows=rows,
            image_loader=image_loader,
            batch_size=args.batch_size,
        )
    except EmbedderUnavailable as e:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            results_dir / f"{args.embedder}-UNAVAILABLE.json",
            {
                "embedder": args.embedder,
                "status": "unavailable",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"\nEmbedder '{args.embedder}' unavailable: {e}", file=sys.stderr)
        return 5

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.embedder}-eval.json"
    write_json(out_path, report)
    append_summary(report, results_dir / "summary.md")

    m = report["metrics"]
    print()
    print(f"Results for {args.embedder}:")
    print(f"  photos       : {report['n_photos']}")
    print(f"  individuals  : {report['n_individuals']}")
    print(f"  top-1        : {m['top_1']:.1%}")
    print(f"  top-3        : {m['top_3']:.1%}")
    print(f"  top-5        : {m['top_5']:.1%}")
    print(f"  MRR          : {m['mrr']:.3f}")
    print(f"  status       : {status_for_top3(m['top_3'])}")
    print(f"  wrote        : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
