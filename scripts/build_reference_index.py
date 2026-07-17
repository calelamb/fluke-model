#!/usr/bin/env python3
"""Build a rights-attested reference index from a complete rebuild request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.identify_runtime import IdentifierRuntime  # noqa: E402
from fluke_model.index_store import AtomicIndexStore  # noqa: E402
from fluke_model.network import ReferenceImageFetcher  # noqa: E402
from fluke_model.rebuild import ProductionRebuilder  # noqa: E402
from fluke_model.settings import ServiceSettings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a rights-attested Fluke reference index")
    parser.add_argument(
        "--request", required=True, help="JSON containing references and rightsAttestation"
    )
    parser.add_argument("--out-dir", default="artifacts/reference-index")
    parser.add_argument("--model-artifact-dir", type=Path, required=True)
    parser.add_argument("--allowed-host", action="append", required=True)
    args = parser.parse_args()

    settings = ServiceSettings(
        api_key="local-index-build-command-is-not-a-service-key",
        index_dir=Path(args.out_dir),
        allowed_reference_hosts=frozenset(host.lower() for host in args.allowed_host),
        model_artifact_dir=args.model_artifact_dir,
    )
    store = AtomicIndexStore(settings.index_dir)
    runtime = IdentifierRuntime(
        index_store=store,
        model_artifact_dir=settings.model_artifact_dir,
    )
    fetcher = ReferenceImageFetcher(
        allowed_hosts=settings.allowed_reference_hosts,
        max_bytes=settings.max_image_bytes,
        max_pixels=settings.max_image_pixels,
    )
    rebuilder = ProductionRebuilder(
        settings=settings, runtime=runtime, store=store, fetcher=fetcher
    )
    payload = json.loads(Path(args.request).read_text())
    print(json.dumps(rebuilder(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
