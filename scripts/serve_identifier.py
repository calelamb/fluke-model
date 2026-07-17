#!/usr/bin/env python3
"""Production entrypoint for Fluke's rights-gated identifier service."""

from __future__ import annotations

import os
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
from fluke_model.service import ServiceDependencies, create_app  # noqa: E402
from fluke_model.settings import ServiceSettings  # noqa: E402

settings = ServiceSettings.from_env()
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
rebuilder = ProductionRebuilder(settings=settings, runtime=runtime, store=store, fetcher=fetcher)
app = create_app(settings, ServiceDependencies(runtime=runtime, rebuild=rebuilder))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "4100")))
