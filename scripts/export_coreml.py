#!/usr/bin/env python3
"""Export the pinned local DINOv2 artifact as an iOS 17 Core ML package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fluke_model.coreml_artifact import CoreMLExportError, publish_coreml_export
from fluke_model.model_artifact import ModelArtifactError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing non-empty output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed atomic exporter CLI."""
    arguments = _parser().parse_args(argv)
    try:
        metadata = publish_coreml_export(
            arguments.artifact_dir,
            arguments.output_dir,
            replace=arguments.replace,
        )
    except (CoreMLExportError, ModelArtifactError) as error:
        print(f"Core ML export failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "model_sha256": metadata.model_sha256,
                "package_sha256": metadata.package_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
