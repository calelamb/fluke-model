#!/usr/bin/env python3
"""Build a production mobile release from approved, rights-cleared source evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.mobile_release_builder import BuildOptions, build_mobile_release  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, evaluate, and independently verify a production mobile release"
    )
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--rights", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--manifest-version", required=True)
    parser.add_argument("--minimum-app-build", type=int, required=True)
    parser.add_argument("--maximum-app-build", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = build_mobile_release(
            corpus_manifest_path=args.corpus_manifest,
            corpus_root=args.corpus_root,
            evaluation_plan_path=args.evaluation_plan,
            rights_path=args.rights,
            model_artifact_dir=args.model_artifact,
            model_package_path=args.model_package,
            export_metadata_path=args.model_metadata,
            output_dir=args.output_dir,
            options=BuildOptions(
                manifest_version=args.manifest_version,
                minimum_app_build=args.minimum_app_build,
                maximum_app_build=args.maximum_app_build,
            ),
        )
    except (OSError, RuntimeError, UnicodeError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
