#!/usr/bin/env python3
"""Build the synthetic rights-attested reference index for container CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.ci_fixture import build_ci_reference_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    build_ci_reference_index(args.out_dir, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
