#!/usr/bin/env python3
"""Fetch and verify the exact DINOv2 production artifact during image build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.model_artifact import download_dinov2_artifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    download_dinov2_artifact(args.out_dir)


if __name__ == "__main__":
    main()
