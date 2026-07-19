#!/usr/bin/env python3
"""Verify every non-negotiable gate for one fixed-layout mobile release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.mobile_release import (  # noqa: E402
    REPORT_FILENAME,
    report_payload,
    validate_report_destination,
    verify_mobile_release_directory,
    write_mobile_release_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail closed unless every mobile model release gate passes"
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help=f"report destination (default: RELEASE_DIR/{REPORT_FILENAME})",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    report_path = args.report or args.release_dir / REPORT_FILENAME
    report = verify_mobile_release_directory(args.release_dir)
    try:
        validate_report_destination(args.release_dir, report_path)
        write_mobile_release_report(report_path, report)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        print(f"error: release report could not be written: {error}", file=sys.stderr)
        print(json.dumps(report_payload(report), allow_nan=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report_payload(report), allow_nan=False, indent=2, sort_keys=True))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
