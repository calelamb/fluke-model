#!/usr/bin/env python3
"""Export exact verified catalog resources for a future iOS bundle install."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.mobile_catalog import manifest_payload  # noqa: E402
from fluke_model.mobile_catalog_export import export_verified_mobile_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reverify and atomically export exact iOS IdentifierCatalog resources"
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--app-build", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export_verified_mobile_catalog(
            args.release_dir, args.output_dir, app_build=args.app_build
        )
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(manifest_payload(manifest), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
