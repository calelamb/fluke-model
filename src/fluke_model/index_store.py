"""Atomic publication of immutable, versioned reference indexes."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_INDEX_FILES = frozenset({"index.faiss", "metadata.json", "index_info.json", "rights.json"})


@dataclass(frozen=True)
class AtomicIndexStore:
    root: Path

    @property
    def versions_dir(self) -> Path:
        return self.root / "versions"

    def create_version(self, version: str) -> Path:
        self._validate_version(version)
        path = self.versions_dir / version
        path.mkdir(parents=True, exist_ok=False)
        return path

    def publish(self, version_dir: Path) -> None:
        resolved = version_dir.resolve()
        versions = self.versions_dir.resolve()
        if resolved.parent != versions:
            raise ValueError("index version must be staged inside the versions directory")
        self._validate_version(resolved.name)
        missing = REQUIRED_INDEX_FILES.difference(path.name for path in resolved.iterdir())
        if missing:
            raise ValueError("index version is missing required index files")

        self.root.mkdir(parents=True, exist_ok=True)
        pointer = self.root / "current.json"
        temporary = self.root / f".current-{uuid4().hex}.json"
        payload = json.dumps({"version": resolved.name}, separators=(",", ":"))
        with temporary.open("w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)

    def current_version_dir(self) -> Path:
        payload = json.loads((self.root / "current.json").read_text())
        version = payload.get("version")
        self._validate_version(version)
        path = self.versions_dir / version
        if not path.is_dir():
            raise ValueError("published index version is unavailable")
        return path

    @staticmethod
    def _validate_version(version: object) -> None:
        if not isinstance(version, str) or VERSION_PATTERN.fullmatch(version) is None:
            raise ValueError("invalid index version")
