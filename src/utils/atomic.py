"""Atomic file writes."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_file_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_json_atomic(path: Path, data: dict) -> None:
    """Serialize ``data`` as UTF-8 JSON and write atomically."""
    payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    write_file_atomic(path, payload)
