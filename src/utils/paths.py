"""Safe path resolution utilities."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafePathError(ValueError):
    """Raised when a path is considered unsafe (traversal, absolute, etc.)."""


def safe_resolve(base_dir: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` against ``base_dir`` while preventing traversal.

    Rules:
    - absolute paths are rejected
    - ``..`` segments are rejected
    - the resolved path must remain inside ``base_dir``
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise UnsafePathError("relative_path must be a non-empty string")

    # Reject absolute paths (posix or windows)
    if PurePosixPath(relative_path).is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise UnsafePathError(f"absolute path is not allowed: {relative_path!r}")
    if relative_path.startswith(("/", "\\")):
        raise UnsafePathError(f"absolute path is not allowed: {relative_path!r}")

    # Reject traversal segments
    parts = relative_path.replace("\\", "/").split("/")
    for part in parts:
        if part == "..":
            raise UnsafePathError(f"path traversal is not allowed: {relative_path!r}")

    base_resolved = base_dir.resolve()
    candidate = (base_resolved / relative_path).resolve()

    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise UnsafePathError(
            f"path escapes base directory: {relative_path!r}"
        ) from exc

    return candidate


def relative_to_storage(absolute_path: Path, storage_base: Path) -> str:
    """Return POSIX-style path of ``absolute_path`` relative to ``storage_base``."""
    rel = absolute_path.resolve().relative_to(storage_base.resolve())
    return rel.as_posix()
