"""Process-wide cache for heavy objects like loaded model pipelines.

RunPod reuses the same Python process for multiple requests on the same
worker instance, so module-level state survives between invocations. This
module provides a thread-safe lazy loader keyed by a string (typically the
workflow name + a fingerprint of the model paths), so the real FLUX.2 /
Wan 2.2 pipelines can be loaded once and reused on subsequent requests.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from .utils.logging import get_logger


logger = get_logger(__name__)


_cache: dict[str, Any] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def get_or_load(key: str, loader: Callable[[], Any]) -> Any:
    """Return a cached object for ``key`` or build it via ``loader`` once.

    Thread-safe: concurrent requests for the same key will block until the
    first load completes, then all get the same instance.
    """
    cached = _cache.get(key)
    if cached is not None:
        logger.info("model cache hit: %s", key)
        return cached

    lock = _get_lock(key)
    with lock:
        cached = _cache.get(key)
        if cached is not None:
            logger.info("model cache hit (after lock): %s", key)
            return cached
        logger.info("model cache miss: loading %s", key)
        obj = loader()
        _cache[key] = obj
        return obj


def invalidate(key: str) -> None:
    _cache.pop(key, None)


def clear() -> None:
    _cache.clear()


def fingerprint(model_paths: dict[str, str]) -> str:
    """Small stable fingerprint for a model_paths dict, used as part of the key."""
    items = sorted(model_paths.items())
    return "|".join(f"{k}={v}" for k, v in items)
