"""Wan 2.2 image-to-video generator (MVP stub).

The real pipeline is loaded lazily and cached in-process via
:mod:`src.model_cache`, so it is kept in memory between requests on the
same worker and is not reloaded on every invocation.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from .. import config
from .. import model_cache
from ..schemas import GenerateRequest
from ..utils.logging import get_logger
from ..utils.video import write_placeholder_mp4


logger = get_logger(__name__)


def _load_pipeline(model_paths: dict[str, str]) -> Any:
    """Load Wan 2.2 I2V pipeline. MVP stub: returns a marker dict.

    Replace the body with real model loading; it is invoked at most once
    per worker process per unique set of model paths.
    """
    logger.info("loading Wan 2.2 I2V pipeline (stub) with paths=%s", model_paths)
    return {"loaded": True, "model_paths": dict(model_paths)}


def get_pipeline(model_paths: dict[str, str]) -> Any:
    key = "wan22_i2v::" + model_cache.fingerprint(model_paths)
    return model_cache.get_or_load(key, lambda: _load_pipeline(model_paths))


def generate_wan22_i2v(request: GenerateRequest, model_paths: dict[str, str]) -> Path:
    """Generate a video using Wan 2.2 I2V.

    MVP: writes a placeholder MP4 with OpenCV and returns its path.
    Real implementation should be plugged in here without changing the handler.
    """
    if not request.input_image_base64:
        # Handler should have caught this; double-check at generator boundary.
        raise ValueError("input_image_base64 is required for wan22_i2v")

    config.TMP_GENERATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load (or reuse cached) pipeline.
    pipeline = get_pipeline(model_paths)
    logger.info("wan22_i2v using pipeline id=%s", id(pipeline))

    width = request.width or 832
    height = request.height or 480
    frames = request.frames or 16
    fps = request.fps or 16

    tmp_name = f"wan22_{secrets.token_hex(6)}.mp4"
    tmp_path = config.TMP_GENERATION_DIR / tmp_name

    logger.info("wan22_i2v MVP stub: writing placeholder %s", tmp_path)
    write_placeholder_mp4(tmp_path, width=width, height=height, frames=frames, fps=fps)
    return tmp_path
