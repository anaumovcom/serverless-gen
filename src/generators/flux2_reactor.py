"""FLUX.2 + ReActor generator (MVP stub).

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
from ..utils.images import write_placeholder_png
from ..utils.logging import get_logger


logger = get_logger(__name__)


def _load_pipeline(model_paths: dict[str, str]) -> Any:
    """Load FLUX.2 + ReActor pipeline. MVP stub: returns a marker dict.

    Replace the body of this function with real model loading (diffusers,
    safetensors, insightface, etc). The result is cached by
    :func:`get_pipeline`, so this function is called **at most once per
    worker process per unique set of model paths**.
    """
    logger.info("loading FLUX.2 + ReActor pipeline (stub) with paths=%s", model_paths)
    # Real implementation example (pseudo):
    #   pipe = FluxPipeline.from_single_file(model_paths["flux2_diffusion"], ...)
    #   pipe.to("cuda")
    #   face_analyser = insightface.app.FaceAnalysis(...)
    #   swapper = insightface.model_zoo.get_model(model_paths["inswapper_128"])
    #   return {"pipe": pipe, "face_analyser": face_analyser, "swapper": swapper}
    return {"loaded": True, "model_paths": dict(model_paths)}


def get_pipeline(model_paths: dict[str, str]) -> Any:
    key = "flux2_reactor::" + model_cache.fingerprint(model_paths)
    return model_cache.get_or_load(key, lambda: _load_pipeline(model_paths))


def generate_flux2_reactor(request: GenerateRequest, model_paths: dict[str, str]) -> Path:
    """Generate an image with FLUX.2, optionally apply ReActor face swap.

    MVP: writes a placeholder PNG to the tmp generation directory and returns its path.
    Real implementation should be plugged in here without changing the handler.
    """
    config.TMP_GENERATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load (or reuse cached) pipeline.
    pipeline = get_pipeline(model_paths)
    logger.info("flux2_reactor using pipeline id=%s", id(pipeline))

    width = request.width or 1024
    height = request.height or 1024

    tmp_name = f"flux2_{secrets.token_hex(6)}.png"
    tmp_path = config.TMP_GENERATION_DIR / tmp_name

    text = f"FLUX.2 MVP\nprompt: {request.prompt}"
    if request.face_image_base64:
        text += "\n[face swap requested]"
    if request.loras:
        text += "\nLoRAs: " + ", ".join(l.name for l in request.loras)

    logger.info("flux2_reactor MVP stub: writing placeholder %s", tmp_path)
    write_placeholder_png(tmp_path, text, width=width, height=height)
    return tmp_path
