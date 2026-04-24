"""RunPod Serverless handler for generation worker."""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import config, storage
from .generators.flux2_reactor import generate_flux2_reactor
from .generators.wan22_i2v import generate_wan22_i2v
from .model_paths import (
    InvalidLoRANameError,
    LoRANotFoundError,
    ModelNotFoundError,
    verify_loras,
    verify_required_models,
)
from .schemas import FileMetadata, GenerateRequest
from .utils.base64_utils import Base64DecodeError, decode_base64_file
from .utils.images import read_image_size
from .utils.logging import get_logger
from .utils.video import probe_video


logger = get_logger("generation-worker.handler")


SUPPORTED_ACTIONS = {"health", "generate"}
SUPPORTED_WORKFLOWS = {"flux2_reactor", "wan22_i2v"}


def _error(code: str, message: str, details: dict | None = None) -> dict:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def _health() -> dict:
    return {"ok": True, "service": "generation-worker"}


def _handle_generate(raw_input: dict, job_id: str | None) -> dict:
    # 5.1 Validate request
    try:
        request = GenerateRequest(**raw_input)
    except ValidationError as exc:
        return _error("INVALID_REQUEST", "request validation failed", {"errors": exc.errors()})

    # 5.2 Validate workflow
    if request.workflow_type not in SUPPORTED_WORKFLOWS:
        return _error(
            "UNSUPPORTED_WORKFLOW",
            f"workflow_type {request.workflow_type!r} is not supported",
            {"supported": sorted(SUPPORTED_WORKFLOWS)},
        )

    # 5.3 Verify required models
    try:
        model_paths = verify_required_models(request.workflow_type)
    except ModelNotFoundError as exc:
        return _error(
            "MODEL_NOT_FOUND",
            str(exc),
            {"key": exc.key, "path": str(exc.path)},
        )

    # 5.4 Verify LoRAs
    try:
        verify_loras(request.loras)
    except InvalidLoRANameError as exc:
        return _error(
            "INVALID_REQUEST",
            str(exc),
            {"name": exc.lora_name, "reason": exc.reason},
        )
    except LoRANotFoundError as exc:
        return _error(
            "LORA_NOT_FOUND",
            str(exc),
            {"name": exc.name, "path": str(exc.path)},
        )

    # 5.5 wan22_i2v requires input_image_base64
    if request.workflow_type == "wan22_i2v" and not request.input_image_base64:
        return _error(
            "INPUT_IMAGE_REQUIRED",
            "wan22_i2v requires input_image_base64",
        )

    # Early validation of base64 payloads (do not decode the whole file yet if not needed,
    # but verify decoding succeeds so generators can rely on them).
    if request.input_image_base64 is not None:
        try:
            decode_base64_file(request.input_image_base64)
        except Base64DecodeError as exc:
            return _error("INVALID_REQUEST", f"input_image_base64: {exc}")
    if request.face_image_base64 is not None:
        try:
            decode_base64_file(request.face_image_base64)
        except Base64DecodeError as exc:
            return _error("FACE_IMAGE_INVALID", f"face_image_base64: {exc}")

    created_at = storage.utc_now_iso()
    file_id = storage.create_file_id()

    # 5.6 Run generator
    try:
        if request.workflow_type == "flux2_reactor":
            tmp_output = generate_flux2_reactor(request, model_paths)
            result_type = "image"
            mime_type = "image/png"
        else:  # wan22_i2v
            tmp_output = generate_wan22_i2v(request, model_paths)
            result_type = "video"
            mime_type = "video/mp4"
    except Exception as exc:  # noqa: BLE001
        logger.exception("generation failed")
        return _error("GENERATION_FAILED", str(exc), {"workflow_type": request.workflow_type})

    # 5.7 Save result
    try:
        if result_type == "image":
            abs_path, relative_path = storage.save_generated_image(tmp_output, file_id)
        else:
            abs_path, relative_path = storage.save_generated_video(tmp_output, file_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("save failed")
        return _error("SAVE_FAILED", str(exc))

    # 5.8 Preview
    try:
        if result_type == "image":
            _, preview_rel = storage.create_preview_for_image(abs_path, file_id)
        else:
            _, preview_rel = storage.create_preview_for_video(abs_path, file_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("preview failed: %s", exc)
        preview_rel = None

    # Gather dimensions / video info
    width = request.width
    height = request.height
    frames = request.frames
    fps = request.fps
    duration_sec: float | None = None

    try:
        if result_type == "image":
            w, h = read_image_size(abs_path)
            width = width or w
            height = height or h
        else:
            probe = probe_video(abs_path)
            width = width or probe.get("width")
            height = height or probe.get("height")
            frames = frames or probe.get("frames")
            fps = fps or probe.get("fps")
            duration_sec = probe.get("duration_sec")
    except Exception as exc:  # noqa: BLE001
        logger.warning("probe failed: %s", exc)

    try:
        size_bytes = abs_path.stat().st_size
    except OSError:
        size_bytes = 0

    filename = abs_path.name
    updated_at = storage.utc_now_iso()

    # 5.9 Metadata
    metadata = FileMetadata(
        id=file_id,
        job_id=job_id,
        type=result_type,  # type: ignore[arg-type]
        workflow_type=request.workflow_type,
        status="completed",
        created_at=created_at,
        updated_at=updated_at,
        filename=filename,
        relative_path=relative_path,
        preview_relative_path=preview_rel,
        metadata_relative_path=None,
        mime_type=mime_type,
        size_bytes=size_bytes,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        duration_sec=duration_sec,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
        loras=request.loras,
        generation_params=request.generation_params,
        model_paths=model_paths,
    )

    # 5.10 Save metadata (first pass to get its relative path, then rewrite including itself)
    try:
        metadata_dict = metadata.model_dump()
        _, metadata_rel = storage.save_metadata(metadata_dict, file_id)
        metadata.metadata_relative_path = metadata_rel
        metadata_dict = metadata.model_dump()
        storage.save_metadata(metadata_dict, file_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("metadata save failed")
        return _error("SAVE_FAILED", f"metadata: {exc}")

    # Cleanup tmp file (best-effort)
    try:
        Path(tmp_output).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass

    # 5.11 Return JSON with paths
    return {
        "ok": True,
        "id": file_id,
        "type": result_type,
        "workflow_type": request.workflow_type,
        "filename": filename,
        "relative_path": relative_path,
        "preview_relative_path": preview_rel,
        "metadata_relative_path": metadata.metadata_relative_path,
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless entry point."""
    try:
        config.ensure_runtime_dirs()
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to ensure runtime dirs")
        return _error("UNKNOWN_ERROR", f"runtime dirs: {exc}")

    try:
        raw_input = (event or {}).get("input") or {}
        job_id = (event or {}).get("id")

        if not isinstance(raw_input, dict):
            return _error("INVALID_REQUEST", "event.input must be an object")

        action = raw_input.get("action", "generate")

        if action == "health":
            return _health()

        if action not in SUPPORTED_ACTIONS:
            return _error(
                "INVALID_ACTION",
                f"unknown action: {action!r}",
                {"supported": sorted(SUPPORTED_ACTIONS)},
            )

        # action == "generate"
        return _handle_generate(raw_input, job_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("unhandled error")
        return _error(
            "UNKNOWN_ERROR",
            str(exc),
            {"traceback": traceback.format_exc()},
        )


if __name__ == "__main__":
    import runpod  # type: ignore

    runpod.serverless.start({"handler": handler})
