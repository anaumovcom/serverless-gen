"""Model path registry and verification (no downloads)."""
from __future__ import annotations

from pathlib import Path

from . import config
from .schemas import LoRARequest


class ModelNotFoundError(FileNotFoundError):
    def __init__(self, key: str, path: Path):
        super().__init__(f"model not found: {key} -> {path}")
        self.key = key
        self.path = path


class LoRANotFoundError(FileNotFoundError):
    def __init__(self, name: str, path: Path):
        super().__init__(f"LoRA not found: {name} -> {path}")
        self.name = name
        self.path = path


class InvalidLoRANameError(ValueError):
    def __init__(self, name: str, reason: str):
        super().__init__(f"invalid LoRA name {name!r}: {reason}")
        self.lora_name = name
        self.reason = reason


# Required models per workflow
_REQUIRED_MODELS: dict[str, dict[str, Path]] = {
    "flux2_reactor": {
        "flux2_diffusion": config.DIFFUSION_MODELS_DIR / "flux2_dev_fp8mixed.safetensors",
        "flux2_text_encoder": config.TEXT_ENCODERS_DIR / "mistral_3_small_flux2_bf16.safetensors",
        "flux2_vae": config.VAE_DIR / "flux2-vae.safetensors",
        "insightface_buffalo_l": config.INSIGHTFACE_DIR / "buffalo_l.zip",
        "inswapper_128": config.INSIGHTFACE_DIR / "inswapper_128.onnx",
    },
    "wan22_i2v": {
        "wan22_ti2v_5b": config.DIFFUSION_MODELS_DIR / "wan2.2_ti2v_5B_fp16.safetensors",
        "wan22_vae": config.VAE_DIR / "wan2.2_vae.safetensors",
        "wan22_text_encoder": config.TEXT_ENCODERS_DIR / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    },
}


def get_model_base_dir() -> Path:
    return config.MODEL_BASE_DIR


def get_required_model_paths(workflow_type: str) -> dict[str, Path]:
    if workflow_type not in _REQUIRED_MODELS:
        raise KeyError(f"unknown workflow_type: {workflow_type}")
    return dict(_REQUIRED_MODELS[workflow_type])


def verify_required_models(workflow_type: str) -> dict[str, str]:
    """Verify required model files exist. Raises ``ModelNotFoundError`` if any is missing.

    Returns a dict of key -> absolute path string (for metadata).
    """
    paths = get_required_model_paths(workflow_type)
    resolved: dict[str, str] = {}
    for key, path in paths.items():
        if not path.exists() or not path.is_file():
            raise ModelNotFoundError(key, path)
        resolved[key] = str(path)
    return resolved


def _validate_lora_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise InvalidLoRANameError(str(name), "name must be a non-empty string")
    if name.startswith("/") or name.startswith("\\"):
        raise InvalidLoRANameError(name, "absolute paths are not allowed")
    if ":" in name and len(name) > 1 and name[1] == ":":
        raise InvalidLoRANameError(name, "absolute paths are not allowed")
    if "/" in name or "\\" in name:
        raise InvalidLoRANameError(name, "subdirectories are not allowed")
    if ".." in name.split("/") or ".." in name.split("\\") or name == "..":
        raise InvalidLoRANameError(name, "'..' is not allowed")


def verify_loras(loras: list[LoRARequest]) -> list[str]:
    """Verify that each LoRA file exists under ``loras`` dir. Returns absolute paths."""
    resolved: list[str] = []
    for lora in loras:
        _validate_lora_name(lora.name)
        path = config.LORAS_DIR / lora.name
        if not path.exists() or not path.is_file():
            raise LoRANotFoundError(lora.name, path)
        resolved.append(str(path))
    return resolved
