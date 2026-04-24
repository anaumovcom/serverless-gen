"""Configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path


RUNPOD_VOLUME_DIR: Path = Path(os.environ.get("RUNPOD_VOLUME_DIR", "/runpod-volume"))
MODEL_BASE_DIR: Path = Path(os.environ.get("MODEL_BASE_DIR", "/runpod-volume/models"))
STORAGE_BASE_DIR: Path = Path(os.environ.get("STORAGE_BASE_DIR", "/runpod-volume/storage"))
TMP_DIR: Path = Path(os.environ.get("TMP_DIR", "/runpod-volume/tmp"))

# Subdirectories under storage
IMAGES_DIR: Path = STORAGE_BASE_DIR / "images"
VIDEOS_DIR: Path = STORAGE_BASE_DIR / "videos"
PREVIEWS_DIR: Path = STORAGE_BASE_DIR / "previews"
METADATA_DIR: Path = STORAGE_BASE_DIR / "metadata"
INPUTS_DIR: Path = STORAGE_BASE_DIR / "inputs"

# Model subdirectories
DIFFUSION_MODELS_DIR: Path = MODEL_BASE_DIR / "diffusion_models"
TEXT_ENCODERS_DIR: Path = MODEL_BASE_DIR / "text_encoders"
VAE_DIR: Path = MODEL_BASE_DIR / "vae"
INSIGHTFACE_DIR: Path = MODEL_BASE_DIR / "insightface"
LORAS_DIR: Path = MODEL_BASE_DIR / "loras"
CHECKPOINTS_DIR: Path = MODEL_BASE_DIR / "checkpoints"
CUSTOM_DIR: Path = MODEL_BASE_DIR / "custom"

# Temp dir used by generators
TMP_GENERATION_DIR: Path = TMP_DIR / "generation"


def ensure_runtime_dirs() -> None:
    """Create storage/tmp directories if they don't exist. Models are NOT created."""
    for d in (
        STORAGE_BASE_DIR,
        IMAGES_DIR,
        VIDEOS_DIR,
        PREVIEWS_DIR,
        METADATA_DIR,
        INPUTS_DIR,
        TMP_GENERATION_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
