"""Pydantic data schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoRARequest(BaseModel):
    name: str
    strength_model: float = 1.0
    strength_clip: float = 1.0


class GenerateRequest(BaseModel):
    action: str = "generate"
    workflow_type: Literal["flux2_reactor", "wan22_i2v"]

    prompt: str
    negative_prompt: str | None = None

    seed: int | None = None
    width: int | None = None
    height: int | None = None

    steps: int | None = None
    guidance_scale: float | None = None

    input_image_base64: str | None = None
    face_image_base64: str | None = None

    frames: int | None = None
    fps: int | None = None

    loras: list[LoRARequest] = Field(default_factory=list)
    generation_params: dict = Field(default_factory=dict)


class FileMetadata(BaseModel):
    id: str
    job_id: str | None = None

    type: Literal["image", "video"]
    workflow_type: str
    status: str

    created_at: str
    updated_at: str

    filename: str
    relative_path: str
    preview_relative_path: str | None = None
    metadata_relative_path: str | None = None

    mime_type: str
    size_bytes: int

    width: int | None = None
    height: int | None = None

    frames: int | None = None
    fps: int | None = None
    duration_sec: float | None = None

    prompt: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None

    loras: list[LoRARequest] = Field(default_factory=list)
    generation_params: dict = Field(default_factory=dict)
    model_paths: dict = Field(default_factory=dict)
