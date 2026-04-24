"""Storage layer: save generated files, previews and metadata."""
from __future__ import annotations

import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .utils.atomic import write_file_atomic, write_json_atomic
from .utils.images import make_image_preview
from .utils.logging import get_logger
from .utils.paths import relative_to_storage
from .utils.video import extract_first_frame


logger = get_logger(__name__)


def create_file_id() -> str:
    """Generate a short hex id (10 chars)."""
    return secrets.token_hex(5)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_date_path() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}/{now.month:02d}/{now.day:02d}"


def _copy_into(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as f:
        data = f.read()
    write_file_atomic(target, data)


def save_generated_image(source_path: Path, file_id: str) -> tuple[Path, str]:
    date = get_date_path()
    filename = f"{file_id}_output.png"
    abs_path = config.IMAGES_DIR / date / filename
    _copy_into(source_path, abs_path)
    rel = relative_to_storage(abs_path, config.STORAGE_BASE_DIR)
    return abs_path, rel


def save_generated_video(source_path: Path, file_id: str) -> tuple[Path, str]:
    date = get_date_path()
    filename = f"{file_id}_result.mp4"
    abs_path = config.VIDEOS_DIR / date / filename
    _copy_into(source_path, abs_path)
    rel = relative_to_storage(abs_path, config.STORAGE_BASE_DIR)
    return abs_path, rel


def create_preview_for_image(image_path: Path, file_id: str) -> tuple[Path | None, str | None]:
    date = get_date_path()
    filename = f"{file_id}_preview.jpg"
    abs_path = config.PREVIEWS_DIR / date / filename
    try:
        make_image_preview(image_path, abs_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("image preview failed: %s", exc)
        return None, None
    rel = relative_to_storage(abs_path, config.STORAGE_BASE_DIR)
    return abs_path, rel


def create_preview_for_video(video_path: Path, file_id: str) -> tuple[Path | None, str | None]:
    date = get_date_path()
    filename = f"{file_id}_preview.jpg"
    abs_path = config.PREVIEWS_DIR / date / filename
    ok = extract_first_frame(video_path, abs_path)
    if not ok:
        return None, None
    rel = relative_to_storage(abs_path, config.STORAGE_BASE_DIR)
    return abs_path, rel


def save_metadata(metadata: dict, file_id: str) -> tuple[Path, str]:
    date = get_date_path()
    filename = f"{file_id}.json"
    abs_path = config.METADATA_DIR / date / filename
    write_json_atomic(abs_path, metadata)
    rel = relative_to_storage(abs_path, config.STORAGE_BASE_DIR)
    return abs_path, rel
