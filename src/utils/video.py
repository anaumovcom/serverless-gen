"""Video helper utilities."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .logging import get_logger


logger = get_logger(__name__)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_first_frame(video_path: Path, dst_jpg: Path) -> bool:
    """Extract first frame to ``dst_jpg`` via ffmpeg.

    Returns True on success, False otherwise. Does not raise.
    """
    if not _ffmpeg_available():
        logger.warning("ffmpeg not available; skipping video preview")
        return False

    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "3",
        str(dst_jpg),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            logger.warning("ffmpeg failed: %s", result.stderr.decode("utf-8", errors="ignore"))
            return False
        return dst_jpg.exists() and dst_jpg.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffmpeg error: %s", exc)
        return False


def probe_video(video_path: Path) -> dict:
    """Probe video with ffprobe. Returns a dict with width/height/frames/fps/duration if possible."""
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return {}
        out: dict = {}
        for line in result.stdout.decode("utf-8", errors="ignore").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()

        parsed: dict = {}
        if "width" in out:
            try:
                parsed["width"] = int(out["width"])
            except ValueError:
                pass
        if "height" in out:
            try:
                parsed["height"] = int(out["height"])
            except ValueError:
                pass
        if "nb_frames" in out and out["nb_frames"].isdigit():
            parsed["frames"] = int(out["nb_frames"])
        if "avg_frame_rate" in out and "/" in out["avg_frame_rate"]:
            num, den = out["avg_frame_rate"].split("/")
            try:
                num_i, den_i = int(num), int(den)
                if den_i:
                    parsed["fps"] = round(num_i / den_i, 3)
            except ValueError:
                pass
        if "duration" in out:
            try:
                parsed["duration_sec"] = float(out["duration"])
            except ValueError:
                pass
        return parsed
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffprobe error: %s", exc)
        return {}


def write_placeholder_mp4(dst: Path, width: int = 832, height: int = 480, frames: int = 16, fps: int = 16) -> None:
    """Write a short placeholder MP4 using OpenCV (no ffmpeg required)."""
    import numpy as np
    import cv2

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("failed to open cv2.VideoWriter for placeholder mp4")
    try:
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            intensity = int(255 * (i + 1) / max(frames, 1))
            frame[:, :] = (intensity // 3, intensity // 2, intensity)
            cv2.putText(
                frame,
                f"frame {i + 1}/{frames}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
