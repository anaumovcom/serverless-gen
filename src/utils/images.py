"""Image helper utilities."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


PREVIEW_MAX_SIZE = (512, 512)


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


def make_image_preview(src: Path, dst: Path, max_size: tuple[int, int] = PREVIEW_MAX_SIZE) -> None:
    """Create a JPG preview of ``src`` at ``dst`` with ``max_size`` bound."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size)
        img.save(dst, format="JPEG", quality=85, optimize=True)


def write_placeholder_png(dst: Path, text: str, width: int = 1024, height: int = 1024) -> None:
    """Write a simple placeholder PNG containing ``text`` for MVP generators."""
    from PIL import ImageDraw, ImageFont

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(32, 32, 48))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Wrap text crudely
    max_chars = max(10, width // 12)
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)

    y = 32
    for line in lines[:40]:
        draw.text((32, y), line, fill=(230, 230, 230), font=font)
        y += 18

    img.save(dst, format="PNG", optimize=True)
