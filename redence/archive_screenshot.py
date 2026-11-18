"""Resolve archived crawl screenshots for agent vision (oldest PNG in screenshots/)."""

from __future__ import annotations

import base64
import io
from pathlib import Path


def oldest_screenshot_in_archive(archive_path: str | Path | None) -> Path | None:
    """
    Return the PNG in ``{archive_path}/screenshots/`` with the earliest mtime.
  ``archive_path`` is the crawl run root (already includes playwright/<slug>/).
    """
    if not archive_path:
        return None
    shots_dir = Path(archive_path) / "screenshots"
    if not shots_dir.is_dir():
        return None

    candidates: list[tuple[float, Path]] = []
    for entry in shots_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".png":
            continue
        try:
            candidates.append((entry.stat().st_mtime, entry))
        except OSError:
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def png_to_vision_b64(path: Path, *, jpeg_quality: int = 45) -> str | None:
    """Encode a PNG as base64 JPEG for Ollama (falls back to raw PNG if Pillow missing)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None

    try:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(raw).decode("ascii")


def load_archive_vision_b64(archive_path: str | Path | None) -> tuple[str | None, Path | None]:
    """Oldest archive PNG as vision base64, or (None, None) if unavailable."""
    png_path = oldest_screenshot_in_archive(archive_path)
    if png_path is None:
        return None, None
    b64 = png_to_vision_b64(png_path)
    if not b64:
        return None, None
    return b64, png_path







