"""Text rendering utilities with pluggable backends.

The initial implementation relies on Pillow for rasterization, while exposing
an interface that can later be backed by Cairo, Skia, or any hardware renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

cairo = None
try:  # Preferred high-quality backend (pycairo).
    import cairo as _cairo  # type: ignore
    cairo = _cairo
except ImportError:  # pragma: no cover - optional dependency
    try:
        import cairocffi as _cairo  # type: ignore
        cairo = _cairo
    except ImportError:
        cairo = None

try:  # Pillow is the baseline fallback renderer.
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Pillow is required for text rendering. "
        "Install it via `pip install Pillow`."
    ) from exc
def _load_freetype():  # pragma: no cover - optional dependency helper
    try:
        import freetype  # type: ignore
    except ImportError:
        return None
    return freetype


class CairoTextRenderer:
    """Renderer that prefers Cairo + FreeType, falling back elsewhere if unavailable."""

    name = "cairo"

    def __init__(self) -> None:
        if cairo is None:
            raise ImportError("Cairo backend requires `pycairo` or `cairocffi` to be installed.")
        freetype = _load_freetype()
        if freetype is None:
            raise ImportError("Cairo backend requires `freetype-py` for custom font loading.")
        self._cairo = cairo
        self._freetype = freetype

    def render(self, config: TextRenderConfig) -> TextRenderResult:
        if config.stroke_width:
            raise NotImplementedError("Cairo backend does not yet support strokes; use Pillow for that.")

        width, height = config.canvas_size
        surface = self._cairo.ImageSurface(self._cairo.FORMAT_A8, width, height)
        ctx = self._cairo.Context(surface)
        bg = config.background / 255.0
        ctx.set_source_rgba(bg, bg, bg, bg)
        ctx.paint()
        fill = config.fill / 255.0
        ctx.set_source_rgba(fill, fill, fill, fill)

        face = self._freetype.Face(str(config.font_path))
        face.set_char_size(config.font_size * 64)
        metrics = face.size
        ascent = getattr(metrics, "ascender", config.font_size * 0.8) / 64.0
        descent = -getattr(metrics, "descender", -config.font_size * 0.2) / 64.0
        raw_height = getattr(metrics, "height", 0) / 64.0
        line_height = raw_height if raw_height > 0 else ascent + descent or config.font_size

        text_width = self._measure_line_width(face, config.text)
        if config.position is None:
            start_x = (width - text_width) / 2.0
            start_y = (height - line_height) / 2.0
        else:
            start_x, start_y = config.position

        baseline_x = start_x
        baseline_y = start_y + ascent
        pen_x = baseline_x
        pen_y = baseline_y
        prev_char: int | None = None

        ctx.translate(0, 0)

        for char in config.text:
            if char == "\n":
                pen_y += line_height
                pen_x = baseline_x
                prev_char = None
                continue

            char_code = ord(char)
            if prev_char is not None and face.has_kerning:
                kerning = face.get_kerning(prev_char, char_code).x / 64.0
                pen_x += kerning

            face.load_char(char_code, self._freetype.FT_LOAD_RENDER | self._freetype.FT_LOAD_TARGET_NORMAL)
            glyph = face.glyph
            bitmap = glyph.bitmap
            bw = bitmap.width
            bh = bitmap.rows
            if bw and bh:
                pitch = abs(bitmap.pitch) if hasattr(bitmap, "pitch") else bw
                aligned_stride = (bw + 3) & ~3  # Cairo requires 4-byte aligned stride.
                source = bitmap.buffer
                if pitch == aligned_stride:
                    glyph_bytes = bytearray(source)
                else:
                    glyph_bytes = bytearray(aligned_stride * bh)
                    for row in range(bh):
                        src_offset = row * pitch
                        dst_offset = row * aligned_stride
                        glyph_bytes[dst_offset : dst_offset + bw] = source[src_offset : src_offset + bw]

                glyph_surface = self._cairo.ImageSurface.create_for_data(
                    glyph_bytes,
                    self._cairo.FORMAT_A8,
                    bw,
                    bh,
                    aligned_stride,
                )
                ctx.mask_surface(glyph_surface, pen_x + glyph.bitmap_left, pen_y - glyph.bitmap_top)
                glyph_surface.finish()

            pen_x += glyph.advance.x / 64.0
            prev_char = char_code

        surface.flush()
        stride = surface.get_stride()
        buffer = np.frombuffer(surface.get_data(), dtype=np.uint8)
        mask = buffer.reshape((height, stride))[:, :width].copy()

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            bbox = (0, 0, 0, 0)
            area = 0
        else:
            min_x = int(xs.min())
            max_x = int(xs.max())
            min_y = int(ys.min())
            max_y = int(ys.max())
            bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            area = int((mask > 0).sum())

        area_ratio = area / float(width * height) if width and height else 0.0
        return TextRenderResult(config=config, mask=mask, bbox=bbox, area=area, area_ratio=area_ratio)

    def _measure_line_width(self, face, text: str) -> float:
        width = 0.0
        line_width = 0.0
        prev_char: int | None = None
        for char in text:
            if char == "\n":
                width = max(width, line_width)
                line_width = 0.0
                prev_char = None
                continue
            char_code = ord(char)
            if prev_char is not None and face.has_kerning:
                line_width += face.get_kerning(prev_char, char_code).x / 64.0
            face.load_char(char_code, self._freetype.FT_LOAD_DEFAULT)
            line_width += face.glyph.advance.x / 64.0
            prev_char = char_code
        width = max(width, line_width)
        return width


@dataclass(slots=True)
class TextRenderConfig:
    """Configuration for rendering a single piece of text."""

    text: str
    font_path: Path
    font_size: int
    canvas_size: tuple[int, int] = (1024, 1024)
    fill: int = 255
    background: int = 0
    stroke_width: int = 0
    stroke_fill: int = 0
    position: tuple[int, int] | None = None


@dataclass(slots=True)
class TextRenderResult:
    """Structured output containing the raster + measurements."""

    config: TextRenderConfig
    mask: np.ndarray  # uint8 mask of shape (H, W)
    bbox: tuple[int, int, int, int]  # (x, y, w, h) relative to canvas
    area: int
    area_ratio: float


@runtime_checkable
class TextRenderer(Protocol):
    """Protocol implemented by concrete backends (Pillow, Cairo, Skia, ...)."""

    name: str

    def render(self, config: TextRenderConfig) -> TextRenderResult:
        """Render text according to ``config`` and return the raster output."""


class PillowTextRenderer:
    """Default renderer that uses Pillow's ImageDraw to rasterize text."""

    name = "pillow"

    def render(self, config: TextRenderConfig) -> TextRenderResult:
        width, height = config.canvas_size
        image = Image.new("L", (width, height), color=config.background)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(config.font_path), config.font_size)

        xy = config.position
        if xy is None:
            # Center the text by default.
            left, top, right, bottom = draw.textbbox(
                (0, 0),
                config.text,
                font=font,
                stroke_width=config.stroke_width,
            )
            text_w = right - left
            text_h = bottom - top
            xy = ((width - text_w) // 2, (height - text_h) // 2)

        draw.text(
            xy,
            config.text,
            font=font,
            fill=config.fill,
            stroke_width=config.stroke_width,
            stroke_fill=config.stroke_fill,
        )

        mask = np.array(image, dtype=np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            bbox = (0, 0, 0, 0)
            area = 0
        else:
            min_x = int(xs.min())
            max_x = int(xs.max())
            min_y = int(ys.min())
            max_y = int(ys.max())
            bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            area = int((mask > 0).sum())

        area_ratio = area / float(width * height) if width and height else 0.0

        return TextRenderResult(
            config=config,
            mask=mask,
            bbox=bbox,
            area=area,
            area_ratio=area_ratio,
        )


def get_text_renderer(preferred: str | None = None) -> TextRenderer:
    """Return the best available renderer, preferring Cairo/Skia over Pillow."""

    normalized = (preferred or "").strip().lower()
    supported = {"", "cairo", "skia", "pillow"}
    if normalized not in supported:
        raise ValueError(f"Unknown renderer '{preferred}'")

    def candidate_order() -> list[str]:
        if normalized:
            return [normalized]
        return ["cairo", "skia", "pillow"]

    last_error: Exception | None = None
    for name in candidate_order():
        try:
            if name == "cairo":
                return CairoTextRenderer()
            if name == "skia":  # pragma: no cover - placeholder for future backend
                raise ImportError("Skia backend not implemented yet.")
            if name == "pillow":
                return PillowTextRenderer()
        except (ImportError, NotImplementedError) as exc:
            last_error = exc
            if normalized:
                raise
            continue

    raise RuntimeError("No text renderer backend is available") from last_error


__all__ = [
    "TextRenderConfig",
    "TextRenderResult",
    "TextRenderer",
    "CairoTextRenderer",
    "PillowTextRenderer",
    "get_text_renderer",
]
