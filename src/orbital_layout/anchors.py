"""Anchor generation utilities for text layout experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class Anchor:
    """Represents an anchor coordinate plus its index within the grid."""

    index: int
    row: int
    col: int
    x: float
    y: float

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class AnchorRequest:
    """Parameters describing how to place anchors on a rectangular canvas."""

    canvas_size: Tuple[float, float] = (1024.0, 1024.0)
    grid_dims: Tuple[int, int] = (7, 7)
    margin: Tuple[float, float] = (100.0, 100.0)


def _linspace(start: float, end: float, count: int) -> List[float]:
    if count <= 0:
        return []
    if count == 1:
        return [start + (end - start) / 2.0]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def generate_anchor_grid(
    *,
    canvas_size: Tuple[float, float] = (1024.0, 1024.0),
    grid_dims: Tuple[int, int] = (7, 7),
    margin: Tuple[float, float] = (100.0, 100.0),
) -> List[Anchor]:
    """Return evenly spaced anchor coordinates inside the canvas margins."""

    width, height = canvas_size
    cols, rows = grid_dims
    margin_x, margin_y = margin

    if width <= 0 or height <= 0:
        raise ValueError("canvas_size must be positive in both dimensions")
    if cols <= 0 or rows <= 0:
        raise ValueError("grid_dims must be positive")

    usable_w = max(width - 2 * margin_x, 0.0)
    usable_h = max(height - 2 * margin_y, 0.0)
    left = margin_x
    top = margin_y
    right = margin_x + usable_w
    bottom = margin_y + usable_h

    xs = _linspace(left, right, cols)
    ys = _linspace(top, bottom, rows)

    anchors: List[Anchor] = []
    index = 0
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            anchors.append(Anchor(index=index, row=row, col=col, x=x, y=y))
            index += 1
    return anchors


def _text_bbox(
    text: str,
    font_size: float,
    *,
    glyph_width_factor: float,
) -> Tuple[float, float]:
    glyphs = max(len(text), 1)
    width = glyphs * font_size * glyph_width_factor
    height = font_size
    return width, height


def select_text_anchors(
    text: str,
    font_size: float,
    *,
    canvas_size: Tuple[float, float] = (1024.0, 1024.0),
    grid_dims: Tuple[int, int] = (7, 7),
    margin: Tuple[float, float] = (100.0, 100.0),
    padding: Tuple[float, float] = (0.0, 0.0),
    glyph_width_factor: float = 0.55,
) -> List[Anchor]:
    """Return anchor positions whose text bounds remain inside the margins."""

    if font_size <= 0:
        raise ValueError("font_size must be positive")

    anchors = generate_anchor_grid(canvas_size=canvas_size, grid_dims=grid_dims, margin=margin)
    text_w, text_h = _text_bbox(text, font_size, glyph_width_factor=glyph_width_factor)
    pad_x, pad_y = padding
    half_w = text_w / 2.0 + pad_x
    half_h = text_h / 2.0 + pad_y

    width, height = canvas_size
    margin_x, margin_y = margin
    min_x = margin_x + half_w
    max_x = width - margin_x - half_w
    min_y = margin_y + half_h
    max_y = height - margin_y - half_h

    def fits(anchor: Anchor) -> bool:
        x, y = anchor.x, anchor.y
        return min_x <= x <= max_x and min_y <= y <= max_y

    return [anchor for anchor in anchors if fits(anchor)]
