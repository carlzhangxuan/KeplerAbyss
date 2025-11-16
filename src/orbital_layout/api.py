"""High-level APIs for batch text rendering workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .anchors import generate_anchor_grid
from .text_rendering import (
    TextRenderConfig,
    TextRenderer,
    TextRenderResult,
    get_text_renderer,
)


@dataclass(slots=True)
class TextCandidate:
    """User-provided description of a text snippet to rasterize."""

    name: str
    text: str
    font_path: Path | str
    font_size: int
    fill: int = 255
    background: int = 0
    stroke_width: int = 0
    stroke_fill: int = 0
    position: tuple[int, int] | None = None
    canvas_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:  # pragma: no cover - trivial attribute normalization
        self.font_path = Path(self.font_path)
        if self.canvas_size is not None:
            self.canvas_size = (int(self.canvas_size[0]), int(self.canvas_size[1]))


def render_text_candidates(
    prj_id: str,
    candidates: Sequence[TextCandidate],
    *,
    canvas_size: tuple[int, int] = (1024, 1024),
    output_root: Path | str = "output",
    renderer: TextRenderer | None = None,
    anchor_grid_dims: tuple[int, int] = (15, 7),
    anchor_border_ratio: float = 0.05,
    include_anchor_candidates: bool = True,
    render_anchor_visualizations: bool = True,
) -> dict[str, Any]:
    """Render multiple text candidates and emit mask assets + JSON summary.

    Parameters
    ----------
    prj_id:
        Identifier used to create ``output_root / prj_id`` to store artifacts.
    candidates:
        Iterable of :class:`TextCandidate` definitions to render.
    canvas_size:
        Default canvas size used unless a candidate overrides ``canvas_size``.
    output_root:
        Directory under which project-specific asset folders are created.
    renderer:
        Optional renderer. When ``None`` the best available backend is selected.
    anchor_grid_dims:
        ``(cols, rows)`` for the anchor grid used to propose placement centers.
    anchor_border_ratio:
        Fraction of the canvas width/height reserved as a border where anchors/text
        cannot extend. Defaults to ``0.05`` (5%).
    include_anchor_candidates:
        When ``True`` (default) each candidate entry includes an ``anchor_candidates``
        list describing viable anchor positions.
    render_anchor_visualizations:
        When enabled (default), renders a scatter plot PNG of all candidate anchors
        per text block for quick inspection.
    """

    if not prj_id:
        raise ValueError("prj_id must be non-empty")
    if not candidates:
        raise ValueError("candidates must not be empty")

    if renderer is None:
        renderer = get_text_renderer()

    project_dir = Path(output_root) / prj_id
    project_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "project_id": prj_id,
        "renderer": renderer.name,
        "default_canvas_size": list(canvas_size),
        "output_dir": str(project_dir),
        "candidates": {},
    }

    for candidate in candidates:
        effective_canvas = candidate.canvas_size or canvas_size
        config = TextRenderConfig(
            text=candidate.text,
            font_path=candidate.font_path,
            font_size=candidate.font_size,
            canvas_size=effective_canvas,
            fill=candidate.fill,
            background=candidate.background,
            stroke_width=candidate.stroke_width,
            stroke_fill=candidate.stroke_fill,
            position=candidate.position,
        )
        result = renderer.render(config)

        full_mask_path = project_dir / f"{candidate.name}_mask.png"
        Image.fromarray(result.mask, mode="L").save(full_mask_path)

        fitted_mask = _extract_fitted_mask(result)
        fitted_mask_path = project_dir / f"{candidate.name}_mask_fit.png"
        Image.fromarray(fitted_mask, mode="L").save(fitted_mask_path)

        summary["candidates"][candidate.name] = {
            "text": candidate.text,
            "font_path": str(candidate.font_path),
            "font_size": candidate.font_size,
            "fill": candidate.fill,
            "background": candidate.background,
            "canvas_size": list(effective_canvas),
            "bbox": {
                "x": result.bbox[0],
                "y": result.bbox[1],
                "w": result.bbox[2],
                "h": result.bbox[3],
            },
            "area": result.area,
            "area_ratio": result.area_ratio,
            "mask_path": str(full_mask_path),
            "fitted_mask_path": str(fitted_mask_path),
            "fitted_shape": list(fitted_mask.shape),
        }

        anchor_candidates: list[dict[str, float | int | bool]] = []
        if include_anchor_candidates:
            anchor_candidates = _candidate_anchors_for_fit(
                fitted_shape=fitted_mask.shape,
                canvas_size=effective_canvas,
                grid_dims=anchor_grid_dims,
                border_ratio=anchor_border_ratio,
            )
            summary["candidates"][candidate.name]["anchor_candidates"] = anchor_candidates

        if render_anchor_visualizations and anchor_candidates:
            viz_path = _render_anchor_visualization(
                project_dir=project_dir,
                candidate_name=candidate.name,
                canvas_size=effective_canvas,
                anchors=anchor_candidates,
            )
            summary["candidates"][candidate.name]["anchor_visualization_path"] = viz_path

    candidates_json_path = project_dir / "candidates.json"
    summary["candidates_json_path"] = str(candidates_json_path)
    candidates_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _extract_fitted_mask(result: TextRenderResult) -> np.ndarray:
    """Return a tightly-cropped binary mask (values 0 or 255)."""

    x, y, w, h = result.bbox
    if w <= 0 or h <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    cropped = result.mask[y : y + h, x : x + w]
    binary = (cropped > 0).astype(np.uint8)
    return (binary * 255).astype(np.uint8)


def _candidate_anchors_for_fit(
    *,
    fitted_shape: tuple[int, int],
    canvas_size: tuple[int, int],
    grid_dims: tuple[int, int],
    border_ratio: float,
) -> list[dict[str, float | int | bool]]:
    width, height = canvas_size
    if width <= 0 or height <= 0:
        return []
    border_ratio = max(0.0, min(border_ratio, 0.49))
    margin_x = width * border_ratio
    margin_y = height * border_ratio
    anchors = generate_anchor_grid(
        canvas_size=(float(width), float(height)),
        grid_dims=grid_dims,
        margin=(margin_x, margin_y),
    )
    fit_h, fit_w = fitted_shape
    half_w = fit_w / 2.0
    half_h = fit_h / 2.0
    min_x = margin_x + half_w
    max_x = width - margin_x - half_w
    min_y = margin_y + half_h
    max_y = height - margin_y - half_h
    valid: list[dict[str, float | int | bool]] = []
    for anchor in anchors:
        if min_x <= anchor.x <= max_x and min_y <= anchor.y <= max_y:
            valid.append(
                {
                    "index": anchor.index,
                    "row": anchor.row,
                    "col": anchor.col,
                    "x": anchor.x,
                    "y": anchor.y,
                    "fits": True,
                }
            )
    if valid:
        return valid
    if not anchors:
        return []
    center = (width / 2.0, height / 2.0)
    fallback_anchor = min(
        anchors,
        key=lambda a: (a.x - center[0]) ** 2 + (a.y - center[1]) ** 2,
    )
    return [
        {
            "index": fallback_anchor.index,
            "row": fallback_anchor.row,
            "col": fallback_anchor.col,
            "x": fallback_anchor.x,
            "y": fallback_anchor.y,
            "fits": False,
        }
    ]


def _render_anchor_visualization(
    *,
    project_dir: Path,
    candidate_name: str,
    canvas_size: tuple[int, int],
    anchors: list[dict[str, float | int | bool]],
) -> str:
    width, height = canvas_size
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    radius = max(min(width, height) // 200, 4)
    for anchor in anchors:
        x = float(anchor["x"])
        y = float(anchor["y"])
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=(255, 64, 64))
    viz_path = project_dir / f"{candidate_name}_anchor_map.png"
    image.save(viz_path)
    return str(viz_path)


__all__ = ["TextCandidate", "render_text_candidates"]
