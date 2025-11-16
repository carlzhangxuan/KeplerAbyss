from __future__ import annotations

from pathlib import Path

import json
import numpy as np
from PIL import Image

from orbital_layout.api import TextCandidate, render_text_candidates


def test_render_text_candidates(tmp_path: Path) -> None:
    candidate = TextCandidate(
        name="title",
        text="テスト",
        font_path=Path("fonts/NotoSansJP-Bold.ttf"),
        font_size=48,
    )

    summary = render_text_candidates(
        prj_id="unit",
        candidates=[candidate],
        canvas_size=(256, 256),
        output_root=tmp_path,
        anchor_grid_dims=(5, 5),
        anchor_border_ratio=0.05,
        render_anchor_visualizations=True,
    )

    manifest_path = Path(summary["candidates_json_path"])
    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = saved["candidates"]["title"]
    mask_path = Path(result["mask_path"])
    fitted_path = Path(result["fitted_mask_path"])

    assert mask_path.exists()
    assert fitted_path.exists()

    with Image.open(fitted_path) as img:
        arr = np.array(img)
    unique_values = set(arr.flatten())
    assert unique_values.issubset({0, 255})
    assert 255 in unique_values
    assert result["bbox"]["w"] > 0
    assert result["bbox"]["h"] > 0

    anchors = result.get("anchor_candidates")
    assert anchors, "Expected at least one anchor candidate"
    sample = anchors[0]
    for key in ("index", "row", "col", "x", "y", "fits"):
        assert key in sample

    viz_path = result.get("anchor_visualization_path")
    assert viz_path, "Expected anchor visualization path when enabled"
    assert Path(viz_path).exists()


def test_anchor_candidate_fallback(tmp_path: Path) -> None:
    candidate = TextCandidate(
        name="wide",
        text="Ｘ" * 20,
        font_path=Path("fonts/NotoSansJP-Bold.ttf"),
        font_size=256,
    )

    summary = render_text_candidates(
        prj_id="fallback",
        candidates=[candidate],
        canvas_size=(512, 512),
        output_root=tmp_path,
        anchor_grid_dims=(5, 3),
        anchor_border_ratio=0.1,
    )

    data = json.loads(Path(summary["candidates_json_path"]).read_text(encoding="utf-8"))
    anchors = data["candidates"]["wide"].get("anchor_candidates")
    assert anchors, "Fallback anchor should be recorded even when text is too wide"
    assert anchors[0]["fits"] is False
