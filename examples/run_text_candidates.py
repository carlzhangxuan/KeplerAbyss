"""CLI helper to render multiple text candidates from a JSON spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbital_layout import TextCandidate, render_text_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON file describing prj_id, canvas_size, and candidates.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for generated masks (default: output/).",
    )
    parser.add_argument(
        "--anchor-cols",
        type=int,
        default=15,
        help="Number of anchor columns (default: 15)",
    )
    parser.add_argument(
        "--anchor-rows",
        type=int,
        default=7,
        help="Number of anchor rows (default: 7)",
    )
    parser.add_argument(
        "--anchor-border",
        type=float,
        default=0.05,
        help="Border ratio kept free of anchors/text (default: 0.05).",
    )
    parser.add_argument(
        "--no-anchor-candidates",
        action="store_true",
        help="Skip anchor candidate computation and omit the field from output.",
    )
    parser.add_argument(
        "--no-anchor-map",
        action="store_true",
        help="Skip writing the per-candidate anchor visualization PNG.",
    )
    return parser.parse_args()


def load_candidates(config_path: Path) -> tuple[str, tuple[int, int], list[TextCandidate]]:
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    prj_id = spec["prj_id"]
    canvas_size = tuple(spec.get("canvas_size", [1024, 1024]))  # type: ignore[arg-type]
    candidates = [TextCandidate(**candidate) for candidate in spec["candidates"]]
    return prj_id, canvas_size, candidates


def main() -> None:
    args = parse_args()
    prj_id, canvas_size, candidates = load_candidates(args.config)
    summary = render_text_candidates(
        prj_id,
        candidates,
        canvas_size=canvas_size,
        output_root=args.output_root,
        anchor_grid_dims=(args.anchor_cols, args.anchor_rows),
        anchor_border_ratio=args.anchor_border,
        include_anchor_candidates=not args.no_anchor_candidates,
        render_anchor_visualizations=not args.no_anchor_map,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
