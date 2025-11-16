"""Project-specific text layout simulation using LayoutSimulator.

Reads a JSON payload (matching `examples/input/text_layout_jp.json`) and computes
final positions for the title/subtitle pair using the gravitational layout model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Allow running the script from the repo without installing the package first.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SRC_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from orbital_layout import Anchor, GravityWell, LayoutSimulator, Planet, Satellite, select_text_anchors  # noqa: E402

DEFAULT_INPUT = Path(__file__).with_name("input").joinpath("text_layout_jp.json")
DEFAULT_OUTPUT = Path(__file__).with_name("text_layout_positions.json")
DEFAULT_TIMELINE = Path(__file__).with_name("text_layout_sim_data.json")

ANCHOR_GRID_DIMS = (7, 10)
ANCHOR_MARGIN_RATIO = 0.05
ANCHOR_PADDING = (16.0, 12.0)
ANCHOR_GLYPH_WIDTH = 0.55
ANCHOR_WELL_STRENGTH_SCALE = 120.0
ANCHOR_WELL_FALLOFF = 2.1
ANCHOR_WELL_RADIUS = 100.0
ANCHOR_DAMPING_MULTIPLIER = 0.85
ANCHOR_NUDGE_THRESHOLD = 120.0
ANCHOR_CAPTURE_RADIUS = 120.0
ANCHOR_CAPTURE_SPEED = 32.0
PAIRWISE_REPEL_SCALE = 0.0008
ANCHOR_SHARED_LABELS: set[str] = {"subtitle"}


@dataclass
class SatelliteConfig:
    label: str
    mass: float
    initial: tuple[float, float]
    satellite: Satellite


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required_keys = [
        "mask_canvas_size",
        "mask_bbox",
        "mask_area",
        "title",
        "subtitle",
    ]
    for key in required_keys:
        if key not in data:
            msg = f"Missing key '{key}' in payload {path}"
            raise KeyError(msg)
    return data


def _planet_from_payload(payload: dict[str, Any]) -> Planet:
    bbox = payload["mask_bbox"]
    cx = float(bbox["x"]) + float(bbox["w"]) / 2.0
    cy = float(bbox["y"]) + float(bbox["h"]) / 2.0
    mass = min(float(payload["mask_area"]), 5000.0)
    return Planet(x=cx, y=cy, mass=mass)


def _velocity_toward(center: tuple[float, float], start: tuple[float, float], speed: float) -> tuple[float, float]:
    dx = center[0] - start[0]
    dy = center[1] - start[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= 1e-6 or speed <= 0.0:
        return 0.0, 0.0
    scale = speed / dist
    scale *= 1
    return dx * scale, dy * scale


def _satellites_from_payload(
    payload: dict[str, Any],
    *,
    planet: Planet,
    initial_speed: float = 0.0,
    include_subtitle: bool = True,
) -> list[SatelliteConfig]:
    canvas = payload["mask_canvas_size"]
    canvas_cx = float(canvas["width"]) / 2.0
    canvas_cy = float(canvas["height"]) / 2.0
    title_pos = (canvas_cx, canvas_cy)
    subtitle_pos = (canvas_cx, canvas_cy)

    bbox = payload["mask_bbox"]
    mask_cx = float(bbox["x"]) + float(bbox["w"]) / 2.0
    mask_cy = float(bbox["y"]) + float(bbox["h"]) / 2.0
    mask_center = (mask_cx, mask_cy)

    title_area = float(payload["title"]["area"])
    subtitle_area = float(payload["subtitle"]["area"])

    def velocity_from(position: tuple[float, float]) -> tuple[float, float]:
        if initial_speed <= 0.0:
            return 0.0, 0.0
        if math.isclose(position[0], mask_center[0], abs_tol=1e-6) and math.isclose(
            position[1], mask_center[1], abs_tol=1e-6
        ):
            return 0.0, -initial_speed
        return _velocity_toward(mask_center, position, initial_speed)

    title_vx, title_vy = velocity_from(title_pos)
    subtitle_vx, subtitle_vy = velocity_from(subtitle_pos)

    title_sat = Satellite(x=title_pos[0], y=title_pos[1], vx=title_vx, vy=title_vy, mass=title_area)
    subtitle_sat = Satellite(x=subtitle_pos[0], y=subtitle_pos[1], vx=subtitle_vx, vy=subtitle_vy, mass=subtitle_area)

    configs = [
        SatelliteConfig(label="title", mass=title_area, initial=title_pos, satellite=title_sat),
    ]
    if include_subtitle:
        configs.append(
            SatelliteConfig(label="subtitle", mass=subtitle_area, initial=subtitle_pos, satellite=subtitle_sat)
        )
    return configs


def _gravity_strength_from_mass(mass: float) -> float:
    # Scale linearly with the mask mass to keep the repulsive hill dominant even far from the center.
    return mass * 5000.0


def _serialize_anchor(anchor: Anchor) -> dict[str, float | int]:
    return {
        "index": anchor.index,
        "row": anchor.row,
        "col": anchor.col,
        "x": anchor.x,
        "y": anchor.y,
        "radius": ANCHOR_WELL_RADIUS,
    }


def _flatten_anchor_lists(anchor_map: dict[str, list[Anchor]] | None) -> list[Anchor]:
    if not anchor_map:
        return []
    flattened: list[Anchor] = []
    seen: set[Anchor] = set()
    for anchors in anchor_map.values():
        for anchor in anchors:
            if anchor in seen:
                continue
            seen.add(anchor)
            flattened.append(anchor)
    return flattened


def _candidate_anchors(
    payload: dict[str, Any],
    label: str,
    *,
    grid_dims: tuple[int, int] = ANCHOR_GRID_DIMS,
    padding: tuple[float, float] = ANCHOR_PADDING,
    glyph_width_factor: float = ANCHOR_GLYPH_WIDTH,
) -> list[Anchor]:
    if label not in payload:
        return []
    text_block = payload[label]
    text = text_block.get("text", "")
    font_size = float(text_block.get("font_size", 64.0))
    canvas = payload["mask_canvas_size"]
    canvas_w = float(canvas["width"])
    canvas_h = float(canvas["height"])
    margin_x = max(canvas_w * ANCHOR_MARGIN_RATIO, padding[0])
    margin_y = max(canvas_h * ANCHOR_MARGIN_RATIO, padding[1])
    anchors = select_text_anchors(
        text,
        font_size,
        canvas_size=(canvas_w, canvas_h),
        grid_dims=grid_dims,
        margin=(margin_x, margin_y),
        padding=padding,
        glyph_width_factor=glyph_width_factor,
    )
    return [
        Anchor(
            index=anchor.index,
            row=anchor.row,
            col=anchor.col,
            x=anchor.x,
            y=anchor.y,
        )
        for anchor in anchors
    ]


def _build_anchor_map(
    payload: dict[str, Any],
    *,
    include_subtitle: bool,
    allowed_labels: Iterable[str] | None = None,
) -> dict[str, list[Anchor]]:
    labels = ["title"]
    if include_subtitle:
        labels.append("subtitle")
    if allowed_labels is not None:
        allow = set(allowed_labels)
        labels = [label for label in labels if label in allow]
    anchor_map: dict[str, list[Anchor]] = {}
    for label in labels:
        anchors = _candidate_anchors(payload, label)
        if anchors:
            anchor_map[label] = anchors
    return anchor_map


def _anchor_gravity_wells(
    anchor_map: dict[str, list[Anchor]],
    sat_configs: list[SatelliteConfig],
    fallback_anchors: list[Anchor] | None = None,
) -> list[GravityWell]:
    wells: list[GravityWell] = []
    for config in sat_configs:
        anchors = anchor_map.get(config.label)
        if (not anchors) and fallback_anchors:
            anchors = fallback_anchors
        if not anchors:
            continue
        per_anchor_strength = max(config.mass * ANCHOR_WELL_STRENGTH_SCALE / max(len(anchors), 1), 1.0)
        scaled_strength = per_anchor_strength * (ANCHOR_WELL_RADIUS ** ANCHOR_WELL_FALLOFF)
        for anchor in anchors:
            wells.append(
                GravityWell(
                    x=anchor.x,
                    y=anchor.y,
                    strength=scaled_strength,
                    falloff=ANCHOR_WELL_FALLOFF,
                    polarity="attract",
                )
            )
    return wells


def _anchor_local_damping(
    anchor_map: dict[str, list[Anchor]],
    sat_configs: list[SatelliteConfig],
    threshold: float = ANCHOR_NUDGE_THRESHOLD,
    multiplier: float = ANCHOR_DAMPING_MULTIPLIER,
    fallback_anchors: list[Anchor] | None = None,
) -> list[tuple[str, list[tuple[float, float, float]]]]:
    records: list[tuple[str, list[tuple[float, float, float]]]] = []
    for config in sat_configs:
        anchors = anchor_map.get(config.label, [])
        if (not anchors) and fallback_anchors:
            anchors = fallback_anchors
        triples: list[tuple[float, float, float]] = []
        for anchor in anchors:
            triples.append((anchor.x, anchor.y, threshold))
        if triples:
            records.append((config.label, triples))
    return records


def _damping_multiplier_for_position(
    position: tuple[float, float],
    anchor_records: list[tuple[str, list[tuple[float, float, float]]]],
    multiplier: float = ANCHOR_DAMPING_MULTIPLIER,
) -> float:
    px, py = position
    for _, triples in anchor_records:
        for ax, ay, threshold in triples:
            dx = px - ax
            dy = py - ay
            if dx * dx + dy * dy <= threshold * threshold:
                return multiplier
    return 1.0


def _satellite_speed(satellite: Satellite) -> float:
    return float((satellite.vx * satellite.vx + satellite.vy * satellite.vy) ** 0.5)


def _apply_anchor_capture(
    simulator: LayoutSimulator,
    anchor_map: dict[str, list[Anchor]],
    capture_state: dict[str, Anchor],
    labels: list[str],
    *,
    radius: float = ANCHOR_CAPTURE_RADIUS,
    speed_threshold: float = ANCHOR_CAPTURE_SPEED,
    global_anchors: list[Anchor] | None = None,
    anchor_claims: dict[Anchor, set[str]] | None = None,
    shared_labels: set[str] | None = None,
) -> None:
    if not anchor_map and not global_anchors:
        return
    radius_sq = radius * radius
    anchor_claims = anchor_claims if anchor_claims is not None else {}
    shared_labels = shared_labels or set()
    for idx, satellite in enumerate(simulator.satellites):
        label = labels[idx]
        anchors = anchor_map.get(label)
        if (not anchors) and global_anchors:
            anchors = global_anchors
        if not anchors:
            continue
        speed = _satellite_speed(satellite)
        current_anchor = capture_state.get(label)
        if current_anchor is not None:
            if speed > speed_threshold:
                capture_state.pop(label, None)
                labels_set = anchor_claims.get(current_anchor)
                if labels_set is not None:
                    labels_set.discard(label)
                    if not labels_set:
                        anchor_claims.pop(current_anchor, None)
            else:
                satellite.x = current_anchor.x
                satellite.y = current_anchor.y
                satellite.vx = 0.0
                satellite.vy = 0.0
            continue
        if speed > speed_threshold:
            continue
        for anchor in anchors:
            claimed_by = anchor_claims.get(anchor)
            if claimed_by and label not in shared_labels:
                continue
            dx = satellite.x - anchor.x
            dy = satellite.y - anchor.y
            if dx * dx + dy * dy <= radius_sq:
                satellite.x = anchor.x
                satellite.y = anchor.y
                satellite.vx = 0.0
                satellite.vy = 0.0
                capture_state[label] = anchor
                anchor_claims.setdefault(anchor, set()).add(label)
                break


def _assign_remaining_to_anchors(
    simulator: LayoutSimulator,
    labels: list[str],
    capture_state: dict[str, Anchor],
    anchor_claims: dict[Anchor, set[str]],
    anchors: list[Anchor],
    shared_labels: set[str] | None = None,
) -> None:
    if not anchors:
        return
    shared_labels = shared_labels or set()
    for idx, label in enumerate(labels):
        if label in capture_state:
            continue
        satellite = simulator.satellites[idx]
        available = anchors if label in shared_labels else [a for a in anchors if not anchor_claims.get(a)]
        if not available:
            continue
        anchor = min(available, key=lambda a: (satellite.x - a.x) ** 2 + (satellite.y - a.y) ** 2)
        satellite.x = anchor.x
        satellite.y = anchor.y
        satellite.vx = 0.0
        satellite.vy = 0.0
        capture_state[label] = anchor
        anchor_claims.setdefault(anchor, set()).add(label)


def _nearest_anchor(point: tuple[float, float], anchors: list[Anchor]) -> dict[str, float | int] | None:
    if not anchors:
        return None
    px, py = point
    best_anchor: Anchor | None = None
    best_dist2 = float("inf")
    for anchor in anchors:
        dx = px - anchor.x
        dy = py - anchor.y
        dist2 = dx * dx + dy * dy
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_anchor = anchor
    if best_anchor is None:
        return None
    return {
        **_serialize_anchor(best_anchor),
        "distance": best_dist2 ** 0.5,
    }


def _clamp_to_bbox(position: tuple[float, float], bbox: dict[str, float]) -> tuple[float, float]:
    min_x = float(bbox["x"])
    min_y = float(bbox["y"])
    max_x = min_x + float(bbox["w"])
    max_y = min_y + float(bbox["h"])
    x = min(max(position[0], min_x), max_x)
    y = min(max(position[1], min_y), max_y)
    return x, y


def _serialize_positions(positions: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in positions]


def _positions_for_output(
    simulator: LayoutSimulator,
    bbox: dict[str, float],
    *,
    clamp: bool,
) -> list[tuple[float, float]]:
    positions = simulator.positions()
    if not clamp:
        return positions
    return [_clamp_to_bbox(pos, bbox) for pos in positions]


def _build_timeline_payload(
    payload: dict[str, Any],
    planet: Planet,
    sat_configs: list[SatelliteConfig],
    timeline: list[dict[str, Any]],
    anchor_map: dict[str, list[Anchor]] | None,
) -> dict[str, Any]:
    return {
        "canvas": {
            "width": int(payload["mask_canvas_size"]["width"]),
            "height": int(payload["mask_canvas_size"]["height"]),
        },
        "planet": {"x": planet.x, "y": planet.y, "mass": planet.mass},
        "satellites": [
            {
                "label": config.label,
                "mass": config.mass,
            }
            for config in sat_configs
        ],
        "timeline": timeline,
        "anchors": {
            label: [_serialize_anchor(anchor) for anchor in anchors]
            for label, anchors in (anchor_map or {}).items()
        },
        "metadata": {
            "mask_bbox": payload["mask_bbox"],
            "input_path": payload.get("__path__"),
        },
    }


def _build_simulator(
    payload: dict[str, Any],
    *,
    satellites: Iterable[Satellite],
    planet: Planet,
    anchor_wells: Iterable[GravityWell] | None = None,
    repel_scale: float = PAIRWISE_REPEL_SCALE,
) -> LayoutSimulator:
    canvas = payload["mask_canvas_size"]
    well_strength = _gravity_strength_from_mass(planet.mass)
    gravity_wells = [
        GravityWell(
            x=planet.x,
            y=planet.y,
            strength=well_strength,
            falloff=1.6,
            polarity="repel",
        ),
    ]
    if anchor_wells:
        gravity_wells.extend(anchor_wells)
    repel_scale = max(repel_scale, 0.0)
    return LayoutSimulator(
        planet,
        list(satellites),
        canvas_size=(int(canvas["width"]), int(canvas["height"])),
        repel_k=planet.mass * repel_scale,
        damping=0.993,
        boundary_k=180.0,
        boundary_margin=24.0,
        boundary_slope=0.5,
        boundary_span_ratio=0.05,
        viscous_drag=0.08,
        settle_speed=0.03,
        gravity_wells=gravity_wells,
    )


def run_simulation(
    input_path: Path,
    *,
    output_path: Path | None = None,
    steps: int = 800,
    dt: float = 0.08,
    clamp_to_mask: bool = False,
    timeline_output: Path | None = None,
    timeline_interval: int = 1,
    initial_speed: float = 200,
    use_anchor_wells: bool = True,
    include_subtitle: bool = True,
    anchor_capture_radius: float = ANCHOR_CAPTURE_RADIUS,
    anchor_capture_speed: float = ANCHOR_CAPTURE_SPEED,
    anchor_labels: Iterable[str] | None = None,
    pairwise_repel_scale: float = PAIRWISE_REPEL_SCALE,
) -> dict[str, Any]:
    payload = _load_payload(input_path)
    payload["__path__"] = str(input_path)
    planet = _planet_from_payload(payload)
    sat_configs = _satellites_from_payload(
        payload,
        planet=planet,
        initial_speed=initial_speed,
        include_subtitle=include_subtitle,
    )
    allowed_labels = list(anchor_labels) if anchor_labels is not None else ["title"]
    anchor_map = _build_anchor_map(
        payload,
        include_subtitle=include_subtitle,
        allowed_labels=allowed_labels,
    )
    all_anchors = _flatten_anchor_lists(anchor_map)
    anchor_wells: list[GravityWell] = []
    if use_anchor_wells and anchor_map:
        anchor_wells = _anchor_gravity_wells(anchor_map, sat_configs, fallback_anchors=all_anchors)
        anchor_damping_records = _anchor_local_damping(
            anchor_map,
            sat_configs,
            fallback_anchors=all_anchors,
        )
    else:
        anchor_damping_records = []
    capture_state: dict[str, Anchor] = {}
    anchor_claims: dict[Anchor, set[str]] = {}

    simulator = _build_simulator(
        payload,
        satellites=[cfg.satellite for cfg in sat_configs],
        planet=planet,
        anchor_wells=anchor_wells,
        repel_scale=pairwise_repel_scale,
    )

    bbox = payload["mask_bbox"]
    config_labels = [cfg.label for cfg in sat_configs]
    _apply_anchor_capture(
        simulator,
        anchor_map,
        capture_state,
        config_labels,
        radius=anchor_capture_radius,
        speed_threshold=anchor_capture_speed,
        global_anchors=all_anchors,
        anchor_claims=anchor_claims,
        shared_labels=ANCHOR_SHARED_LABELS,
    )

    record_every = max(1, timeline_interval)
    timeline_frames: list[dict[str, Any]] = []
    if timeline_output is not None:
        positions = _positions_for_output(simulator, bbox, clamp=clamp_to_mask)
        timeline_frames.append({"step": 0, "positions": _serialize_positions(positions)})

    for step in range(1, steps + 1):
        simulator.step(dt=dt)
        if anchor_damping_records:
            for sat in simulator.satellites:
                damp = _damping_multiplier_for_position((sat.x, sat.y), anchor_damping_records)
                if damp < 1.0:
                    sat.vx *= damp
                    sat.vy *= damp
        _apply_anchor_capture(
            simulator,
            anchor_map,
            capture_state,
            config_labels,
            radius=anchor_capture_radius,
            speed_threshold=anchor_capture_speed,
            global_anchors=all_anchors,
            anchor_claims=anchor_claims,
            shared_labels=ANCHOR_SHARED_LABELS,
        )
        if timeline_output is not None and (step % record_every == 0):
            positions = _positions_for_output(simulator, bbox, clamp=clamp_to_mask)
            timeline_frames.append({"step": step, "positions": _serialize_positions(positions)})

    _assign_remaining_to_anchors(
        simulator,
        config_labels,
        capture_state,
        anchor_claims,
        all_anchors,
        shared_labels=ANCHOR_SHARED_LABELS,
    )

    positions = _positions_for_output(simulator, bbox, clamp=clamp_to_mask)
    result = {
        "canvas": payload["mask_canvas_size"],
        "mask_bbox": bbox,
        "planet": {"x": planet.x, "y": planet.y, "mass": planet.mass},
        "steps": steps,
        "dt": dt,
        "satellites": {},
        "clamped_to_mask": clamp_to_mask,
    }

    for config, (x, y) in zip(sat_configs, positions):
        final_pos = (x, y)
        if clamp_to_mask:
            final_pos = _clamp_to_bbox(final_pos, bbox)
        result["satellites"][config.label] = {
            "mass": config.mass,
            "initial": {"x": config.initial[0], "y": config.initial[1]},
            "final": {"x": final_pos[0], "y": final_pos[1]},
        }
        nearest = _nearest_anchor(final_pos, anchor_map.get(config.label, []))
        if nearest is not None:
            result["satellites"][config.label]["nearest_anchor"] = nearest
        captured_anchor = capture_state.get(config.label)
        if captured_anchor is not None:
            result["satellites"][config.label]["captured_anchor"] = _serialize_anchor(captured_anchor)

    if anchor_map:
        result["anchors"] = {
            label: [_serialize_anchor(anchor) for anchor in anchors]
            for label, anchors in anchor_map.items()
        }

    if output_path is not None:
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if timeline_output is not None:
        final_frame = {"step": steps, "positions": _serialize_positions(positions)}
        if timeline_frames and timeline_frames[-1]["step"] == steps:
            timeline_frames[-1] = final_frame
        else:
            timeline_frames.append(final_frame)
        timeline_payload = _build_timeline_payload(
            payload,
            planet,
            sat_configs,
            timeline_frames,
            anchor_map,
        )
        timeline_output.write_text(json.dumps(timeline_payload, indent=2), encoding="utf-8")
        result["timeline_output"] = str(timeline_output)
        result["timeline_frames"] = len(timeline_frames)
        result["timeline_interval"] = record_every

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run orbital-layout on a text mask JSON")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the JSON payload")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Optional JSON output path")
    parser.add_argument("--steps", type=int, default=800, help="Number of integrator steps to run")
    parser.add_argument("--dt", type=float, default=0.08, help="Timestep per integration step")
    clamp_group = parser.add_mutually_exclusive_group()
    clamp_group.add_argument(
        "--clamp",
        dest="clamp",
        action="store_true",
        help="Clamp timeline/final positions inside the mask bbox",
    )
    clamp_group.add_argument(
        "--no-clamp",
        dest="clamp",
        action="store_false",
        help="Emit raw simulator coordinates (default)",
    )
    parser.set_defaults(clamp=False)
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=200.0,
        help="Initial speed (pixels/step) aimed toward the planet center",
    )
    parser.add_argument(
        "--no-subtitle",
        action="store_true",
        help="Disable the subtitle satellite so only the title is simulated",
    )
    parser.add_argument(
        "--no-anchor-wells",
        action="store_true",
        help="Disable anchor-derived gravity wells and nearest-anchor output",
    )
    parser.add_argument(
        "--timeline-output",
        type=Path,
        default=None,
        help="Optional path to dump a sim_data-style timeline for the visualizer",
    )
    parser.add_argument(
        "--timeline-interval",
        type=int,
        default=1,
        help="Record every Nth frame in the timeline (1 = every step)",
    )
    parser.add_argument(
        "--capture-radius",
        type=float,
        default=ANCHOR_CAPTURE_RADIUS,
        help="Radius (px) around anchors where speed-based capture can occur",
    )
    parser.add_argument(
        "--capture-speed",
        type=float,
        default=ANCHOR_CAPTURE_SPEED,
        help="Speed threshold (px/step); slower satellites get captured",
    )
    parser.add_argument(
        "--anchor-labels",
        choices=["title", "subtitle"],
        nargs="+",
        default=None,
        help="Limit which labels generate anchors (default = title only)",
    )
    parser.add_argument(
        "--repel-scale",
        type=float,
        default=PAIRWISE_REPEL_SCALE,
        help="Scale factor for pairwise repulsion (default lowers the push apart)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_simulation(
        args.input,
        output_path=args.output,
        steps=args.steps,
        dt=args.dt,
        clamp_to_mask=args.clamp,
        timeline_output=args.timeline_output,
        timeline_interval=args.timeline_interval,
        initial_speed=args.initial_speed,
        use_anchor_wells=not args.no_anchor_wells,
        include_subtitle=not args.no_subtitle,
        anchor_capture_radius=args.capture_radius,
        anchor_capture_speed=args.capture_speed,
        anchor_labels=args.anchor_labels,
        pairwise_repel_scale=args.repel_scale,
    )
    print(json.dumps(result, indent=2))
    print(f"Saved layout result to {args.output}")


if __name__ == "__main__":
    main()
