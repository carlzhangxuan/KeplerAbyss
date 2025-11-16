"""Render a heatmap of the orbital-layout force field on the canvas."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (SRC_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from orbital_layout import Anchor, LayoutSimulator, Satellite  # noqa: E402
from run_text_layout import (  # noqa: E402
    ANCHOR_CAPTURE_RADIUS,
    ANCHOR_CAPTURE_SPEED,
    ANCHOR_SHARED_LABELS,
    DEFAULT_INPUT,
    PAIRWISE_REPEL_SCALE,
    SatelliteConfig,
    _anchor_gravity_wells,
    _anchor_local_damping,
    _apply_anchor_capture,
    _assign_remaining_to_anchors,
    _flatten_anchor_lists,
    _build_anchor_map,
    _build_simulator,
    _damping_multiplier_for_position,
    _load_payload,
    _planet_from_payload,
    _satellites_from_payload,
)


def _simulate_state(
    input_path: Path,
    *,
    steps: int,
    dt: float,
    initial_speed: float,
    include_subtitle: bool,
    use_anchor_wells: bool,
    anchor_labels: Iterable[str] | None,
    anchor_capture_radius: float,
    anchor_capture_speed: float,
    pairwise_repel_scale: float,
) -> tuple[dict, LayoutSimulator, list[SatelliteConfig], dict[str, list[Anchor]]]:
    payload = _load_payload(input_path)
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
    anchor_wells: list = []
    if use_anchor_wells and anchor_map:
        anchor_wells = _anchor_gravity_wells(anchor_map, sat_configs, fallback_anchors=all_anchors)
        anchor_damping_records = _anchor_local_damping(
            anchor_map,
            sat_configs,
            fallback_anchors=all_anchors,
        )
    else:
        anchor_damping_records = []
    simulator = _build_simulator(
        payload,
        satellites=[cfg.satellite for cfg in sat_configs],
        planet=planet,
        anchor_wells=anchor_wells,
        repel_scale=pairwise_repel_scale,
    )
    capture_state: dict[str, Anchor] = {}
    anchor_claims: dict[Anchor, set[str]] = {}
    labels = [cfg.label for cfg in sat_configs]
    _apply_anchor_capture(
        simulator,
        anchor_map,
        capture_state,
        labels,
        radius=anchor_capture_radius,
        speed_threshold=anchor_capture_speed,
        global_anchors=all_anchors,
        anchor_claims=anchor_claims,
        shared_labels=ANCHOR_SHARED_LABELS,
    )
    for _ in range(steps):
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
            labels,
            radius=anchor_capture_radius,
            speed_threshold=anchor_capture_speed,
            global_anchors=all_anchors,
            anchor_claims=anchor_claims,
            shared_labels=ANCHOR_SHARED_LABELS,
        )
    _assign_remaining_to_anchors(
        simulator,
        labels,
        capture_state,
        anchor_claims,
        all_anchors,
        shared_labels=ANCHOR_SHARED_LABELS,
    )
    return payload, simulator, sat_configs, anchor_map


def _force_vector(simulator: LayoutSimulator, point: tuple[float, float]) -> tuple[float, float]:
    probe = Satellite(x=point[0], y=point[1], mass=1.0)
    fx = 0.0
    fy = 0.0

    # Pairwise repulsion vs. real satellites
    for sat in simulator.satellites:
        dx = probe.x - sat.x
        dy = probe.y - sat.y
        r2 = dx * dx + dy * dy + 1e-6
        r = math.sqrt(r2)
        f_mag = simulator.repel_k * probe.mass * sat.mass / r2
        fx += f_mag * dx / r
        fy += f_mag * dy / r

    # Gravity wells (planet + anchors)
    if simulator.gravity_wells:
        gfx, gfy = simulator._gravity_wells_force(probe)
        fx += gfx
        fy += gfy

    # Soft boundary force keeps things inside mask
    if simulator.boundary_k > 0.0 and simulator.boundary_slope > 0.0:
        bfx, bfy = simulator._boundary_force(probe)
        fx += bfx
        fy += bfy

    return fx, fy


def _sample_force_field(
    simulator: LayoutSimulator,
    canvas: dict,
    grid_cols: int,
    grid_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = float(canvas["width"])
    height = float(canvas["height"])
    xs = np.linspace(0.0, width, grid_cols)
    ys = np.linspace(0.0, height, grid_rows)
    magnitudes = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    vectors = np.zeros((grid_rows, grid_cols, 2), dtype=np.float64)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            fx, fy = _force_vector(simulator, (float(x), float(y)))
            vectors[yi, xi, 0] = fx
            vectors[yi, xi, 1] = fy
            magnitudes[yi, xi] = math.hypot(fx, fy)
    return xs, ys, magnitudes, vectors


def _render_heatmap(
    payload: dict,
    simulator: LayoutSimulator,
    sat_configs: list[SatelliteConfig],
    anchor_map: dict[str, list[Anchor]],
    *,
    grid_cols: int,
    grid_rows: int,
    cmap: str,
    output_path: Path,
    quiver_stride: int,
) -> None:
    canvas = payload["mask_canvas_size"]
    xs, ys, magnitudes, vectors = _sample_force_field(simulator, canvas, grid_cols, grid_rows)
    heat = np.log1p(magnitudes)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = (0, float(canvas["width"]), float(canvas["height"]), 0)
    im = ax.imshow(heat, extent=extent, origin="upper", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log(1 + |force|)")

    # Mask bbox
    bbox = payload["mask_bbox"]
    rect = plt.Rectangle(
        (bbox["x"], bbox["y"]),
        bbox["w"],
        bbox["h"],
        edgecolor="white",
        facecolor="none",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
    )
    ax.add_patch(rect)

    # Anchors
    for label, anchors in anchor_map.items():
        ax.scatter(
            [anchor.x for anchor in anchors],
            [anchor.y for anchor in anchors],
            s=30,
            label=f"{label} anchors",
            marker="x",
            alpha=0.8,
        )

    # Satellites (final positions)
    for config, sat in zip(sat_configs, simulator.satellites):
        ax.scatter(
            sat.x,
            sat.y,
            s=80,
            label=f"{config.label} final",
            edgecolor="black",
            linewidth=0.5,
        )

    # Quiver overlay (downsampled)
    if quiver_stride > 0:
        quiver_xs = xs[::quiver_stride]
        quiver_ys = ys[::quiver_stride]
        Qx, Qy = np.meshgrid(quiver_xs, quiver_ys)
        sampled = vectors[::quiver_stride, ::quiver_stride]
        U = sampled[:, :, 0]
        V = -sampled[:, :, 1]
        ax.quiver(Qx, Qy, U, V, color="white", alpha=0.4, scale=80000)

    ax.set_xlim(0, float(canvas["width"]))
    ax.set_ylim(float(canvas["height"]), 0)
    ax.set_title("Force field magnitude")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the orbital-layout force field as a heatmap")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to the JSON payload")
    parser.add_argument("--steps", type=int, default=800, help="Number of integration steps to run")
    parser.add_argument("--dt", type=float, default=0.08, help="Simulation timestep")
    parser.add_argument("--initial-speed", type=float, default=200.0, help="Initial velocity toward the planet center")
    parser.add_argument("--no-subtitle", action="store_true", help="Disable the subtitle satellite")
    parser.add_argument("--no-anchor-wells", action="store_true", help="Disable anchor wells")
    parser.add_argument(
        "--anchor-labels",
        choices=["title", "subtitle"],
        nargs="+",
        default=None,
        help="Restrict which labels produce anchors (default = title only)",
    )
    parser.add_argument("--capture-radius", type=float, default=ANCHOR_CAPTURE_RADIUS, help="Anchor capture radius")
    parser.add_argument("--capture-speed", type=float, default=ANCHOR_CAPTURE_SPEED, help="Anchor capture speed threshold")
    parser.add_argument(
        "--pairwise-repel-scale",
        type=float,
        default=PAIRWISE_REPEL_SCALE,
        help="Pairwise repulsion scale factor",
    )
    parser.add_argument("--grid-cols", type=int, default=160, help="Number of samples across the canvas width")
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=160,
        help="Number of samples across the canvas height (default matches cols)",
    )
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap name to use")
    parser.add_argument(
        "--quiver-stride",
        type=int,
        default=20,
        help="Downsampling stride for vector arrows (0 disables quiver)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("force_field_heatmap.png"),
        help="Where to save the rendered heatmap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_rows = args.grid_rows if args.grid_rows > 0 else args.grid_cols
    payload, simulator, sat_configs, anchor_map = _simulate_state(
        args.input,
        steps=args.steps,
        dt=args.dt,
        initial_speed=args.initial_speed,
        include_subtitle=not args.no_subtitle,
        use_anchor_wells=not args.no_anchor_wells,
        anchor_labels=args.anchor_labels,
        anchor_capture_radius=args.capture_radius,
        anchor_capture_speed=args.capture_speed,
        pairwise_repel_scale=args.pairwise_repel_scale,
    )
    _render_heatmap(
        payload,
        simulator,
        sat_configs,
        anchor_map,
        grid_cols=args.grid_cols,
        grid_rows=grid_rows,
        cmap=args.cmap,
        output_path=args.output,
        quiver_stride=max(args.quiver_stride, 0),
    )
    print(f"Force field heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
