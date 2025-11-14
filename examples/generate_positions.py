"""Generate simulation timeline data for the orbital_layout visualizer."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from orbital_layout import GravityWell, LayoutSimulator, Planet, Satellite

OUTPUT_PATH = Path(__file__).with_name("sim_data.json")


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count < 2:
        return [(start + end) / 2.0]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _build_even_satellites(
    per_axis: int = 4,
    canvas_size: tuple[int, int] = (100, 100),
    margin: float = 12.0,
    base_speed: float = 0.5,
    rng: random.Random | None = None,
) -> list[Satellite]:
    width, height = canvas_size
    xs = _linspace(margin, width - margin, per_axis)
    ys = _linspace(margin, height - margin, per_axis)
    satellites: list[Satellite] = []
    rng = rng or random.Random()
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            heading = rng.uniform(0.0, 2 * math.pi)
            speed = base_speed * (0.25 + 0.5 * rng.random())
            mass = 0.8 + 2.7 * rng.random()
            satellites.append(
                Satellite(
                    x=x,
                    y=y,
                    vx=math.cos(heading) * speed,
                    vy=math.sin(heading) * speed,
                    mass=mass,
                )
            )
    return satellites


def _well_strength_from_mass(mass: float, *, base: float = 20.0, min_strength: float = 80.0) -> float:
    return max(mass * base, min_strength)


def _repulsive_core_well(center: tuple[float, float], *, strength: float) -> list[GravityWell]:
    cx, cy = center
    return [
        GravityWell(
            x=cx,
            y=cy,
            strength=strength,
            falloff=2.1,
            polarity="repel",
        )
    ]


def run_simulation(
    steps: int = 1000,
    dt: float = 0.12,
    *,
    seed: int = 1337,
    planet_mass: float = 128.0,
) -> None:
    rng = random.Random(seed)
    canvas = (100, 100)
    planet = Planet(x=65.0, y=35.0, mass=planet_mass)
    satellites = _build_even_satellites(canvas_size=canvas, rng=rng)
    well_strength = _well_strength_from_mass(planet.mass)

    simulator = LayoutSimulator(
        planet,
        satellites,
        canvas_size=canvas,
        repel_k=45.0,
        damping=0.995,
        boundary_k=60.0,
        boundary_margin=15.0,
        noise_sigma=0.0,
        viscous_drag=1.25,
        settle_speed=0.04,
        gravity_wells=_repulsive_core_well((planet.x, planet.y), strength=well_strength),
    )

    timeline = [
        {
            "step": 0,
            "positions": simulator.positions(),
        }
    ]

    for step in range(1, steps + 1):
        simulator.step(dt=dt)
        timeline.append(
            {
                "step": step,
                "positions": simulator.positions(),
            }
        )

    payload = {
        "canvas": {"width": 100, "height": 100},
        "planet": {"x": planet.x, "y": planet.y, "mass": planet.mass},
        "satellites": [{"mass": sat.mass} for sat in satellites],
        "timeline": timeline,
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved timeline with {len(timeline)} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_simulation()
