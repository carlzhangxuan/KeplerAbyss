from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple


@dataclass
class Planet:
    x: float
    y: float
    mass: float = 1.0


@dataclass
class Satellite:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0


@dataclass
class GravityWell:
    x: float
    y: float
    strength: float = 100.0
    falloff: float = 2.0
    polarity: Literal["attract", "repel"] = "attract"


class LayoutSimulator:
    """Minimal 2D planetary layout simulator (CPU scalar version).

    Later you can reimplement this with PyTorch tensors and GPU support.
    """

    def __init__(
        self,
        planet: Planet,
        satellites: Iterable[Satellite],
        canvas_size: Tuple[int, int],
        *,
        repel_k: float = 1.0,
        damping: float = 0.9,
        boundary_k: float = 0.0,
        boundary_margin: float = 10.0,
        boundary_slope: float = 0.1,
        boundary_span_ratio: float = 0.5,
        noise_sigma: float = 0.0,
        noise_seed: int | None = None,
        viscous_drag: float = 0.0,
        settle_speed: float = 0.0,
        gravity_wells: Iterable[GravityWell] | None = None,
    ) -> None:
        self.planet = planet
        self.satellites: List[Satellite] = list(satellites)
        self.W, self.H = canvas_size
        self.repel_k = repel_k
        self.damping = damping
        self.boundary_k = boundary_k
        self.boundary_margin = boundary_margin
        self.boundary_slope = max(boundary_slope, 0.0)
        self.boundary_span_ratio = min(max(boundary_span_ratio, 1e-3), 1.0)
        self.noise_sigma = noise_sigma
        self._noise_rng = random.Random(noise_seed) if noise_sigma > 0.0 else None
        self.viscous_drag = max(viscous_drag, 0.0)
        self.settle_speed = max(settle_speed, 0.0)
        self.gravity_wells: List[GravityWell] = list(gravity_wells or [])

    def step(self, dt: float = 1.0) -> None:
        """Advance one simulation step."""

        forces = [(0.0, 0.0) for _ in self.satellites]

        # 1) Pairwise repulsion
        n = len(self.satellites)
        for i in range(n):
            for j in range(i + 1, n):
                fx, fy = self._repel_force(self.satellites[i], self.satellites[j])
                forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

        # 2) Additional gravity wells
        if self.gravity_wells:
            for i, satellite in enumerate(self.satellites):
                gfx, gfy = self._gravity_wells_force(satellite)
                forces[i] = (forces[i][0] + gfx, forces[i][1] + gfy)

        # 3) Soft boundary push
        if self.boundary_k > 0.0:
            for i, satellite in enumerate(self.satellites):
                bfx, bfy = self._boundary_force(satellite)
                forces[i] = (forces[i][0] + bfx, forces[i][1] + bfy)

        # 4) Global viscous drag (linear with velocity)
        if self.viscous_drag > 0.0:
            for i, satellite in enumerate(self.satellites):
                forces[i] = (
                    forces[i][0] - self.viscous_drag * satellite.vx,
                    forces[i][1] - self.viscous_drag * satellite.vy,
                )

        # 5) Integrate
        for (fx, fy), satellite in zip(forces, self.satellites):
            ax = fx / satellite.mass
            ay = fy / satellite.mass
            jitter_x = 0.0
            jitter_y = 0.0
            if self.noise_sigma > 0.0:
                rng = self._noise_rng or random
                scale = self.noise_sigma * math.sqrt(max(dt, 1e-6))
                jitter_x = rng.gauss(0.0, scale)
                jitter_y = rng.gauss(0.0, scale)

            satellite.vx = (satellite.vx + ax * dt + jitter_x) * self.damping
            satellite.vy = (satellite.vy + ay * dt + jitter_y) * self.damping
            satellite.x += satellite.vx * dt
            satellite.y += satellite.vy * dt

            if self.settle_speed > 0.0:
                speed = math.hypot(satellite.vx, satellite.vy)
                if speed < self.settle_speed:
                    satellite.vx = 0.0
                    satellite.vy = 0.0

            satellite.x = min(max(satellite.x, 0.0), float(self.W))
            satellite.y = min(max(satellite.y, 0.0), float(self.H))

    def _repel_force(self, s1: Satellite, s2: Satellite) -> Tuple[float, float]:
        dx = s1.x - s2.x
        dy = s1.y - s2.y
        r2 = dx * dx + dy * dy + 1e-6
        r = math.sqrt(r2)
        f_mag = self.repel_k * s1.mass * s2.mass / r2
        return f_mag * dx / r, f_mag * dy / r

    def _boundary_force(self, satellite: Satellite) -> Tuple[float, float]:
        if self.boundary_k <= 0.0 or self.boundary_slope <= 0.0:
            return 0.0, 0.0

        span_x = max(self.boundary_margin, self.boundary_span_ratio * float(self.W))
        span_y = max(self.boundary_margin, self.boundary_span_ratio * float(self.H))
        sat_mass = max(satellite.mass, 1e-6)

        def smoothstep(t: float) -> float:
            return t * t * (3.0 - 2.0 * t)

        def slope_push(dist: float, span: float, direction: float) -> float:
            span = max(span, 1e-6)
            if dist >= span:
                return 0.0
            normalized = 1.0 - (dist / span)
            weight = smoothstep(max(min(normalized, 1.0), 0.0))
            base = self.boundary_k * self.boundary_slope
            return direction * base * weight * sat_mass

        fx = 0.0
        fy = 0.0

        # Left edge pushes +x, right edge pushes -x
        fx += slope_push(satellite.x, span_x, +1.0)
        fx += slope_push(self.W - satellite.x, span_x, -1.0)

        # Top edge pushes +y, bottom edge pushes -y
        fy += slope_push(satellite.y, span_y, +1.0)
        fy += slope_push(self.H - satellite.y, span_y, -1.0)

        return fx, fy

    def _gravity_wells_force(self, satellite: Satellite) -> Tuple[float, float]:
        if not self.gravity_wells:
            return 0.0, 0.0

        fx = 0.0
        fy = 0.0
        for well in self.gravity_wells:
            dx = well.x - satellite.x
            dy = well.y - satellite.y
            r = math.sqrt(dx * dx + dy * dy) + 1e-6
            denom = max(r ** well.falloff, 1e-6)
            f_mag = well.strength / denom
            direction = 1.0 if well.polarity == "attract" else -1.0
            fx += direction * f_mag * dx / r
            fy += direction * f_mag * dy / r
        return fx, fy

    def positions(self) -> List[Tuple[float, float]]:
        return [(satellite.x, satellite.y) for satellite in self.satellites]
