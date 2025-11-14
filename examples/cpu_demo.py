"""Quick CPU demo for the orbital_layout LayoutSimulator."""

from __future__ import annotations

import time

from orbital_layout import GravityWell, LayoutSimulator, Planet, Satellite


def main() -> None:
    planet = Planet(x=50.0, y=50.0, mass=25.0)
    satellites = [
        Satellite(x=15.0, y=20.0, vx=4.0, vy=0.0, mass=3.0),
        Satellite(x=85.0, y=20.0, vx=2.5, vy=0.0, mass=2.0),
        Satellite(x=20.0, y=80.0, vx=3.5, vy=0.0, mass=1.5),
        Satellite(x=80.0, y=75.0, vx=2.0, vy=0.0, mass=2.5),
    ]

    simulator = LayoutSimulator(
        planet,
        satellites,
        canvas_size=(100, 100),
        repel_k=50.0,
        damping=0.92,
        boundary_k=80.0,
        boundary_margin=12.0,
        gravity_wells=[
            GravityWell(
                x=planet.x,
                y=planet.y,
                strength=180.0,
                falloff=2.2,
                polarity="repel",
            )
        ],
    )

    print("Initial positions:")
    print(simulator.positions())

    for step in range(1, 51):
        simulator.step(dt=0.25)
        if step % 10 == 0:
            print(f"After {step} steps: {simulator.positions()}")
            time.sleep(0.05)

    print("Final positions:")
    for idx, (x, y) in enumerate(simulator.positions()):
        print(f"  Satellite {idx}: ({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
