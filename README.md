# orbital-layout

A 2D planetary-satellite based layout engine template with room for PyTorch acceleration.

## Installation

```bash
pip install orbital-layout
# or with optional PyTorch support
pip install orbital-layout[torch]
```

## Quick start

```python
from orbital_layout import Planet, Satellite, LayoutSimulator

planet = Planet(x=50.0, y=50.0, mass=10.0)
sats = [Satellite(x=10.0, y=10.0, mass=1.0)]
sim = LayoutSimulator(planet, sats, canvas_size=(100, 100))

for _ in range(100):
    sim.step(dt=0.1)

print(sim.positions())
```

## Force model

The current release keeps only three force fields so the layout behavior stays easy to reason about:

1. **Inverse-square satellite repulsion** – higher `repel_k` pushes nodes apart more aggressively.
2. **Configurable gravity wells (`GravityWell`)** – set `polarity="repel"` for a “hill” or `"attract"` for a traditional well to pull nodes toward or push them away from a region.
3. **Soft canvas boundary** – `boundary_k` controls how strongly the border pushes back, while `boundary_margin` defines how far from the edge the linear falloff starts, keeping nodes inside smoothly.

```python
from orbital_layout import GravityWell

sim = LayoutSimulator(
    planet,
    sats,
    canvas_size=(100, 100),
    repel_k=35.0,
    boundary_k=60.0,
    boundary_margin=12.0,
    gravity_wells=[
        GravityWell(x=planet.x, y=planet.y, strength=180.0, falloff=2.2, polarity="repel"),
    ],
)
```

To help the orbits settle more quickly, tweak:

* `viscous_drag`: linear velocity damping (force proportional to speed); larger values bleed momentum faster.
* `settle_speed`: when the speed magnitude drops below this threshold the integrator zeros `vx/vy`, preventing tiny jitters near wells.

## Satellite mass

`Satellite.mass` defaults to 1.0, but every force still follows $F = m a$:

* Repulsion and gravity-well forces multiply by satellite mass, so heavier nodes shove others away more strongly.
* Any force is divided by satellite mass to produce acceleration, so heavier nodes react more slowly under the same external force.
* Passing custom `mass` values to `_build_even_satellites` lets you mimic sluggish “heavy” nodes or nimble “light” ones.

Example:

```python
satellites = [
    Satellite(x=30, y=30, mass=5.0),  # heavy, drifts slowly
    Satellite(x=70, y=30, mass=1.0),  # light, gets pushed around easily
]
```

The current repulsion formula depends only on distance, but dividing by mass on integration still captures the “heavy is steadier, light is nimble” behavior.

## Development

```bash
pip install -e .[dev,torch]
pytest -q
```

## Visualization demo

Static CPU visualization (served from any basic local HTTP server):

```bash
python examples/generate_positions.py
cd examples
python -m http.server 8000
```

Then visit <http://localhost:8000/visualizer.html> in your browser to inspect the animation (driven by `sim_data.json`).

## Roadmap

1. Re-implement `LayoutSimulator` with PyTorch tensors and GPU acceleration.
2. Add `examples/` or `notebooks/` with visualization demos.
3. Extend the force model for real-world text/layout constraints.
