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
3. **Soft canvas boundary** – `boundary_k` now acts like the mass of a virtual “force wall.”
    * `boundary_slope` (default `0.1` ⇒ 10 %) sets the incline of that wall so nodes feel as if they’re rolling down a slope into the canvas.
    * Forces scale with satellite mass internally so the resulting acceleration stays constant—heavy and light labels both roll back inward at the same rate.
    * `boundary_span_ratio` (default `0.5`) lets the wall extend inwards for half of each side before flattening out, and `boundary_margin` still overrides it when you want an even larger soft zone.
    * Corners naturally blend both axes, so satellites near `(0, 0)` feel pushes from the left and top walls simultaneously.

```python
from orbital_layout import GravityWell

sim = LayoutSimulator(
    planet,
    sats,
    canvas_size=(100, 100),
    repel_k=35.0,
    boundary_k=100.0,  # behaves like a 100-mass virtual wall
    boundary_margin=12.0,
    boundary_slope=0.1,
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

## Text mask layout demo

The repo now includes a JSON-driven helper that turns a mask payload (see `examples/input/text_layout_jp.json`) into concrete title/subtitle coordinates using the same force model:

```bash
python examples/run_text_layout.py \
    --input examples/input/text_layout_jp.json \
    --output examples/text_layout_positions.json
```

By default the script:

* Treats `mask_canvas_size` as the simulator canvas.
* Uses the `mask_bbox` center/area as the planet position and mass.
* Seeds two satellites (title, subtitle) at the bbox top-left, offset vertically by 10 px, with masses derived from each text block’s `area`.
* Kicks both satellites toward the planet center with configurable speed (`--initial-speed`, default `30.0`) so they glide into the mask before gravity/repulsion take over.

The resulting JSON mirrors the input labels and records both initial and stabilized coordinates, so you can feed it back into your creative tooling.

Want to preview the motion on the existing SVG visualizer? Ask the script to emit a `sim_data`-style timeline and point the viewer at it:

```bash
python examples/run_text_layout.py \
    --input examples/input/text_layout_jp.json \
    --timeline-output examples/text_layout_sim_data.json \
    --timeline-interval 2
cd examples
python -m http.server 8000
```

Then open <http://localhost:8000/visualizer.html?data=text_layout_sim_data.json>. The `--timeline-interval` flag lets you down-sample long runs (recording every 2nd/5th/etc. frame) so the JSON stays lightweight.

The runner now also projects the discrete anchor grid (generated via `select_text_anchors`) onto the mask and turns every candidate point into a gentle attractor. After the physics simulation settles, the title/subtitle snap to the closest anchor, and the serialized JSON includes the matched anchor metadata. Pass `--no-anchor-wells` if you prefer the old free-form behavior.

> **全局吸附提示**：即便命令行没有为 `subtitle` 显式生成锚点，模拟器现在也会让所有卫星共享同一批 anchor：
>
> * Anchor wells / damping 会对所有标签生效，只要场景里存在 anchor；
> * 捕获逻辑会在运行期间自动分配未被占用的锚点；
> * 如果某个卫星在积分阶段始终没进入吸附半径，收尾阶段也会强制把它贴到最近的空锚点上，确保最终输出一定落在 anchor 上。

### Force field heatmap for debugging

Curious why a node refuses to converge? The `examples/visualize_force_field.py` helper replays the same simulation parameters and samples the combined field (planet hill, anchor wells, boundary slope, and pairwise repulsion) across the canvas, coloring strong regions brightly and overlaying the final satellite/anchor positions:

```bash
python examples/visualize_force_field.py \
    --input examples/input/text_layout_jp.json \
    --output examples/force_field_heatmap.png \
    --grid-cols 200
```

The script exports a PNG heatmap (default `examples/force_field_heatmap.png`) plus an optional quiver overlay so you can see the field direction at a glance. Reuse the same CLI flags as `run_text_layout.py` (`--no-subtitle`, `--anchor-labels`, `--pairwise-repel-scale`, etc.) to inspect alternate setups.

### Anchor helper for static layouts

When you just need discrete anchor positions (for instance, picking safe starting points for titles) you can use the new helpers:

```python
from orbital_layout import generate_anchor_grid, select_text_anchors

anchors = generate_anchor_grid(
    canvas_size=(1024, 1024),
    grid_dims=(7, 7),
    margin=(100, 100),
)

# Filter anchors that can fit a single-line string at the given font size + padding
available = select_text_anchors(
    "リモートプレイ専用端末",
    font_size=96,
    canvas_size=(1024, 1024),
    grid_dims=(7, 7),
    margin=(100, 100),
    padding=(32, 16),
)

for anchor in available:
    print(anchor.index, anchor.row, anchor.col, (anchor.x, anchor.y))
```

The generator tiles the canvas uniformly (default 7×7 = 49 anchors) inside the specified margin, and every `Anchor` carries its zero-based index plus `(row, col)` so you can map back to grid coordinates (e.g., anchor 11 ⇒ row `11 // 7`, col `11 % 7`). The selector removes anchors whose text bounding box (estimated from `len(text) * font_size`) would collide with the reserved whitespace. No line-wrapping is performed—the helper assumes a single line of text and lets you tune the `glyph_width_factor` and padding if your font behaves differently.

## Batch text mask API

When you only need rasterized masks (no physics), the new high-level API wraps Cairo/Pillow rendering so you can describe candidates declaratively:

```python
from pathlib import Path
from orbital_layout import TextCandidate, render_text_candidates

summary = render_text_candidates(
    prj_id="campaign_001",
    candidates=[
        TextCandidate(
            name="title",
            text="プレイステーションポータル",
            font_path=Path("fonts/NotoSansJP-Bold.ttf"),
            font_size=72,
        ),
        TextCandidate(
            name="subtitle",
            text="リモートプレイ専用端末",
            font_path=Path("fonts/NotoSansJP-Regular.ttf"),
            font_size=48,
        ),
    ],
    canvas_size=(1024, 1024),
    output_root=Path("output"),
    anchor_grid_dims=(15, 7),
    anchor_border_ratio=0.05,
)

print(summary["candidates_json_path"])  # -> output/campaign_001/candidates.json
```

Each candidate writes two PNGs under `output/<prj_id>/`: the full canvas mask and a tightly-cropped binary mask (values `0` or `255`) for easier downstream compositing. The returned summary dictionary (also written to `candidates.json`) echoes the input config plus `bbox`, `area`, and the saved file paths, and now includes `anchor_candidates`: every grid cell (default 15×7) whose center can host the fitted mask without crossing the 5 % border. When no anchor fits, we still record the closest grid cell with `"fits": false` so downstream tools have a fallback center. To visualize the options quickly, each entry also records `anchor_visualization_path`, pointing at a 1024×1024 scatter plot (red dots) of the usable anchors.

Prefer a CLI? Point the helper at a JSON spec (see `examples/input/text_candidates_demo.json`) and it will emit the same assets:

```bash
python examples/run_text_candidates.py \
    examples/input/text_candidates_demo.json \
    --output-root output \
    --anchor-cols 15 \
    --anchor-rows 7 \
    --anchor-border 0.05
```
Pass `--no-anchor-candidates` if you only need the mask assets, or `--no-anchor-map` to skip the anchor visualization PNGs.

## Roadmap

1. Re-implement `LayoutSimulator` with PyTorch tensors and GPU acceleration.
2. Add `examples/` or `notebooks/` with visualization demos.
3. Extend the force model for real-world text/layout constraints.
