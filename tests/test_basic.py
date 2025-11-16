import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orbital_layout import (
    Anchor,
    LayoutSimulator,
    Planet,
    Satellite,
    generate_anchor_grid,
    select_text_anchors,
)
from examples.run_text_layout import _apply_anchor_capture


def test_simulation_runs() -> None:
    planet = Planet(x=50.0, y=50.0, mass=10.0)
    satellites = [
        Satellite(x=10.0, y=10.0, mass=1.0),
        Satellite(x=90.0, y=10.0, mass=1.0),
    ]
    simulator = LayoutSimulator(planet, satellites, canvas_size=(100, 100))

    for _ in range(10):
        simulator.step(dt=0.1)

    positions = simulator.positions()
    assert len(positions) == 2
    for x, y in positions:
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 100.0


def test_generate_anchor_grid_default() -> None:
    anchors = generate_anchor_grid()
    assert len(anchors) == 49
    xs = {round(anchor.x, 3) for anchor in anchors}
    ys = {round(anchor.y, 3) for anchor in anchors}
    assert len(xs) == 7
    assert len(ys) == 7
    assert anchors[0].index == 0
    assert anchors[-1].row == 6 and anchors[-1].col == 6


def test_select_text_anchors_respects_padding() -> None:
    anchors = select_text_anchors(
        "PlayStation Portal",
        font_size=80,
        padding=(30, 15),
        margin=(80, 80),
    )
    # Larger font size shrinks available anchors but should leave at least one row/column
    assert len(anchors) > 0
    for anchor in anchors:
        assert 80 + 30 <= anchor.x <= 1024 - 80 - 30
        assert 80 + 15 <= anchor.y <= 1024 - 80 - 15
        assert isinstance(anchor.index, int)


def test_boundary_slope_pushes_inward_and_blends_at_corners() -> None:
    planet = Planet(x=50.0, y=50.0, mass=5.0)
    simulator = LayoutSimulator(
        planet,
        [],
        canvas_size=(100, 100),
        boundary_k=100.0,
        boundary_slope=0.1,
        boundary_span_ratio=0.5,
    )

    near_left = Satellite(x=5.0, y=50.0)
    mid = Satellite(x=45.0, y=50.0)
    fx_left, _ = simulator._boundary_force(near_left)
    fx_mid, _ = simulator._boundary_force(mid)
    assert fx_left > fx_mid >= 0.0

    corner = Satellite(x=5.0, y=5.0)
    fx_corner, fy_corner = simulator._boundary_force(corner)
    assert fx_corner > 0.0 and fy_corner > 0.0


def test_boundary_acceleration_is_mass_invariant() -> None:
    planet = Planet(x=50.0, y=50.0, mass=5.0)
    simulator = LayoutSimulator(
        planet,
        [],
        canvas_size=(100, 100),
        boundary_k=120.0,
        boundary_slope=0.1,
    )

    light = Satellite(x=4.0, y=40.0, mass=10.0)
    heavy = Satellite(x=4.0, y=40.0, mass=4000.0)
    fx_light, _ = simulator._boundary_force(light)
    fx_heavy, _ = simulator._boundary_force(heavy)

    ax_light = fx_light / light.mass
    ax_heavy = fx_heavy / heavy.mass
    assert math.isclose(ax_light, ax_heavy, rel_tol=1e-6)


def test_anchor_capture_respects_speed_threshold() -> None:
    planet = Planet(x=0.0, y=0.0, mass=10.0)
    satellite = Satellite(x=100.0, y=100.0, vx=1.0, vy=1.0, mass=100.0)
    simulator = LayoutSimulator(planet, [satellite], canvas_size=(400, 400))
    anchor = Anchor(index=0, row=0, col=0, x=110.0, y=110.0)
    anchor_map = {"title": [anchor]}
    captured: dict[str, Anchor] = {}
    labels = ["title"]

    # Slow satellite inside radius should get captured and clamped to anchor
    satellite.x = 109.0
    satellite.y = 109.0
    satellite.vx = 1.0
    satellite.vy = 1.0
    _apply_anchor_capture(simulator, anchor_map, captured, labels, radius=5.0, speed_threshold=3.0)
    assert captured["title"] == anchor
    assert math.isclose(satellite.x, anchor.x)
    assert math.isclose(satellite.y, anchor.y)
    assert satellite.vx == 0.0 and satellite.vy == 0.0

    # Give it a burst of speed; capture should release and leave capture state empty
    satellite.vx = 10.0
    satellite.vy = 0.0
    _apply_anchor_capture(simulator, anchor_map, captured, labels, radius=5.0, speed_threshold=3.0)
    assert "title" not in captured


def test_anchor_capture_uses_global_anchors_when_label_missing() -> None:
    planet = Planet(x=0.0, y=0.0, mass=5.0)
    satellite = Satellite(x=110.0, y=110.0, mass=10.0)
    simulator = LayoutSimulator(planet, [satellite], canvas_size=(400, 400))
    shared_anchor = Anchor(index=5, row=0, col=5, x=112.0, y=112.0)
    anchor_map = {"title": [shared_anchor]}
    capture_state: dict[str, Anchor] = {}
    anchor_claims: dict[Anchor, set[str]] = {}
    labels = ["subtitle"]

    satellite.x = 111.0
    satellite.y = 111.0
    _apply_anchor_capture(
        simulator,
        anchor_map,
        capture_state,
        labels,
        radius=5.0,
        speed_threshold=3.0,
        global_anchors=[shared_anchor],
        anchor_claims=anchor_claims,
        shared_labels={"subtitle"},
    )

    assert capture_state["subtitle"] == shared_anchor
    assert anchor_claims[shared_anchor] == {"subtitle"}


def test_subtitle_can_share_anchor_with_title() -> None:
    planet = Planet(x=0.0, y=0.0, mass=5.0)
    title_sat = Satellite(x=104.0, y=104.0, mass=10.0)
    subtitle_sat = Satellite(x=104.5, y=104.5, mass=5.0)
    simulator = LayoutSimulator(planet, [title_sat, subtitle_sat], canvas_size=(400, 400))
    shared_anchor = Anchor(index=3, row=0, col=3, x=105.0, y=105.0)
    anchor_map = {"title": [shared_anchor], "subtitle": [shared_anchor]}
    capture_state: dict[str, Anchor] = {}
    anchor_claims: dict[Anchor, set[str]] = {}
    labels = ["title", "subtitle"]

    _apply_anchor_capture(
        simulator,
        anchor_map,
        capture_state,
        labels,
        radius=5.0,
        speed_threshold=3.0,
        shared_labels={"subtitle"},
        anchor_claims=anchor_claims,
    )

    assert capture_state["title"] == shared_anchor
    assert capture_state["subtitle"] == shared_anchor
    assert anchor_claims[shared_anchor] == {"title", "subtitle"}
