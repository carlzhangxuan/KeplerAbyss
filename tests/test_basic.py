from orbital_layout import LayoutSimulator, Planet, Satellite


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
