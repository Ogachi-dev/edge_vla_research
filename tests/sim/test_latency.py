import numpy as np

from evla.sim.latency import ConstantLatency, EmpiricalLatency, MeasuredLatency, make_latency


def test_measured():
    assert MeasuredLatency(scale=2.0, add_ms=10).sample(100.0, np.random.default_rng(0)) == 210.0


def test_constant_ignores_measured():
    assert ConstantLatency(500).sample(1.0, np.random.default_rng(0)) == 500


def test_empirical_draws_from_samples():
    m = EmpiricalLatency((1.0, 2.0, 3.0))
    rng = np.random.default_rng(0)
    xs = {m.sample(0, rng) for _ in range(100)}
    assert xs == {1.0, 2.0, 3.0}


def test_make_latency():
    assert make_latency({"type": "constant", "ms": 5}) == ConstantLatency(5)
