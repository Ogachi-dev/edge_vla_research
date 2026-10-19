import numpy as np
import pytest

from evla.policy.equivalence import compare_chunks, make_fixed_noise


def test_identical():
    a = np.random.default_rng(0).normal(size=(50, 7))
    r = compare_chunks(a, a)
    assert r.max_abs == 0 and r.rmse == 0 and r.cosine == pytest.approx(1.0) and r.gripper_sign_flips == 0


def test_per_dim_and_flips():
    a = np.ones((50, 7))
    b = a.copy()
    b[:, 6] = -1          # グリッパの符号が全部逆
    b[:, 0] += 0.1
    r = compare_chunks(a, b)
    assert r.gripper_sign_flips == 50
    assert r.per_dim_max_abs[0] == pytest.approx(0.1)
    assert r.per_dim_max_abs[6] == pytest.approx(2.0)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        compare_chunks(np.zeros((50, 7)), np.zeros((50, 6)))


def test_fixed_noise_deterministic():
    assert np.array_equal(make_fixed_noise(50, 32, 1), make_fixed_noise(50, 32, 1))
    assert make_fixed_noise(50, 32, 1).dtype == np.float32
