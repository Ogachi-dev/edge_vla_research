import math

import numpy as np
import pytest

from evla.grasp.geometry import (
    OpeningTable,
    approach_angle_deg,
    fold_deg,
    invert,
    jaw_opening_m,
    pose_to_matrix,
    relative_pose,
    yaw_misalignment_deg,
)


def rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])


def rand_T(rng):
    q = rng.normal(size=4)
    return pose_to_matrix(rng.normal(size=3), q / np.linalg.norm(q))


def test_pose_to_matrix_identity():
    assert np.allclose(pose_to_matrix([1, 2, 3], [0, 0, 0, 1]), [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])


def test_quat_z90():
    s = math.sqrt(0.5)
    assert np.allclose(pose_to_matrix([0, 0, 0], [0, 0, s, s])[:3, :3], rz(90))


def test_relative_pose_properties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b, c = rand_T(rng), rand_T(rng), rand_T(rng)
        assert np.allclose(relative_pose(a, a), np.eye(4))
        assert np.allclose(invert(a), np.linalg.inv(a))
        assert np.allclose(relative_pose(a, b) @ relative_pose(b, c), relative_pose(a, c))


@pytest.mark.parametrize("deg,expected", [(0, 0), (14.9, 14.9), (44, 44), (46, -44), (90, 0), (-91, -1)])
def test_fold(deg, expected):
    assert fold_deg(deg) == pytest.approx(expected)


@pytest.mark.parametrize("deg,expected", [(0, 0), (14.9, 14.9), (44, 44), (46, -44), (90, 0), (-91, -1)])
def test_yaw_misalignment(deg, expected):
    assert yaw_misalignment_deg(rz(deg)) == pytest.approx(expected, abs=1e-9)


def test_yaw_is_90deg_periodic():
    rng = np.random.default_rng(1)
    for deg in rng.uniform(-180, 180, size=100):
        assert yaw_misalignment_deg(rz(deg)) == pytest.approx(yaw_misalignment_deg(rz(deg + 90)), abs=1e-9)


def test_approach_angle():
    down = np.diag([1.0, -1.0, -1.0])        # z軸が下向き（x軸まわりに180°）
    assert approach_angle_deg(down) == pytest.approx(0.0)
    assert approach_angle_deg(np.eye(3)) == pytest.approx(180.0)


def test_jaw_opening_interp():
    t = OpeningTable((0.0, 1.0), (0.0, 0.05))
    assert jaw_opening_m(0.5, t) == pytest.approx(0.025)
    assert jaw_opening_m(2.0, t) == pytest.approx(0.05)   # 範囲外は端の値
