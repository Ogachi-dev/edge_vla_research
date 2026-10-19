import math

import numpy as np

from evla.planning.randomize import sample_params, sample_scene

RANGES = {
    "cube_xy_m": {"x": [0.15, 0.28], "y": [-0.12, 0.12]},
    "cube_yaw_deg": [-45, 45],
    "box_xy_m": {"x": [0.15, 0.25], "y": [0.15, 0.22]},
    "min_cube_box_dist_m": 0.10,
    "cube_colors": [[1, 0, 0], [0, 0, 1]],
    "q_home_rad": [0, 0, 0, 0, 0],
    "q_init_noise_rad": 0.05,
    "approach_height_m": [0.06, 0.10],
    "vel_scale": [0.3, 0.6],
    "acc_scale": [0.3, 0.6],
    "perturb_prob": 0.3,
    "perturb_mag_m": [0.01, 0.02],
    "gripper_close_rad": 0.1,
}


def test_deterministic_with_seed():
    a = sample_scene(np.random.default_rng(42), RANGES)
    b = sample_scene(np.random.default_rng(42), RANGES)
    assert a == b


def test_ranges_and_min_distance():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample_scene(rng, RANGES)
        assert 0.15 <= s.cube_xy_m[0] <= 0.28
        assert math.dist(s.cube_xy_m, s.box_xy_m) >= 0.10
        assert -math.pi / 4 <= s.cube_yaw_rad <= math.pi / 4


def test_perturb_rate():
    rng = np.random.default_rng(0)
    n = sum(any(sample_params(rng, RANGES).grasp_offset_m) for _ in range(2000))
    assert 0.25 < n / 2000 < 0.35
