import math

import numpy as np
import pytest

from evla.planning.build import PlanConfig, UnreachableError, build_plan, joint_interp
from evla.planning.episode_plan import ArmSegment
from evla.planning.ik_so101 import fk_tcp
from evla.planning.randomize import SceneSample, TrajectoryParams

from .test_ik_so101 import GEO

SCENE = SceneSample(cube_xy_m=(0.2, -0.05), cube_yaw_rad=0.3, cube_color=(1, 0, 0), box_xy_m=(0.2, 0.15),
                    q_init_rad=(0.0, 0.0, 0.0, 0.0, 0.0))


def params(offset=(0.0, 0.0, 0.0)):
    return TrajectoryParams(approach_height_m=0.08, vel_scale=0.5, acc_scale=0.5, planner_seed=0,
                            grasp_offset_m=offset, gripper_close_rad=0.1)


def test_plan_order_and_geometry():
    plan = build_plan(0, SCENE, params(), GEO, PlanConfig(), joint_interp)
    names = [s.name for s in plan.segments]
    assert names == ["approach", "descend", "close", "lift", "transfer", "lower", "open", "retreat", "wait"]
    descend = plan.segments[1]
    assert isinstance(descend, ArmSegment)
    x, y, z, phi, _ = fk_tcp(descend.trajectory.positions_rad[-1], GEO)
    assert np.allclose([x, y, z], [0.2, -0.05, 0.015], atol=1e-9)
    assert phi == pytest.approx(-math.pi / 2)
    assert plan.duration_s > 5


def test_perturbation_adds_correct_segment():
    plan = build_plan(0, SCENE, params((0.015, 0.0, 0.0)), GEO, PlanConfig(), joint_interp)
    assert "correct" in [s.name for s in plan.segments]
    assert plan.perturbations[0]["type"] == "grasp_offset"


def test_times_increase():
    plan = build_plan(0, SCENE, params(), GEO, PlanConfig(), joint_interp)
    for s in plan.segments:
        if isinstance(s, ArmSegment):
            assert np.all(np.diff(s.trajectory.times_s) > 0)


def test_unreachable():
    far = SceneSample(cube_xy_m=(0.8, 0.0), cube_yaw_rad=0.0, cube_color=(1, 0, 0), box_xy_m=(0.2, 0.15),
                      q_init_rad=(0.0,) * 5)
    with pytest.raises(UnreachableError):
        build_plan(0, far, params(), GEO, PlanConfig(), joint_interp)
