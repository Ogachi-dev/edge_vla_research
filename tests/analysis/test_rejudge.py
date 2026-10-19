import pandas as pd

from evla.analysis.rejudge import ThresholdScale, rejudge_episode, sensitivity_curve
from evla.grasp.judge import GraspThresholds

BASE = GraspThresholds(cube_size_m=0.03, vertical_tol_m=0.01, forward_tol_m=0.01)


def events(offset_x, success=True):
    geo = {"offset_m": (offset_x, 0.0, 0.0), "yaw_err_deg": 0.0, "approach_deg": 0.0, "opening_m": 0.05,
           "q_gripper_rad": 0.0, "close_commanded": True}
    ev = [{"type": "grasp_candidate"}, {"type": "attached", "geometry": geo}]
    return ev + ([{"type": "success"}] if success else [])


def test_scale_one_matches_online():
    assert rejudge_episode(events(0.009), BASE, ThresholdScale()) is True


def test_stricter_threshold_can_fail():
    assert rejudge_episode(events(0.009), BASE, ThresholdScale(position=0.5)) is False
    assert rejudge_episode(events(0.004), BASE, ThresholdScale(position=0.5)) is True


def test_looser_is_unknown_and_failure_stays_failure():
    assert rejudge_episode(events(0.0), BASE, ThresholdScale(position=1.5)) is None
    assert rejudge_episode(events(0.0, success=False), BASE, ThresholdScale()) is False


def test_curve_is_monotone():
    df = pd.DataFrame({"config": ["a"] * 4, "events": [events(x) for x in (0.001, 0.003, 0.006, 0.009)]})
    c = sensitivity_curve(df, BASE, [0.25, 0.5, 0.75, 1.0])
    assert list(c["k"]) == [1, 2, 3, 4]
