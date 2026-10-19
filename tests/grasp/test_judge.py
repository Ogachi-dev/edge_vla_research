import numpy as np
import pytest

from evla.grasp.judge import (
    GraspGeometry,
    GraspState,
    GraspStateMachine,
    GraspThresholds,
    SuccessCriteria,
    check_grasp,
    cube_in_box,
)

TH = GraspThresholds(cube_size_m=0.03, vertical_tol_m=0.01, forward_tol_m=0.01, yaw_tol_deg=15,
                     approach_tol_deg=30, release_open_rad=0.5)
SC = SuccessCriteria(box_inner_xy_m=(0.12, 0.12), box_center_xy_m=(0.2, 0.2), rim_z_m=0.05, still_frames=3)
EPS = 1e-6


def geo(offset=(0, 0, 0), yaw=0.0, approach=0.0, opening=0.05, q=0.0, close=True):
    return GraspGeometry(offset_m=offset, yaw_err_deg=yaw, approach_deg=approach, opening_m=opening,
                         q_gripper_rad=q, close_commanded=close)


def test_centered_grasp_ok():
    assert check_grasp(geo(), TH).ok


@pytest.mark.parametrize("axis,tol,name", [(0, 0.01, "forward"), (2, 0.01, "vertical"), (1, 0.01, "open_axis")])
def test_offset_boundaries(axis, tol, name):
    # opening 0.05, cube 0.03 → 開閉方向の許容 0.01
    inside = [0.0, 0.0, 0.0]
    inside[axis] = tol - EPS
    outside = [0.0, 0.0, 0.0]
    outside[axis] = -(tol + EPS)
    assert check_grasp(geo(offset=tuple(inside)), TH).ok
    d = check_grasp(geo(offset=tuple(outside)), TH)
    assert not d.ok and name in d.failed


def test_yaw_and_approach_and_not_closing():
    assert "yaw" in check_grasp(geo(yaw=15 + EPS), TH).failed
    assert check_grasp(geo(yaw=-15 + EPS), TH).ok
    assert "approach" in check_grasp(geo(approach=31), TH).failed
    assert "not_closing" in check_grasp(geo(close=False), TH).failed


def test_approach_condition_can_be_disabled():
    th = GraspThresholds(cube_size_m=0.03, vertical_tol_m=0.01, forward_tol_m=0.01, approach_tol_deg=None)
    assert check_grasp(geo(approach=80), th).ok


def test_cube_in_box():
    assert cube_in_box([0.2, 0.2, 0.02], SC)
    assert not cube_in_box([0.2 + 0.061, 0.2, 0.02], SC)
    assert not cube_in_box([0.2, 0.2, 0.06], SC)


def test_state_machine_happy_path():
    sm = GraspStateMachine(TH, SC)
    p_box = np.array([0.2, 0.2, 0.015])
    out = sm.update(0.0, geo(close=False), np.zeros(3), False)
    assert out.command is None and sm.state is GraspState.FREE
    out = sm.update(0.1, geo(close=True), np.zeros(3), False)
    assert out.command == "attach" and sm.state is GraspState.ATTACH_PENDING
    assert [e["type"] for e in out.events] == ["grasp_candidate", "attach_sent"]
    sm.update(0.2, geo(close=True), np.zeros(3), True)
    assert sm.state is GraspState.ATTACHED
    out = sm.update(0.3, geo(close=False, q=0.6), p_box, True)
    assert out.command == "detach"
    sm.update(0.4, geo(close=False, q=0.6), p_box, False)
    assert sm.state is GraspState.RELEASED
    results = [sm.update(0.5 + i * 0.03, geo(close=False, q=0.6), p_box, False).success for i in range(6)]
    assert True in results
    assert results.count(True) == 1           # 成功は1回だけ出る


def test_state_machine_rejected_candidate_logged_once():
    sm = GraspStateMachine(TH, SC)
    sm.update(0.0, geo(close=False), np.zeros(3), False)
    out = sm.update(0.1, geo(offset=(0.05, 0, 0), close=True), np.zeros(3), False)
    assert out.command is None
    assert out.events[0]["type"] == "grasp_candidate"
    assert "forward" in out.events[0]["decision"]["failed"]
    out = sm.update(0.2, geo(offset=(0.05, 0, 0), close=True), np.zeros(3), False)
    assert out.events == ()                  # 閉じたままなら2回目は出さない
