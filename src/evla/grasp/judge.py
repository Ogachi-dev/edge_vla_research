"""把持・解放・成功の判定と状態機械（ROS を知らない純粋なロジック）。docs/track2/32_grasp_judge_node.md 4

TCP 座標系の軸の規約（31章 Step 3 で決めたものに合わせて直す）：
    x = 前後（forward）、y = 爪の開閉（open）、z = 接近（approach、爪の先端が向く方向）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.grasp.events import make_event
from evla.grasp.geometry import (
    OpeningTable,
    approach_angle_deg,
    jaw_opening_m,
    relative_pose,
    yaw_misalignment_deg,
)

FORWARD_AXIS, OPEN_AXIS, APPROACH_AXIS = 0, 1, 2


@dataclass(frozen=True)
class GraspThresholds:
    cube_size_m: float
    vertical_tol_m: float            # TCPの接近軸方向
    forward_tol_m: float             # もう1つの軸
    yaw_tol_deg: float = 15.0
    approach_tol_deg: float | None = 30.0    # None で条件4を使わない
    gripper_closed_rad: float = 0.0  # これ以下を「閉じている」とみなす（31章の符号の確認に従う）
    release_open_rad: float = 0.5    # これ以上開いたら解放
    # 開閉方向の許容値は、そのときの開き幅から計算する： (opening - cube_size) / 2


@dataclass(frozen=True)
class GraspGeometry:                 # ログに残す（再判定の材料）
    offset_m: tuple[float, float, float]    # TCP座標系でのキューブ中心のずれ (x, y, z)
    yaw_err_deg: float
    approach_deg: float
    opening_m: float
    q_gripper_rad: float
    close_commanded: bool


@dataclass(frozen=True)
class GraspDecision:
    ok: bool
    failed: tuple[str, ...]          # 満たさなかった条件の名前


def compute_geometry(T_w_tcp: NDArray, T_w_cube: NDArray, q_gripper_rad: float, close_commanded: bool,
                     opening_table: OpeningTable) -> GraspGeometry:
    T_tcp_cube = ___  # 【穴①】TCP から見たキューブの姿勢
    return GraspGeometry(
        offset_m=tuple(float(v) for v in T_tcp_cube[:3, 3]),
        yaw_err_deg=yaw_misalignment_deg(T_tcp_cube[:3, :3], APPROACH_AXIS, OPEN_AXIS),
        approach_deg=approach_angle_deg(T_w_tcp[:3, :3], APPROACH_AXIS),
        opening_m=jaw_opening_m(q_gripper_rad, opening_table),
        q_gripper_rad=float(q_gripper_rad),
        close_commanded=bool(close_commanded),
    )


def open_axis_tol_m(g: GraspGeometry, th: GraspThresholds) -> float:
    """キューブが爪の間に収まるための、開閉方向のずれの許容値。"""
    return ___  # 【穴②】開き幅からキューブ幅を引いた隙間の半分


def check_grasp(g: GraspGeometry, th: GraspThresholds) -> GraspDecision:
    failed = []
    if not g.close_commanded:
        failed.append("not_closing")
    off = np.abs(g.offset_m)
    if ___:  # 【穴③】開閉方向のずれが許容を超えた
        failed.append("open_axis")
    if off[APPROACH_AXIS] > th.vertical_tol_m:
        failed.append("vertical")
    if off[FORWARD_AXIS] > th.forward_tol_m:
        failed.append("forward")
    if ___:  # 【穴④】ヨーのずれが許容を超えた
        failed.append("yaw")
    if th.approach_tol_deg is not None and g.approach_deg > th.approach_tol_deg:
        failed.append("approach")
    return GraspDecision(ok=___, failed=tuple(failed))  # 【穴⑤】失敗した条件が1つも無ければ True


def check_release(q_gripper_rad: float, th: GraspThresholds) -> bool:
    return q_gripper_rad >= th.release_open_rad


@dataclass(frozen=True)
class SuccessCriteria:
    box_inner_xy_m: tuple[float, float]
    box_center_xy_m: tuple[float, float]
    rim_z_m: float
    margin_m: float = 0.0            # 壁からの余裕（キューブの半分など）
    still_frames: int = 15           # N：連続して静止しているフレーム数（30Hzで0.5秒）
    still_tol_m: float = 0.001       # 1フレームあたりの移動量の上限


def cube_in_box(p_w_cube: ArrayLike, sc: SuccessCriteria) -> bool:
    p = np.asarray(p_w_cube, dtype=np.float64)
    half = np.asarray(sc.box_inner_xy_m) / 2 - sc.margin_m
    inside_xy = np.all(___)  # 【穴⑥】箱の中心からのxyのずれが、軸ごとに半幅以内
    return bool(inside_xy and ___)  # 【穴⑦】縁より低い


class GraspState(Enum):
    FREE = auto()
    ATTACH_PENDING = auto()
    ATTACHED = auto()
    DETACH_PENDING = auto()
    RELEASED = auto()


@dataclass(frozen=True)
class StepOutput:
    command: Literal["attach", "detach"] | None = None
    events: tuple[dict[str, Any], ...] = ()
    success: bool | None = None                  # 確定したら True / False


@dataclass
class _Mem:
    prev_close: bool = False
    attach_geometry: GraspGeometry | None = None
    prev_p: NDArray | None = None
    still: int = 0
    finished: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


class GraspStateMachine:
    """入力（幾何量、gzからの接続状態）を受けて、出すべき指令（attach/detach）とイベントを返す。ROSを知らない。

    FREE →（閉じ始めで条件成立）→ ATTACH_PENDING →（gz が attached）→ ATTACHED
         →（開いたら）→ DETACH_PENDING →（gz が detached）→ RELEASED →（静止したら成功/失敗）
    grasp_candidate は「閉じ指令の立ち上がり」ごとに1回、判定結果によらず出す（38章の再判定の材料）。
    """

    def __init__(self, th: GraspThresholds, sc: SuccessCriteria) -> None:
        self.th, self.sc = th, sc
        self.reset()

    def reset(self) -> None:
        self.state = GraspState.FREE
        self.m = _Mem()

    def update(self, t_sim: float, g: GraspGeometry, p_w_cube: NDArray, gz_attached: bool) -> StepOutput:
        ev: list[dict[str, Any]] = []
        cmd: Literal["attach", "detach"] | None = None
        success: bool | None = None
        rising = ___  # 【穴⑧】閉じ指令の「立ち上がり」（前回は閉じていなくて、今回閉じた）
        self.m.prev_close = g.close_commanded

        if self.state is GraspState.FREE:
            if rising:
                d = check_grasp(g, self.th)
                ev.append(make_event("grasp_candidate", t_sim, geometry=asdict(g), decision=asdict(d),
                                     thresholds=asdict(self.th)))
                if d.ok:
                    cmd = "attach"
                    self.m.attach_geometry = g
                    self.state = ___  # 【穴⑨】attach を送ったあと、gz の返事を待つ状態
                    ev.append(make_event("attach_sent", t_sim))
        elif self.state is GraspState.ATTACH_PENDING:
            if gz_attached:
                self.state = GraspState.ATTACHED
                ev.append(make_event("attached", t_sim, geometry=asdict(self.m.attach_geometry)))
        elif self.state is GraspState.ATTACHED:
            if check_release(g.q_gripper_rad, self.th):
                cmd = "detach"
                self.state = GraspState.DETACH_PENDING
                ev.append(make_event("detach_sent", t_sim))
        elif self.state is GraspState.DETACH_PENDING:
            if ___:  # 【穴⑩】gz が「外れた」と返してきた
                self.state = GraspState.RELEASED
                ev.append(make_event("released", t_sim, cube_pos=np.asarray(p_w_cube).tolist()))
        elif self.state is GraspState.RELEASED and not self.m.finished:
            p = np.asarray(p_w_cube, dtype=np.float64)
            moved = np.inf if self.m.prev_p is None else float(np.linalg.norm(p - self.m.prev_p))
            self.m.still = ___  # 【穴⑪】動いていなければ数を1増やし、動いたら0に戻す
            self.m.prev_p = p
            if self.m.still >= self.sc.still_frames:
                success = cube_in_box(p, self.sc)
                self.m.finished = True
                ev.append(make_event("success" if success else "failure", t_sim, cube_pos=p.tolist(),
                                     reason=None if success else "dropped_or_misplaced"))
        return StepOutput(command=cmd, events=tuple(ev), success=success)
