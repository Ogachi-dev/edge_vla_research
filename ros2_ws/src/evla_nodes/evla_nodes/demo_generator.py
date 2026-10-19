"""MoveItPy で関節空間の移動を計画する（実行はしない）。docs/track2/33_moveit_demo_gen.md

episode_runner から plan_joint_move として渡して使う。
MoveItPy の設定まわりは【要確認】が多い。動かなければ evla.planning.build.joint_interp で先に進めてよい。
"""

from __future__ import annotations

import numpy as np

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.planning.episode_plan import JointTrajectoryData

ARM_GROUP = "arm"


class MoveItJointPlanner:
    def __init__(self, node_name: str = "demo_generator_moveit") -> None:
        from moveit.planning import MoveItPy, PlanRequestParameters  # 【要確認】Jazzy の moveit_py
        from moveit_configs_utils import MoveItConfigsBuilder

        config = (MoveItConfigsBuilder("so101", package_name="lerobot_moveit")  # 【要確認】robot 名と package
                  .to_moveit_configs().to_dict())
        self.moveit = MoveItPy(node_name=node_name, config_dict=config)
        self.arm = ___  # 【穴①】計画グループ "arm" を操作するオブジェクトを得るメソッド
        self.model = self.moveit.get_robot_model()
        self._params_cls = PlanRequestParameters

    def _state(self, q: np.ndarray):
        from moveit.core.robot_state import RobotState

        s = RobotState(self.model)
        s.set_joint_group_positions(ARM_GROUP, np.asarray(q, dtype=float))  # 【要確認】グループに固定関節（base_joint）が入っていても5要素でよいか
        s.update()
        return s

    def __call__(self, q0: np.ndarray, q1: np.ndarray, vel_scale: float) -> JointTrajectoryData:
        self.arm.set_start_state(robot_state=self._state(q0))
        self.arm.set_goal_state(robot_state=___)  # 【穴②】目標を「関節角の状態」で与える（姿勢ではなく）
        params = self._params_cls(self.moveit)
        params.max_velocity_scaling_factor = vel_scale
        params.max_acceleration_scaling_factor = vel_scale
        result = self.arm.plan(single_plan_parameters=params)
        if not result:
            raise RuntimeError("MoveIt の計画に失敗")
        jt = result.trajectory.get_robot_trajectory_msg().joint_trajectory
        names = tuple(jt.joint_names)
        times = tuple(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points)
        pos = tuple(tuple(p.positions) for p in jt.points)
        # 最初の点は時刻0（＝今の位置）なので落とす（送ると「今すぐその位置へ」になり、意味がない）
        return JointTrajectoryData(names, times[1:], pos[1:])
