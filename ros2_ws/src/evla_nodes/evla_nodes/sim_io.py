"""Gazebo とのやり取りの共通部品（episode_runner と eval_runner で使う）。docs/track2/34、38

- 観測の取得（2カメラ、関節状態、コントローラの目標値、キューブの姿勢）
- 指令（腕・グリッパの軌道）
- リセット（キューブ・箱の再配置、判定ノードのリセット、腕を初期姿勢へ）
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import numpy as np
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory

from evla.common.stamped_buffer import wait_until
from evla.dataset.conventions import joint_state_to_vector
from evla.planning.episode_plan import JointTrajectoryData

from .action_mapping import ARM_JOINTS, GRIPPER_JOINTS, make_trajectory, seconds_to_duration
from .lockstep import LockstepClock
from .ros_utils import make_pose, transform_to_pose7

T_TOP, T_WRIST = "/camera/top/image", "/camera/wrist/image"
T_JS, T_CUBE = "/joint_states", "/model/cube/pose"
T_ARM_STATE, T_GRIP_STATE = "/arm_controller/controller_state", "/gripper_controller/controller_state"


def image_to_numpy(msg: Image) -> np.ndarray:
    """sensor_msgs/Image（rgb8）→ (H, W, 3) uint8。cv_bridge を使わずに済ませる。"""
    if msg.encoding != "rgb8":
        raise ValueError(f"rgb8 を想定: {msg.encoding}")
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)  # 【穴①】1行のバイト数は step（幅×3 より大きい場合がある）
    return a[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()


@dataclass
class Frame:
    t_sim: float
    images: dict[str, np.ndarray]
    state_rad: np.ndarray      # (6,) LeRobot 順
    action_rad: np.ndarray     # (6,) コントローラの目標値
    cube_pose_w: np.ndarray    # (7,)


class SimIO:
    def __init__(self, node: Node, world: str = "evla", physics_dt: float = 1 / 600, frame_dt: float = 1 / 30,
                 cube_name: str = "cube", box_name: str = "box") -> None:
        self.node = node
        self.world, self.cube_name, self.box_name = world, cube_name, box_name
        self.clock = LockstepClock(node, world, physics_dt, frame_dt)
        self.clock.track(T_TOP, Image)
        self.clock.track(T_WRIST, Image)
        self.clock.track(T_JS, JointState)
        self.clock.track(T_ARM_STATE, JointTrajectoryControllerState)
        self.clock.track(T_GRIP_STATE, JointTrajectoryControllerState)
        self.clock.track(T_CUBE, TFMessage, stamp_of=lambda m: m.transforms[0].header.stamp)
        self.pub_arm = node.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
        self.pub_grip = node.create_publisher(JointTrajectory, "/gripper_controller/joint_trajectory", 10)
        self.cli_pose = node.create_client(SetEntityPose, f"/world/{world}/set_pose")
        self.cli_judge_reset = node.create_client(Trigger, "/grasp_judge/reset")
        self.events: list[dict] = []
        self._ev_lock = threading.Lock()
        node.create_subscription(String, "/grasp/event", self._on_event, 50)

    # ---- イベント ----
    def _on_event(self, msg: String) -> None:
        with self._ev_lock:
            self.events.append(json.loads(msg.data))

    def take_events(self) -> list[dict]:
        with self._ev_lock:
            ev, self.events = self.events, []
        return ev

    # ---- サービス ----
    def _call(self, cli, req, timeout_s: float = 5.0):
        if not cli.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"service {cli.srv_name} が無い")
        fut = cli.call_async(req)
        if not wait_until(fut.done, timeout_s):
            raise TimeoutError(f"service {cli.srv_name} の応答がない")
        return fut.result()

    def set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> None:
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = Entity.MODEL  # 【穴②】「モデル」を表す Entity の定数
        req.pose = make_pose(x, y, z, yaw)
        if not self._call(self.cli_pose, req).success:
            raise RuntimeError(f"set_pose {name} に失敗")

    def reset_judge(self) -> None:
        self._call(self.cli_judge_reset, Trigger.Request())

    # ---- 指令 ----
    def send_arm(self, traj: JointTrajectoryData) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(traj.joint_names)
        msg.points = make_trajectory(traj.joint_names, np.asarray(traj.positions_rad), 1.0).points
        for pt, t in zip(msg.points, traj.times_s, strict=True):
            pt.time_from_start = seconds_to_duration(t)  # 計画どおりの時刻で上書き
        self.pub_arm.publish(msg)

    def send_gripper(self, target_rad: float, duration_s: float) -> None:
        msg = make_trajectory(GRIPPER_JOINTS, np.array([[target_rad]]), 1.0 / duration_s)
        self.pub_grip.publish(msg)

    def send_arm_point(self, q: np.ndarray, duration_s: float) -> None:
        self.pub_arm.publish(make_trajectory(ARM_JOINTS, np.asarray(q)[None, :], 1.0 / duration_s))

    # ---- 観測 ----
    def step_and_observe(self) -> Frame:
        """1フレーム進めて、その時刻のメッセージがそろうのを待ち、Frame にまとめる。"""
        t = self.clock.step_frame()
        m = self.clock.wait_for(t, [T_TOP, T_WRIST, T_JS, T_CUBE, T_ARM_STATE, T_GRIP_STATE],
                                latest=[T_ARM_STATE, T_GRIP_STATE, T_CUBE])
        js: JointState = m[T_JS]
        arm: JointTrajectoryControllerState = m[T_ARM_STATE]
        grip: JointTrajectoryControllerState = m[T_GRIP_STATE]
        ref_names = list(arm.joint_names) + list(grip.joint_names)
        ref_pos = list(arm.reference.positions) + list(grip.reference.positions)  # 【穴③】コントローラが今追従しようとしている目標位置
        return Frame(
            t_sim=t,
            images={"top": image_to_numpy(m[T_TOP]), "wrist": image_to_numpy(m[T_WRIST])},
            state_rad=joint_state_to_vector(js.name, js.position),
            action_rad=joint_state_to_vector(ref_names, ref_pos),
            cube_pose_w=transform_to_pose7(m[T_CUBE].transforms[0].transform),
        )

    def step_frames(self, n: int) -> None:
        for _ in range(n):
            self.clock.step_frame()

    def reset_scene(self, cube_xy, cube_yaw: float, cube_z: float, box_xy, q_init, gripper_open_rad: float,
                    settle_s: float = 1.5) -> None:
        """pause → 判定ノードのリセット（detach）→ キューブと箱を配置 → 腕を初期姿勢へ → 静止を待つ。記録しない。"""
        self.clock.pause()
        self.reset_judge()
        self.step_frames(2)  # detach を反映させる
        self.set_pose(self.box_name, box_xy[0], box_xy[1], 0.0, 0.0)
        self.set_pose(self.cube_name, cube_xy[0], cube_xy[1], cube_z, cube_yaw)
        self.send_arm_point(np.asarray(q_init), 1.0)
        self.send_gripper(gripper_open_rad, 0.5)
        self.step_frames(int(settle_s * 30))
        self.take_events()  # リセット中のイベントは捨てる
