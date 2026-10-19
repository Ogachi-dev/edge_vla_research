"""把持判定ノード（I/O だけ。判定は evla.grasp の純粋関数）。docs/track2/32_grasp_judge_node.md 5 Step 6

購読: /joint_states, /model/cube/pose（TFMessage）, /grasp/state, /gripper_controller/controller_state
配信: /grasp/attach, /grasp/detach（Empty）, /grasp/event（String, JSON）
サービス: ~/reset（std_srvs/Trigger）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.grasp.events import dumps
from evla.grasp.geometry import OpeningTable
from evla.grasp.judge import GraspStateMachine, GraspThresholds, SuccessCriteria, compute_geometry

from .ros_utils import transform_to_matrix


class GraspJudgeNode(Node):
    def __init__(self) -> None:
        super().__init__("grasp_judge")
        p = self.declare_parameter  # 値は config/grasp.yaml から（launch で渡す）
        self.world_frame = p("world_frame", "world").value
        self.tcp_frame = p("tcp_frame", "tcp").value
        self.cube_frame = p("cube_frame", "cube").value     # 【要確認】Pose_V → TFMessage の child_frame_id
        self.gripper_joint = p("gripper_joint", "6").value
        self.th = GraspThresholds(
            cube_size_m=p("cube_size_m", 0.03).value,
            vertical_tol_m=p("vertical_tol_m", 0.01).value,
            forward_tol_m=p("forward_tol_m", 0.01).value,
            yaw_tol_deg=p("yaw_tol_deg", 15.0).value,
            approach_tol_deg=p("approach_tol_deg", 30.0).value,
            gripper_closed_rad=p("gripper_closed_rad", 0.0).value,
            release_open_rad=p("release_open_rad", 0.5).value,
        )
        self.sc = SuccessCriteria(
            box_inner_xy_m=tuple(p("box_inner_xy_m", [0.12, 0.12]).value),
            box_center_xy_m=tuple(p("box_center_xy_m", [0.2, 0.18]).value),
            rim_z_m=p("rim_z_m", 0.05).value,
        )
        csv_path = p("opening_table_csv", "").value
        if not csv_path:
            raise ValueError("grasp.yaml の opening_table_csv（32章 Step 2 で作る開き幅の表）を設定する")
        self.table = OpeningTable.from_csv(Path(csv_path))
        self.event_log = open(p("event_log", "/tmp/grasp_events.jsonl").value, "a")  # noqa: SIM115  ノードの寿命の間開いておく

        self.sm = GraspStateMachine(self.th, self.sc)
        self.q_gripper: float | None = None
        self.q_gripper_ref: float | None = None
        self.T_w_cube: np.ndarray | None = None
        self.gz_attached = False

        self.tf_buffer = Buffer()
        self.tf_listener = ___  # 【穴①】/tf を購読して Buffer に貯めるオブジェクト
        ___(JointState, "/joint_states", self.on_joint_states, 10)  # 【穴②】トピックを購読するメソッド（型, 名前, コールバック, QoS の深さ）
        self.create_subscription(TFMessage, "/model/cube/pose", self.on_cube_pose, 10)
        self.create_subscription(String, "/grasp/state", self.on_grasp_state, 10)
        self.create_subscription(JointTrajectoryControllerState, "/gripper_controller/controller_state",
                                 self.on_gripper_state, 10)
        self.pub_attach = ___(Empty, "/grasp/attach", 10)  # 【穴③】トピックに配信するためのオブジェクトを作るメソッド
        self.pub_detach = self.create_publisher(Empty, "/grasp/detach", 10)
        self.pub_event = self.create_publisher(String, "/grasp/event", 10)
        self.create_service(Trigger, "~/reset", self.on_reset)
        self.create_timer(1.0 / 30.0, self.on_timer)  # use_sim_time なら sim 時刻で 30Hz

    # ---- コールバック（値を覚えるだけ）----
    def on_joint_states(self, msg: JointState) -> None:
        if self.gripper_joint in msg.name:
            self.q_gripper = msg.position[msg.name.index(self.gripper_joint)]

    def on_cube_pose(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            if t.child_frame_id == self.cube_frame:
                self.T_w_cube = transform_to_matrix(t.transform)

    def on_grasp_state(self, msg: String) -> None:
        self.gz_attached = msg.data == "attached"

    def on_gripper_state(self, msg: JointTrajectoryControllerState) -> None:
        if msg.reference.positions:
            self.q_gripper_ref = msg.reference.positions[0]

    def on_reset(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self.sm.reset()
        self.pub_detach.publish(Empty())
        res.success = True
        return res

    # ---- 判定 ----
    def on_timer(self) -> None:
        if self.q_gripper is None or self.T_w_cube is None or self.q_gripper_ref is None:
            return
        try:
            tf = ___  # 【穴④】world から見た tcp の変換（Time() は「最新」の意味）
        except Exception as e:  # noqa: BLE001  起動直後は TF がまだ無い
            self.get_logger().warn(f"TF 待ち: {e}", throttle_duration_sec=5.0)
            return
        T_w_tcp = transform_to_matrix(tf.transform)
        close_cmd = self.q_gripper_ref <= self.th.gripper_closed_rad  # 【要確認】31章で確かめたグリッパの符号（閉じる向き）に合わせる
        g = compute_geometry(T_w_tcp, self.T_w_cube, self.q_gripper, close_cmd, self.table)
        t_sim = self.get_clock().now().nanoseconds / 1e9
        out = self.sm.update(t_sim, g, self.T_w_cube[:3, 3], self.gz_attached)
        if out.command == "attach":
            self.pub_attach.publish(Empty())
        elif out.command == "detach":
            self.pub_detach.publish(Empty())
        for ev in out.events:
            s = dumps(ev)
            self.pub_event.publish(String(data=s))
            self.event_log.write(s + "\n")
            self.event_log.flush()
            self.get_logger().info(f"event: {ev['type']}")


def main() -> None:
    rclpy.init()
    node = GraspJudgeNode()
    try:
        ___  # 【穴⑤】コールバックを処理し続ける（Ctrl+C まで戻らない）
    except KeyboardInterrupt:
        pass
    finally:
        node.event_log.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
