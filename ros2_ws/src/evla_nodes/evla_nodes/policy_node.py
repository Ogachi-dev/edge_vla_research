"""実時間モードのポリシーノード（推論は別スレッド、行動はチャンクごとの JointTrajectory）。docs/track2/37 4.2

時間はすべて sim 時刻で数える（use_sim_time:=true）。推論中もロボットはキューの軌道を実行し続ける（async）。
キューの残りは「最後に送った軌道が終わる sim 時刻」から計算する（コントローラが軌道を実行しているので）。
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory

from evla.dataset.conventions import joint_state_to_vector, rad_vector_to_lerobot
from evla.ipc.client import InferenceClient, PredictReply

from .action_mapping import chunk_to_trajectories
from .sim_io import image_to_numpy


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        p = self.declare_parameter
        self.mode = p("mode", "async").value                  # "sync" | "async"
        self.n_exec = p("n_exec", 25).value
        self.thr = p("async_threshold", 10).value if self.mode == "async" else 0
        self.hz = p("control_hz", 30.0).value
        self.task = p("task", "Pick up the cube and put it in the box.").value
        self.client = InferenceClient(p("server_addr", "tcp://127.0.0.1:5600").value,
                                      timeout_ms=p("timeout_ms", 10_000).value)
        self.metrics_csv = Path(p("metrics_csv", "/tmp/policy_metrics.csv").value)

        self.img: dict[str, np.ndarray] = {}
        self.state_deg: np.ndarray | None = None
        self.create_subscription(Image, "/camera/top/image", lambda m: self._on_image("top", m), qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/wrist/image", lambda m: self._on_image("wrist", m), qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.pub_arm = self.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
        self.pub_grip = self.create_publisher(JointTrajectory, "/gripper_controller/joint_trajectory", 10)
        self.pub_metrics = self.create_publisher(String, "/policy/metrics", 10)

        self.pool = ThreadPoolExecutor(max_workers=1)  # 推論は1本ずつ。ROS のコールバックをブロックしない
        self.inflight: Future[PredictReply] | None = None
        self.t_obs = 0.0                 # 推論に使った観測の sim 時刻
        self.queue_end = 0.0             # 最後に送った軌道が終わる sim 時刻
        self.last_send = None
        self.rows: list[dict] = []
        self.create_timer(1.0 / self.hz, self.on_timer)

    def _on_image(self, cam: str, msg: Image) -> None:
        self.img[cam] = image_to_numpy(msg)

    def _on_js(self, msg: JointState) -> None:
        q = joint_state_to_vector(msg.name, msg.position)
        self.state_deg = rad_vector_to_lerobot(q)  # 学習データと同じ単位（度、グリッパ 0〜100）

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def queue_remaining(self, t: float) -> int:
        return max(0, round((self.queue_end - t) * self.hz))  # 【穴①】軌道の残り時間をステップ数に換算

    def on_timer(self) -> None:
        t = self.now()
        # 1) 推論が終わっていれば、軌道にして送る
        if self.inflight is not None and self.inflight.done():  # 【穴②】別スレッドの推論が終わったか（Future のメソッド）
            reply = self.inflight.result()
            self.inflight = None
            k = 0 if self.mode == "sync" else round((t - self.t_obs) * self.hz)  # 推論中に過ぎたステップ数
            chunk = reply.actions[k : k + self.n_exec]
            if len(chunk) > 0:
                arm, grip = chunk_to_trajectories(chunk, self.hz)
                self.pub_arm.publish(arm)
                self.pub_grip.publish(grip)
                self.queue_end = t + len(chunk) / self.hz
            self._log(t, reply, k, len(chunk))
        # 2) キューが減っていて推論中でなければ、今の観測で推論を始める
        if self.inflight is None and self.queue_remaining(t) <= self.thr and self._ready():
            images = {"top": self.img["top"], "wrist": self.img["wrist"]}
            state = self.state_deg.copy()
            self.t_obs = t
            self.inflight = self.pool.submit(self.client.predict, images, state, self.task, t)  # 【穴③】関数と引数を渡して別スレッドで実行させ、Future を受け取るメソッド

    def _ready(self) -> bool:
        return "top" in self.img and "wrist" in self.img and self.state_deg is not None

    def _log(self, t: float, reply: PredictReply, k: int, n_sent: int) -> None:
        row = {"t_sim": t, "t_obs": self.t_obs, "obs_to_cmd_s": t - self.t_obs, "rtt_ms": reply.rtt_ms,
               "server_ms": reply.server_latency_ms, "discarded": k, "n_sent": n_sent,
               "period_s": None if self.last_send is None else t - self.last_send}
        self.last_send = t
        self.rows.append(row)
        self.pub_metrics.publish(String(data=json.dumps(row)))

    def save_metrics(self) -> None:
        if not self.rows:
            return
        with open(self.metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_metrics()
        node.client.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
