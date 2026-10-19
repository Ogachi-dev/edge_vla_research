"""Gazebo 閉ループ評価のエピソード管理。docs/track2/38_gazebo_eval.md 5 Step 1

mode=lockstep: このノードが時計の所有者。観測 → 推論サーバー → 軌道 → n_exec フレーム step、を繰り返す。
mode=realtime: world を動かし（pause を解除）、制御は policy_node に任せる。このノードはリセットと終了判定だけ。

起動例:
    ros2 run evla_nodes eval_runner --ros-args -p use_sim_time:=true -p mode:=lockstep \
        -p config_name:=so101_ov_cpu -p n_episodes:=50
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.common.run_id import make_run_dir, new_run_id
from evla.dataset.conventions import rad_vector_to_lerobot
from evla.grasp.events import classify_failure
from evla.ipc.client import InferenceClient
from evla.planning.randomize import sample_scene

from .action_mapping import chunk_to_trajectories
from .sim_io import SimIO

FPS = 30
EVAL_SEED_BASE = 900_000   # 学習（データ生成）と重ならないシード


class EvalRunner(Node):
    def __init__(self) -> None:
        super().__init__("eval_runner")
        p = self.declare_parameter
        self.mode = p("mode", "lockstep").value
        self.config_name = p("config_name", "unnamed").value
        self.n_episodes = p("n_episodes", 50).value
        self.n_exec = p("n_exec", 25).value
        self.max_s = p("max_episode_s", 30.0).value
        self.task = p("task", "Pick up the cube and put it in the box.").value
        self.cube_size = p("cube_size_m", 0.03).value
        self.ranges = yaml.safe_load(Path(p("datagen_config", "configs/datagen/default.yaml").value).read_text())
        self.server_addr = p("server_addr", "tcp://127.0.0.1:5600").value
        self.io: SimIO | None = None

    def setup(self) -> None:
        self.io = SimIO(self)
        if self.mode == "lockstep":
            self.client = InferenceClient(self.server_addr, timeout_ms=600_000)

    def run_episode_lockstep(self) -> list[dict]:
        io = self.io
        assert io is not None
        events: list[dict] = []
        n_infer, lat = 0, []
        f = io.step_and_observe()
        t0 = f.t_sim
        while f.t_sim - t0 < self.max_s:  # sim 時刻での上限
            state = rad_vector_to_lerobot(f.state_rad)
            reply = ___  # 【穴①】今の観測で推論サーバーに問い合わせる（ロックステップなので world は止まったまま）
            n_infer += 1
            lat.append(reply.rtt_ms)
            arm, grip = chunk_to_trajectories(reply.actions[: self.n_exec], FPS)
            io.pub_arm.publish(arm)
            io.pub_grip.publish(grip)
            for _ in range(self.n_exec):
                f = io.step_and_observe()
                events += io.take_events()
            if any(e["type"] in ("success", "failure") for e in events):
                break
        return events + [{"type": "_stats", "n_infer": n_infer, "rtt_p50_ms": float(np.median(lat))}]

    def run_episode_realtime(self) -> list[dict]:
        io = self.io
        assert io is not None
        io.clock._call(pause=False)  # 世界を動かす（policy_node が制御する）
        events: list[dict] = []
        t_wall0 = time.monotonic()
        t_sim0 = io.clock.now_ns / 1e9
        while io.clock.now_ns / 1e9 - t_sim0 < self.max_s:
            time.sleep(0.05)
            events += io.take_events()
            if any(e["type"] in ("success", "failure") for e in events):
                break
        io.clock.pause()
        rtf = ___  # 【穴②】sim 時刻の経過 ÷ 壁時計の経過
        return events + [{"type": "_stats", "rtf": rtf}]

    def run(self) -> None:
        io = self.io
        assert io is not None
        run_dir = make_run_dir(new_run_id("t2", f"gz-{self.mode}", {"config": self.config_name}))
        out = open(run_dir / "episodes.jsonl", "w")  # noqa: SIM115  エピソードごとに追記して flush する
        n_ok = 0
        for i in range(self.n_episodes):
            seed = EVAL_SEED_BASE + i
            s = sample_scene(np.random.default_rng(seed), self.ranges)
            io.reset_scene(s.cube_xy_m, s.cube_yaw_rad, self.cube_size / 2, s.box_xy_m, s.q_init_rad, 1.0)
            events = self.run_episode_lockstep() if self.mode == "lockstep" else self.run_episode_realtime()
            reason = classify_failure(events)
            n_ok += reason == "ok"
            out.write(json.dumps({"config": self.config_name, "mode": self.mode, "seed": seed,
                                  "success": reason == "ok", "reason": reason, "events": events}) + "\n")
            out.flush()
            self.get_logger().info(f"[{i + 1}/{self.n_episodes}] {reason}  ({n_ok}/{i + 1})")
        out.close()


def main() -> None:
    rclpy.init()
    node = EvalRunner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.setup()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
