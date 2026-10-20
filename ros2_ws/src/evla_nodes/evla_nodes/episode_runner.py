"""データ生成：ロックステップで区間を実行し、フレームを記録する（時計の所有者）。docs/track2/34_episode_recorder.md

起動例:
    ros2 run evla_nodes episode_runner --ros-args -p use_sim_time:=true \
        -p datagen_config:=configs/datagen/default.yaml -p ik_config:=configs/datagen/so101_geometry.yaml \
        -p out_dir:=data/raw/so101_pickplace_v1 -p planner:=interp
"""

from __future__ import annotations

import json
import math
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from evla.dataset.raw_episode import RawEpisodeWriter
from evla.grasp.events import classify_failure
from evla.planning.build import PlanConfig, UnreachableError, build_plan, joint_interp
from evla.planning.episode_plan import ArmSegment, EpisodePlan, GripperSegment
from evla.planning.ik_so101 import So101Geometry
from evla.planning.randomize import sample_params, sample_scene

from .sim_io import SimIO

FPS = 30


def frames_for(duration_s: float) -> int:
    return math.ceil(duration_s * FPS)  # 【穴①】その時間をカバーするのに必要なフレーム数（切り上げ）


class EpisodeRunner(Node):
    def __init__(self) -> None:
        super().__init__("episode_runner")
        p = self.declare_parameter
        self.ranges = yaml.safe_load(Path(p("datagen_config", "configs/datagen/default.yaml").value).read_text())
        geo_raw = yaml.safe_load(Path(p("ik_config", "configs/datagen/so101_geometry.yaml").value).read_text())
        self.geo = So101Geometry(**{k: tuple(v) if isinstance(v, list) else v for k, v in geo_raw.items()})
        self.out_dir = Path(p("out_dir", "data/raw/so101_pickplace_v1").value)
        self.planner_kind = p("planner", "interp").value         # "interp" | "moveit"
        self.cube_size = p("cube_size_m", 0.03).value
        self.plan_cfg = PlanConfig(grasp_z_m=self.cube_size / 2)
        self.io: SimIO | None = None

    def setup(self) -> None:
        """executor が回り始めてから呼ぶ（サービスの待ち合わせがあるため）。"""
        self.io = SimIO(self)
        if self.planner_kind == "moveit":
            from .demo_generator import MoveItJointPlanner

            self.plan_joint_move = MoveItJointPlanner()
        else:
            self.plan_joint_move = joint_interp

    def make_plan(self, seed: int) -> EpisodePlan:
        rng = np.random.default_rng(seed)  # 【穴②】エピソードごとのシードから乱数生成器を作る
        scene = sample_scene(rng, self.ranges)
        params = sample_params(rng, self.ranges)
        return build_plan(seed, scene, params, self.geo, self.plan_cfg, self.plan_joint_move)

    def run_episode(self, seed: int) -> tuple[bool, str]:
        io = self.io
        assert io is not None
        plan = self.make_plan(seed)
        s = plan.scene
        io.reset_scene(s.cube_xy_m, s.cube_yaw_rad, self.cube_size / 2, s.box_xy_m, s.q_init_rad,
                       self.plan_cfg.gripper_open_rad)
        ep_dir = self.out_dir / f"ep_{seed:06d}"
        writer = RawEpisodeWriter(ep_dir)
        events: list[dict] = []
        t0 = time.perf_counter()
        for seg in plan.segments:
            if isinstance(seg, ArmSegment):
                io.send_arm(seg.trajectory)
                n = frames_for(seg.trajectory.duration_s)
            elif isinstance(seg, GripperSegment):
                io.send_gripper(seg.target_rad, seg.duration_s)
                n = frames_for(seg.duration_s)
            else:
                n = frames_for(seg.duration_s)
            for _ in range(n):
                f = io.step_and_observe()
                writer.add_frame(f.t_sim, f.images, f.state_rad, f.action_rad, f.cube_pose_w)  # 【穴③】1フレームを生エピソードに書き足すメソッド
                events += io.take_events()
        # 成功判定の静止を待つ（記録は続ける）
        for _ in range(FPS * 3):
            if any(e["type"] in ("success", "failure") for e in events):
                break
            f = io.step_and_observe()
            writer.add_frame(f.t_sim, f.images, f.state_rad, f.action_rad, f.cube_pose_w)
            events += io.take_events()
        success = any(e["type"] == "success" for e in events)  # 【穴④】イベントの中に success が1つでもあれば成功
        reason = classify_failure(events)
        for e in events:
            writer.add_event(e)
        writer.finish({"seed": seed, "success": success, "failure_reason": reason,
                       "plan": plan.to_json_dict(), "wall_s": time.perf_counter() - t0})
        return success, reason

    def run(self, n_episodes: int, base_seed: int) -> None:
        stats: dict[str, int] = {}
        for i in range(n_episodes):
            seed = base_seed + i
            if (self.out_dir / f"ep_{seed:06d}" / "meta.json").exists():
                continue  # 再開：完了済みは飛ばす
            try:
                ok, reason = self.run_episode(seed)
            except UnreachableError as e:
                ok, reason = False, "unreachable"
                self.get_logger().warn(str(e))
            except Exception:  # noqa: BLE001  一晩回すので、1エピソードの失敗で止めない
                ok, reason = False, "exception"
                self.get_logger().error(traceback.format_exc())
            stats[reason] = stats.get(reason, 0) + 1
            self.get_logger().info(f"[{i + 1}/{n_episodes}] seed={seed} success={ok} ({reason}) stats={stats}")
        (self.out_dir / "generation_stats.json").write_text(json.dumps(stats, indent=2))


def main() -> None:
    rclpy.init()
    node = EpisodeRunner()
    executor = MultiThreadedExecutor(num_threads=4)  # 【穴⑤】コールバックを複数スレッドで処理する executor（サービス待ちの間も購読を受け取れる）
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.setup()
        node.run(int(node.ranges.get("n_episodes", 300)), int(node.ranges.get("base_seed", 1000)))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
