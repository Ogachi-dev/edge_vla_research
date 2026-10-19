"""ロックステップの時計：world を止めたまま、1フレームずつ進める。docs/track2/34_episode_recorder.md 4.2

使い方（ノードの executor は別スレッドで spin しておく。サービスの応答をコールバックで受けるため）：
    clock = LockstepClock(node, "evla", physics_dt=1/600, frame_dt=1/30)
    clock.pause()
    t = clock.step_frame()
    msgs = clock.wait_for(t, ["/camera/top/image", "/joint_states"])
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.srv import ControlWorld
from rosgraph_msgs.msg import Clock

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.common.stamped_buffer import StampedBuffer, wait_until

from .ros_utils import stamp_to_ns

TOL_NS = 500_000  # 0.5 ms


class LockstepClock:
    """world の時間を進める唯一の主体。"""

    def __init__(self, node: Node, world: str, physics_dt: float, frame_dt: float) -> None:
        ratio = frame_dt / physics_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"frame_dt / physics_dt = {ratio} が整数でない（30章 4.4）")
        self.node = node
        self.steps_per_frame = ___  # 【穴①】1フレームあたりの物理ステップ数（整数に丸める）
        self.frame_dt_ns = int(round(frame_dt * 1e9))
        self._cb_srv = MutuallyExclusiveCallbackGroup()
        self._cb_sub = ReentrantCallbackGroup()  # 購読はサービスの応答待ちと同時に動けるようにする
        self._cli = node.create_client(ControlWorld, f"/world/{world}/control", callback_group=self._cb_srv)
        self._clock_ns = 0
        node.create_subscription(Clock, "/clock", self._on_clock, 10, callback_group=self._cb_sub)
        self.buffers: dict[str, StampedBuffer[Any]] = {}
        if not self._cli.wait_for_service(timeout_sec=30.0):
            raise RuntimeError(f"/world/{world}/control が無い（bridge の引数を確認）")

    def _on_clock(self, msg: Clock) -> None:
        self._clock_ns = stamp_to_ns(msg.clock)

    @property
    def now_ns(self) -> int:
        return self._clock_ns

    def track(self, topic: str, msg_type: type, stamp_of=lambda m: m.header.stamp) -> None:
        """topic を購読して、stamp をキーにバッファへ溜める。"""
        buf: StampedBuffer[Any] = StampedBuffer()
        self.buffers[topic] = buf
        self.node.create_subscription(
            msg_type, topic, lambda m: ___,  # 【穴②】受け取ったメッセージを、その stamp（ns）をキーにしてバッファに入れる
            qos_profile_sensor_data, callback_group=self._cb_sub,
        )

    def _call(self, **fields: Any) -> None:
        req = ControlWorld.Request()
        for k, v in fields.items():
            setattr(req.world_control, k, v)
        fut = ___  # 【穴③】応答を待たずに要求を送り、Future を返すメソッド
        if not wait_until(fut.done, timeout_s=10.0):
            raise TimeoutError(f"ControlWorld {fields} の応答がない")
        if not fut.result().success:
            raise RuntimeError(f"ControlWorld {fields} が失敗")

    def pause(self) -> None:
        self._call(pause=True)

    def step_frame(self) -> float:
        """multi_step = frame_dt / physics_dt を要求し、新しい sim 時刻[s]を返す。"""
        target = self._clock_ns + self.frame_dt_ns
        self._call(pause=True, multi_step=___)  # 【穴④】止めたまま、指定した物理ステップ数だけ進める指定
        if not wait_until(lambda: self._clock_ns >= target - TOL_NS, timeout_s=10.0):
            raise TimeoutError(f"/clock が {target} ns に達しない（今 {self._clock_ns}）")
        return self._clock_ns / 1e9

    def wait_for(self, t_sim: float, topics: Sequence[str], timeout_s: float = 5.0,
                 latest: Sequence[str] = ()) -> dict[str, Any]:
        """topics は stamp == t_sim（±0.5ms）のメッセージが揃うまで待つ。
        latest に入れたトピックは「t_sim 以前の最新」でよい（controller_state など、周期がずれるもの）。
        タイムアウトしたら、どのトピックが欠けたかを含めて例外にする。"""
        t_ns = int(round(t_sim * 1e9))
        out: dict[str, Any] = {}
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for tp in topics:
                if tp not in out:
                    m = self.buffers[tp].latest_at_or_before(t_ns + TOL_NS) if tp in latest \
                        else self.buffers[tp].get(t_ns, TOL_NS)
                    if m is not None:
                        out[tp] = m
            if len(out) == len(topics):
                return out
            time.sleep(0.001)
        missing = [tp for tp in topics if tp not in out]
        raise TimeoutError(f"t={t_sim:.4f} のメッセージが揃わない: {missing}")
