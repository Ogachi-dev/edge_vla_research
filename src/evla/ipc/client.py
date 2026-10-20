"""推論サーバーのクライアント（タイムアウト・再接続つき）。docs/track2/37_policy_node_zmq.md 4.1

REQ ソケットは「送ったら必ず受ける」状態機械なので、タイムアウトしたら作り直す（Lazy Pirate パターン）。
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import zmq

from evla.ipc.protocol import decode, encode


@dataclass(frozen=True)
class PredictReply:
    actions: np.ndarray            # (chunk, action_dim)
    server_latency_ms: float
    breakdown_ms: Mapping[str, float] = field(default_factory=dict)
    rtt_ms: float = float("nan")   # クライアントで測った往復


class InferenceClient:
    def __init__(self, addr: str, timeout_ms: int = 10_000, retries: int = 1) -> None:
        self.addr = addr
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._ctx = zmq.Context.instance()
        self._ids = itertools.count()
        self._sock: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.addr)
        self._sock = s
        self._poller = zmq.Poller()
        self._poller.register(s, zmq.POLLIN)  # 【穴①】「受信できるデータが来たら」を監視する指定

    def _request(self, header: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> tuple[dict, dict, float]:
        header = {**header, "id": next(self._ids)}
        for _attempt in range(self.retries + 1):
            assert self._sock is not None
            t0 = time.perf_counter_ns()
            self._sock.send_multipart(encode(header, arrays), copy=False)  # 【穴②】複数フレームを1メッセージとして送るメソッド
            if self._poller.poll(self.timeout_ms):  # 【穴③】最大 timeout_ms 待って、受信可能なソケットがあれば真
                frames = self._sock.recv_multipart(copy=False)
                rtt = (time.perf_counter_ns() - t0) / 1e6
                head, arrs = decode(frames)
                if head.get("id") != header["id"]:
                    raise RuntimeError(f"応答の id が違う: {head.get('id')} != {header['id']}")
                if head.get("error"):
                    raise RuntimeError(f"server error: {head['error']}")
                return head, arrs, rtt
            self._connect()  # 【穴④】返事が来ないのでソケットを作り直す（REQ は送受の順番が崩れると使えなくなる）
        raise TimeoutError(f"{self.addr} から {self.retries + 1} 回とも応答がない")

    def predict(self, images: Mapping[str, np.ndarray], state: np.ndarray, task: str,
                t_obs_sim: float, noise_seed: int | None = None) -> PredictReply:
        arrays = {**{f"image.{k}": v for k, v in images.items()}, "state": np.asarray(state, np.float32)}
        header = {"type": "predict", "task": task, "t_obs_sim": t_obs_sim, "noise_seed": noise_seed,
                  "t_send_wall_ns": time.time_ns()}
        head, arrs, rtt = self._request(header, arrays)
        return PredictReply(actions=arrs["actions"], server_latency_ms=head["latency_ms"],
                            breakdown_ms=head.get("breakdown_ms", {}), rtt_ms=rtt)

    def ping(self) -> bool:
        try:
            head, _, _ = self._request({"type": "ping"}, {})
        except TimeoutError:
            return False
        return head.get("type") == "pong"

    def describe(self) -> dict[str, Any]:
        head, _, _ = self._request({"type": "describe"}, {})
        return head["describe"]

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
