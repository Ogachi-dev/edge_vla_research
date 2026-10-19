"""推論サーバー：22章の Policy を ZeroMQ で包む（.venv-bench で動かす）。docs/track2/37_policy_node_zmq.md 4.1"""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping
from typing import Any

import numpy as np
import zmq

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.ipc.protocol import decode, encode
from evla.policy.base import Policy
from evla.sim.types import Observation

MAX_ACTION_DIM = 32


class InferenceServer:
    def __init__(self, policy: Policy, bind: str = "tcp://127.0.0.1:5600",
                 image_key_map: Mapping[str, str] | None = None) -> None:
        """image_key_map：ROS側のカメラ名 → Policy が期待する Observation.images のキー。
        SO-101のモデルなら {"top": "top", "wrist": "wrist"}、LIBEROのモデルで疎通確認するなら既定値。"""
        self.policy = policy
        self.image_key_map = dict(image_key_map or {"top": "image", "wrist": "image2"})
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(___)  # 【穴①】「受けて、返事を送る」側のソケットの種類
        ___  # 【穴②】サーバー側はアドレスを「待ち受ける」（クライアント側は connect）
        self._running = True

    def _to_observation(self, head: dict[str, Any], arrays: dict[str, np.ndarray]) -> Observation:
        images = {}
        for k, v in arrays.items():
            if k.startswith("image."):
                cam = k.removeprefix("image.")
                images[___] = v  # 【穴③】対応表にあれば変換、無ければそのままの名前
        return Observation(images=images, state=arrays["state"], task=head["task"])

    def handle(self, head: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
        kind = head.get("type")
        if kind == "ping":
            return {"type": "pong"}, {}
        if kind == "describe":
            return {"type": "describe", "describe": self.policy.describe()}, {}
        if kind == "predict":
            obs = self._to_observation(head, arrays)
            noise = None
            if head.get("noise_seed") is not None:
                rng = np.random.default_rng(head["noise_seed"])
                noise = rng.standard_normal((self.policy.chunk_size, MAX_ACTION_DIM)).astype(np.float32)
            r = self.policy.predict_chunk(obs, noise)
            return ({"type": "result", "latency_ms": r.latency_ms, "breakdown_ms": dict(r.breakdown_ms)},
                    {"actions": r.actions.astype(np.float32)})
        raise ValueError(f"未知の type: {kind}")

    def serve_forever(self) -> None:
        while self._running:
            frames = self._sock.recv_multipart(copy=False)
            t0 = time.perf_counter_ns()
            try:
                head, arrays = decode(frames)
                rep_head, rep_arrays = self.handle(head, arrays)
                rep_head["id"] = head.get("id")
                rep_head["error"] = None
            except Exception as e:  # noqa: BLE001  エラーでもREPは必ず返す（返さないとクライアントのREQが固まる）
                traceback.print_exc()
                rep_head, rep_arrays = {"type": "result", "id": None, "error": repr(e)}, {}
            rep_head["server_ms"] = (time.perf_counter_ns() - t0) / 1e6
            self._sock.send_multipart(___, copy=False)  # 【穴④】返事のヘッダと配列をフレームのリストにする

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._sock.close(linger=0)
