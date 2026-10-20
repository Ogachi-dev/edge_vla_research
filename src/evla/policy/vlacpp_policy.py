"""vla-server（vla.cpp）の ZeroMQ クライアント。docs/track1/25_runtime_vlacpp.md

サーバーの起動は呼び出し側の責任（VlaServerProcess）。
事前に scripts/gen_vla_proto.sh で vla_pb2.py を生成しておく。
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import zmq

from evla.policy.base import ChunkResult, Policy, PolicyConfig
from evla.policy.preprocess import hwc_uint8_to_chw_float, resize_with_pad, select_views
from evla.sim.types import Observation

MAX_STATE_DIM = 32
TOKENIZER_MAX_LEN = 48   # SmolVLA の tokenizer_max_length。【要確認】モデルの config.json


class VlaCppPolicy(Policy):
    def __init__(self, cfg: PolicyConfig) -> None:
        """cfg.server_addr に接続する。
        cfg.extra["tokenizer"]: 既定 "HuggingFaceTB/SmolVLM2-500M-Instruct"
        cfg.extra["recv_timeout_ms"]: 既定 600_000（初回のコンパイルを考慮）"""
        from transformers import AutoTokenizer

        from evla.policy._proto import vla_pb2

        self.cfg = cfg
        self.pb = vla_pb2
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.extra.get("tokenizer", "HuggingFaceTB/SmolVLM2-500M-Instruct")
        )
        self._ids = itertools.count()
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)  # 【穴①】「送って、返事を待つ」側のソケットの種類
        self._sock.setsockopt(zmq.LINGER, 0)  # 【穴②】close 時に未送信のメッセージを待たない設定
        self._sock.setsockopt(zmq.RCVTIMEO, int(cfg.extra.get("recv_timeout_ms", 600_000)))  # 【穴③】受信のタイムアウト[ms]
        self._sock.setsockopt(zmq.REQ_RELAXED, 1)
        self._sock.setsockopt(zmq.REQ_CORRELATE, 1)
        self._sock.connect(cfg.server_addr)
        self._chunk_size: int | None = None
        self._action_dim: int | None = None

    @property
    def chunk_size(self) -> int:
        if self._chunk_size is None:
            raise RuntimeError("1回推論してから参照する")
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("1回推論してから参照する")
        return self._action_dim

    def _encode(self, obs: Observation, noise: npt.NDArray[np.float32] | None, request_id: int) -> bytes:
        """Observation → PredictRequest のバイト列。docs/track1/25 の 2.1 の表のとおり。"""
        req = self.pb.PredictRequest(request_id=request_id)
        for img in select_views(obs, self.cfg.views).values():
            chw = resize_with_pad(hwc_uint8_to_chw_float(img), self.cfg.image_size)
            hwc = np.ascontiguousarray(chw.transpose(1, 2, 0), dtype=np.float32)  # 【穴④】CHW → HWC
            req.images.add(
                encoding=self.pb.Image.F32_RGB_01,
                height=hwc.shape[0], width=hwc.shape[1],
                data=hwc.tobytes(),  # 【穴⑤】配列の生のバイト列
            )
        tokens = self.tokenizer(
            obs.task + "\n", padding=False, truncation=True, max_length=TOKENIZER_MAX_LEN  # 【穴⑥】末尾に改行を付けた指示文
        )["input_ids"]
        req.lang_tokens.extend(int(t) for t in tokens)
        state = np.zeros(MAX_STATE_DIM, dtype=np.float32)
        state[: obs.state.shape[0]] = obs.state  # 【穴⑦】先頭の実次元ぶんだけコピー（残りは0）
        req.state.extend(state.tolist())
        if noise is not None:
            req.noise.extend(noise.astype(np.float32).ravel().tolist())  # 【穴⑧】(50, 32) → (1600,) に平坦化
        return req.SerializeToString()

    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        """送受信。応答の error が空でなければ RuntimeError。
        breakdown_ms に、サーバー側の latency_ms_* と、transport = 往復 − latency_ms_total を入れる。"""
        t0 = time.perf_counter_ns()
        payload = self._encode(obs, noise, next(self._ids))
        t1 = time.perf_counter_ns()
        self._sock.send(payload)
        raw = self._sock.recv()  # タイムアウトなら zmq.Again
        t2 = time.perf_counter_ns()
        rep = self.pb.PredictResponse()
        rep.ParseFromString(raw)  # 【穴⑨】バイト列からメッセージを復元する protobuf のメソッド
        if rep.error:
            raise RuntimeError(f"vla-server error: {rep.error}")
        full = np.asarray(rep.action_chunk, dtype=np.float32).reshape(rep.chunk_size, rep.action_dim)  # 【穴⑩】応答に入っている (行, 列)
        dim = int(self.cfg.extra.get("action_dim", full.shape[1]))
        actions = full[:, :dim]  # 先頭の実次元（LIBEROなら7）
        self._chunk_size, self._action_dim = actions.shape
        t3 = time.perf_counter_ns()
        rtt = (t2 - t1) / 1e6
        return ChunkResult(
            actions=actions,
            latency_ms=(t3 - t0) / 1e6,
            breakdown_ms={
                "preprocess": (t1 - t0) / 1e6,
                "total": rep.latency_ms_total, "inference": rep.latency_ms_inference,
                "vision": rep.latency_ms_vision, "prefill": rep.latency_ms_prefill,
                "denoise": rep.latency_ms_denoise,
                "transport": rtt - rep.latency_ms_total,  # 【穴⑪】往復時間からサーバー内の時間を引いた残り
            },
        )

    def close(self) -> None:
        self._sock.close()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "server_addr": self.cfg.server_addr, "zmq": zmq.zmq_version()}
