"""SmolVLA の PyTorch CPU ベースライン。docs/track1/23_runtime_pytorch.md

スレッド数は、torch を import する「前」に OMP_NUM_THREADS を設定しておくこと（scripts 側の責任）。
LeRobot の API 名は 0.6 系を想定している。【要確認】の行は、入れた版のソースで確かめる。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from evla.policy.base import ChunkResult, Policy, PolicyConfig
from evla.policy.preprocess import hwc_uint8_to_chw_float, select_views
from evla.sim.types import Observation


class TorchSmolVLA(Policy):
    def __init__(self, cfg: PolicyConfig) -> None:
        """cfg.model: HFのrepo_idまたはローカルパス。
        cfg.precision: "fp32" | "bf16"
        cfg.n_threads: torch.set_num_threads に渡す。
        cfg.extra["compile"]: bool、torch.compile を使うか。"""
        from lerobot.policies.factory import (
            make_pre_post_processors,  # 【要確認】版によって場所が違う
        )
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.cfg = cfg
        if cfg.n_threads is not None:
            torch.set_num_threads(cfg.n_threads)  # 【穴①】intra-op のスレッド数を設定する torch の関数（モデルを読む前に呼ぶ）

        self.policy = SmolVLAPolicy.from_pretrained(cfg.model)  # 【穴②】HFのrepo_idかパスからモデルを読むクラスメソッド
        self.policy.config.num_steps = cfg.num_steps  # 【要確認】flow matching のステップ数の属性名
        self.policy.eval()  # 【穴③】dropout などを推論モードにする nn.Module のメソッド
        self.policy.to("cpu")
        # 重みは混合精度で保存されていて（VLM は bf16、state_proj・action_*_proj などは fp32）、
        # from_pretrained はその型のまま読む。bf16 はこのまま使う（全部を bf16 にすると、
        # fp32 の入力と射影の型がぶつかる）。この CPU は bf16 の演算を持たず極端に遅いので、fp32 は明示的に変換する
        if cfg.precision != "bf16":
            self.policy.to(torch.float32)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=cfg.model,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        if cfg.extra.get("compile", False):
            self.policy = torch.compile(self.policy)

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.chunk_size)

    @property
    def action_dim(self) -> int:
        return int(self.policy.config.output_features["action"].shape[0])

    def _to_batch(self, obs: Observation) -> dict[str, Any]:
        """Observation → LeRobotの前処理が期待する辞書。
        画像は (1, 3, H, W) float32 [0,1]。cfg.views にない視点は入れない。"""
        batch: dict[str, Any] = {}
        for view, img in select_views(obs, self.cfg.views).items():
            chw = hwc_uint8_to_chw_float(img)
            batch[f"observation.images.{view}"] = torch.from_numpy(chw).unsqueeze(0)  # 【穴④】先頭にバッチ次元を足す
        batch["observation.state"] = torch.from_numpy(obs.state.astype(np.float32))[None]
        batch["task"] = obs.task  # 【要確認】文字列のままか、[obs.task] のリストか
        return batch

    @torch.inference_mode()  # 【穴⑤】勾配を記録しない推論用のデコレータ（no_grad より軽い）
    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        t0 = time.perf_counter_ns()
        batch = self.pre(self._to_batch(obs))
        t1 = time.perf_counter_ns()
        noise_t = None if noise is None else torch.from_numpy(noise)[None]  # (1, chunk, max_action_dim)
        actions = self.policy.predict_action_chunk(batch, noise=noise_t)  # 【穴⑥】チャンクを丸ごと返すメソッド（select_action は内部キューを使うので使わない）
        t2 = time.perf_counter_ns()
        actions = self.post(actions)  # 逆正規化
        out = actions[0].float().cpu().numpy().astype(np.float32)  # 【穴⑦】バッチ次元を外す
        t3 = time.perf_counter_ns()
        ms = lambda a, b: (b - a) / 1e6  # noqa: E731
        return ChunkResult(
            actions=out,
            latency_ms=ms(t0, t3),
            breakdown_ms={"preprocess": ms(t0, t1), "inference": ms(t1, t2), "postprocess": ms(t2, t3)},
        )

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "dtype": str(next(self.policy.parameters()).dtype),
        }
