"""SmolVLA の OpenVINO 版（physicalai の InferenceModel を使う）。docs/track1/24_runtime_openvino.md

physicalai は v0.x 系で API が変わりやすい。physicalai に依存する行はこのファイルに閉じ込める。
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from evla.policy.base import ChunkResult, Policy, PolicyConfig
from evla.policy.preprocess import hwc_uint8_to_chw_float, select_views
from evla.sim.types import Observation

PRECISION_HINT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


class OVSmolVLA(Policy):
    def __init__(self, cfg: PolicyConfig) -> None:
        """cfg.model: OVモデルのディレクトリ（manifest.json があるもの）
        cfg.device: "CPU" | "GPU"
        cfg.precision: 実行精度のヒント。"fp32" → INFERENCE_PRECISION_HINT="f32"、"fp16" → "f16"
        cfg.n_threads: INFERENCE_NUM_THREADS（CPUのみ）
        cfg.extra: 追加のcompile config（例 {"NUM_STREAMS": 1, "CACHE_DIR": "..."}）"""
        from physicalai.inference import InferenceModel  # 【要確認】import の場所（版で変わりうる）

        self.cfg = cfg
        self._warned_noise = False
        t0 = time.perf_counter()
        self.model = InferenceModel(
            cfg.model, backend="openvino", device=cfg.device, **self._ov_config()  # 【穴①】compile config を「キーワード引数として展開して」渡す
        )
        self.load_s = time.perf_counter() - t0
        # 公開版の manifest は ActionChunkTrimmer（n_action_steps=1）で先頭の1行だけを返す。
        # 実行する行数は executor が決めるので、IR の出力（50×7）をそのまま使う
        self.model.postprocessors = [
            p for p in self.model.postprocessors if type(p).__name__ != "ActionChunkTrimmer"
        ]
        self._chunk_size: int | None = None
        self._action_dim: int | None = None

    def _ov_config(self) -> dict[str, Any]:
        """PERFORMANCE_HINT="LATENCY" を基本に、cfg から compile config を組み立てる。"""
        conf: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}  # 【穴②】バッチ1で最短レイテンシを狙うヒントの値
        if self.cfg.precision in PRECISION_HINT:
            conf["INFERENCE_PRECISION_HINT"] = PRECISION_HINT[self.cfg.precision]
        if self.cfg.n_threads is not None and self.cfg.device == "CPU":
            conf["INFERENCE_NUM_THREADS"] = self.cfg.n_threads  # 【穴③】CPUのスレッド数を指定するプロパティ名
        conf.update(self.cfg.extra)  # extra が最優先
        return conf

    @property
    def chunk_size(self) -> int:
        if self._chunk_size is None:
            raise RuntimeError("1回推論してから参照する（IRから形を読む方法が分かれば差し替える）")
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            raise RuntimeError("1回推論してから参照する")
        return self._action_dim

    def _to_ov_obs(self, obs: Observation) -> dict[str, Any]:
        """Observation → physicalai の観測辞書。キーは "images.<view>"、"state"、"task"。"""
        d: dict[str, Any] = {}
        for view, img in select_views(obs, self.cfg.views).items():
            d[f"images.{view}"] = hwc_uint8_to_chw_float(img)[None]  # 【穴④】(3,H,W) → (1,3,H,W)
        d["state"] = obs.state.astype(np.float32)[None]
        d["task"] = [obs.task]  # 【穴⑤】list[str] で渡す
        return d

    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        """noise を渡せない場合（IRに入力がない場合）、noise が None でなければ警告を1回だけ出して無視する。"""
        if noise is not None and not self._warned_noise:
            warnings.warn("OVのIRはノイズを入力に取らないので、固定ノイズを無視する", stacklevel=2)
            self._warned_noise = True
        t0 = time.perf_counter_ns()
        ov_obs = self._to_ov_obs(obs)
        t1 = time.perf_counter_ns()
        actions = np.asarray(self.model.predict_action_chunk(ov_obs), dtype=np.float32)
        t2 = time.perf_counter_ns()
        if actions.ndim == 3:  # (1, chunk, dim) で返る版もある
            actions = actions[0]
        self._chunk_size, self._action_dim = actions.shape  # 【穴⑥】(chunk, dim) をそのまま分解代入
        return ChunkResult(
            actions=actions,
            latency_ms=(t2 - t0) / 1e6,
            breakdown_ms={"preprocess": (t1 - t0) / 1e6, "inference": (t2 - t1) / 1e6},
        )

    def describe(self) -> dict[str, Any]:
        """openvinoの版、FULL_DEVICE_NAME、実際の EXECUTION_DEVICES、INFERENCE_PRECISION_HINT、
        INFERENCE_NUM_THREADS（compiled_model.get_property で読む）。"""
        import openvino as ov

        info: dict[str, Any] = {**super().describe(), "openvino": ov.__version__, "load_s": self.load_s,
                                "config": self._ov_config()}
        info["full_device_name"] = ov.Core().get_property(self.cfg.device, "FULL_DEVICE_NAME")  # 【穴⑦】デバイスの正式名を表すプロパティ名
        compiled = _find_compiled_model(self.model)
        if compiled is not None:
            for key in ("EXECUTION_DEVICES", "INFERENCE_PRECISION_HINT", "INFERENCE_NUM_THREADS"):
                try:
                    info[key] = str(compiled.get_property(key))
                except RuntimeError:
                    info[key] = None  # GPUには INFERENCE_NUM_THREADS が無い
        return info


def _find_compiled_model(model: Any) -> Any | None:
    """InferenceModel の内部から ov.CompiledModel を探す。【要確認】属性名は physicalai のソースで確かめて直す。"""
    import openvino as ov

    for name in ("compiled_model", "_compiled_model", "model", "adapter"):
        obj = getattr(model, name, None)
        if isinstance(obj, ov.CompiledModel):
            return obj
        inner = getattr(obj, "compiled_model", None)
        if isinstance(inner, ov.CompiledModel):
            return inner
    return None
