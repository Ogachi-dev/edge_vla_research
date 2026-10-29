"""LeRobot の LiberoEnv を包んで Observation を返す。docs/track1/21_libero_setup.md

LeRobot の LIBERO まわりは版で変わりやすい。生の観測の取り出し方は _raw_to_observation に閉じ込め、
最初に `print(raw_obs.keys())` などで実物を見てから【要確認】の行を直すこと。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from evla.sim.types import Observation, StepResult

# LeRobot の TASK_SUITE_MAX_STEPS と同じ値。【要確認】lerobot/envs/libero.py
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
ACTION_TOL = 0.1   # 行動の範囲 [-1, 1] からのはみ出しを許す量
CAMERAS = {"image": "agentview_image", "image2": "robot0_eye_in_hand_image"}


@dataclass(frozen=True)
class LiberoConfig:
    suite: str = "libero_object"
    task_id: int = 0
    image_size: int = 256          # LIBEROの描画解像度（学習データと同じ256を既定にする）
    init_state_index: int = 0      # 固定初期状態のどれを使うか（＝エピソード番号）
    control_mode: str = "relative"


def quat_xyzw_to_axis_angle(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(4,) (x,y,z,w) → (3,) 軸角（単位ベクトル×角度）。角度0付近の特異点を安全に扱う。"""
    q = np.asarray(q, dtype=np.float64)
    w = np.clip(q[3], -1.0, 1.0)  # 【穴①】(x,y,z,w) のうち w 成分
    s = np.sqrt(max(0.0, 1.0 - w * w))  # 【穴②】sin(θ/2)^2 = 1 - cos(θ/2)^2
    if s < 1e-8:  # 【穴③】回転がほぼ0（軸が定まらない）ときの条件
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)  # 【穴④】w = cos(θ/2) から θ を求める
    return q[:3] / s * angle


class LiberoTaskEnv:
    def __init__(self, cfg: LiberoConfig) -> None:
        from lerobot.envs.libero import LiberoEnv  # 【要確認】クラスの場所と引数名
        from libero.libero import benchmark  # .venv-bench にだけ入っている

        self.cfg = cfg
        suite = benchmark.get_benchmark_dict()[cfg.suite]()  # 【穴⑤】スイート名 → スイートのクラス の辞書を返す関数
        self._env = LiberoEnv(
            task_suite=suite,
            task_id=cfg.task_id,
            task_suite_name=cfg.suite,
            obs_type="pixels_agent_pos",
            observation_width=cfg.image_size,
            observation_height=cfg.image_size,
            control_mode=cfg.control_mode,
            episode_index=cfg.init_state_index,
        )
        self._task: str = suite.get_task(cfg.task_id).language
        self._t = 0

    @property
    def task(self) -> str:
        return self._task

    @property
    def max_steps(self) -> int:
        return MAX_STEPS[self.cfg.suite]

    def _raw_to_observation(self, raw: dict[str, Any]) -> Observation:
        """生の観測 → Observation。画像は180°回転して連続メモリにする。"""
        images = {}
        for key in CAMERAS:
            img = raw["pixels"][key]  # 【要確認】キーの階層（"pixels" の下に "image"/"image2" があるか）
            images[key] = np.ascontiguousarray(img[::-1, ::-1])  # 【穴⑥】上下と左右を両方反転（＝180°回転）するスライス
        rs = raw["robot_state"]  # 【要確認】
        state = np.concatenate([  # 【穴⑦】配列をつなげる関数
            rs["eef"]["pos"],
            quat_xyzw_to_axis_angle(rs["eef"]["quat"]),
            rs["gripper"]["qpos"],
        ]).astype(np.float32)
        return Observation(images=images, state=state, task=self._task, sim_step=self._t)

    def reset(self, init_state_index: int | None = None, seed: int | None = None) -> Observation:
        """環境をリセットし、待機ステップ後の最初の観測を返す。
        【要確認】LeRobot の LiberoEnv が reset の中で待機ステップ（num_steps_wait）を回すか。"""
        if init_state_index is not None:
            self._env.episode_index = init_state_index  # 【要確認】属性名
        raw, _info = self._env.reset(seed=seed)
        self._t = 0
        return self._raw_to_observation(raw)

    def step(self, action: npt.NDArray[np.float32]) -> StepResult:
        """action: (7,) float32、範囲[-1, 1]。範囲外なら ValueError（クリップで隠さない）。"""
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (7,):
            raise ValueError(f"行動の形が違う: {a.shape}")
        # SmolVLA の出力は ±1 を少しはみ出す（例 -1.024）。腕の6次元は 0.1 までは丸め、それより大きく外れたら
        # 座標系や逆正規化の誤りとみなして止める。グリッパ（最後の次元）は符号で開閉が決まるので、範囲を確かめずに丸める
        # （遅延が大きい非同期ではチャンクの後ろの行動を使い、グリッパが -1.10 まで外れた。2026-10-30）
        if np.any(np.abs(a[:-1]) > 1.0 + ACTION_TOL):  # 【穴⑧】腕の6次元のうち、範囲 [-1, 1] を（誤差を少し許して）超えた要素
            raise ValueError(f"行動が範囲外: {a}")
        a = np.clip(a, -1.0, 1.0)
        raw, _reward, terminated, truncated, info = self._env.step(a)  # 【穴⑨】行動を1つ実行する gymnasium のメソッド（観測, 報酬, terminated, truncated, info の5つを返す）
        self._t += 1
        success = bool(info.get("is_success", False))
        done = success or terminated or truncated or self._t >= self.max_steps
        return StepResult(obs=self._raw_to_observation(raw), success=success, done=done, info=dict(info))

    def render_frame(self) -> npt.NDArray[np.uint8]:
        """動画保存用の画像（人間が見やすい向き）。"""
        return np.asarray(self._env.render())

    def close(self) -> None:
        self._env.close()
