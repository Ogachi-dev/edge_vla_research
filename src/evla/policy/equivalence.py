"""ランタイム間の数値一致の指標。docs/track1/22_policy_interface.md 4"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class EquivalenceReport:
    max_abs: float               # 全要素の最大絶対誤差
    rmse: float
    cosine: float                # チャンク全体をベクトルとみなしたコサイン類似度
    per_dim_max_abs: list[float] # 次元ごと（グリッパの次元だけ大きい、などを見つける）
    gripper_sign_flips: int      # グリッパの開閉の符号が食い違ったステップ数


def compare_chunks(ref: npt.NDArray, test: npt.NDArray, gripper_dim: int = -1) -> EquivalenceReport:
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(f"形が違う: {ref.shape} vs {test.shape}")
    diff = test - ref
    a, b = ref.ravel(), test.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return EquivalenceReport(
        max_abs=float(np.abs(diff).max()),  # 【穴①】全要素の絶対誤差の最大
        rmse=float(np.sqrt(np.mean(diff**2))),  # 【穴②】二乗平均平方根誤差
        cosine=float(a @ b / denom) if denom > 0 else float("nan"),  # 【穴③】内積をノルムの積で割る
        per_dim_max_abs=np.abs(diff).max(axis=0).tolist(),  # 【穴④】時間方向（行）に沿って最大をとる
        gripper_sign_flips=int(np.sum(np.sign(ref[:, gripper_dim]) != np.sign(test[:, gripper_dim]))),  # 【穴⑤】正負を ±1 にする関数
    )


def make_fixed_noise(chunk_size: int, max_action_dim: int, seed: int) -> npt.NDArray[np.float32]:
    """標準正規分布のノイズ。全ランタイムで同じものを使う。"""
    rng = np.random.default_rng(seed)  # 【穴⑥】シード付きの乱数生成器（新しいAPI）
    return rng.standard_normal((chunk_size, max_action_dim)).astype(np.float32)
