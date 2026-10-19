"""成功率の統計（信頼区間、対応のある検定、必要な試行数）。docs/track1/27_closed_loop_b_c.md 4.2"""

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy import stats

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """成功 k 回 / 試行 n 回の、成功率の Wilson 信頼区間。
    正規近似（p ± z√(p(1-p)/n)）と違い、p が 0 や 1 に近くても区間が [0,1] からはみ出さない。"""
    if n == 0:
        return (0.0, 1.0)
    z = ___  # 【穴①】標準正規分布の上側 α/2 点（95%なら約1.96）。累積分布の逆関数
    p = k / n
    denom = 1 + z**2 / n
    center = ___  # 【穴②】区間の中心 (p + z²/2n) / (1 + z²/n)
    half = z * math.sqrt(___) / denom  # 【穴③】半幅の平方根の中身 p(1-p)/n + z²/4n²
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(a_success: Sequence[bool], b_success: Sequence[bool]) -> dict[str, float]:
    """同じ (task, init_state) の組で比較した、対応のある検定。b01, b10, p値（正確二項検定）。
    b01 = A失敗・B成功、b10 = A成功・B失敗。両方成功や両方失敗の組は差の情報を持たないので使わない。"""
    if len(a_success) != len(b_success):
        raise ValueError("対応のある比較なので長さを揃える")
    b01 = sum(1 for a, b in zip(a_success, b_success, strict=True) if ___)  # 【穴④】Aが失敗してBが成功した組
    b10 = sum(1 for a, b in zip(a_success, b_success, strict=True) if a and not b)
    n = b01 + b10
    p = 1.0 if n == 0 else ___.pvalue  # 【穴⑤】「食い違いが半々で起きる」を帰無仮説にした二項検定
    return {"b01": float(b01), "b10": float(b10), "p_value": float(p)}


def required_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """2群の比率の差を検出するのに必要な1群あたりの試行数（正規近似）。"""
    if p1 == p2:
        raise ValueError("差が0では検出できない")
    za = stats.norm.ppf(1 - alpha / 2)
    zb = ___  # 【穴⑥】検出力に対応する標準正規分布の点
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num**2 / ___)  # 【穴⑦】差の2乗（差が小さいほど n が増える）
