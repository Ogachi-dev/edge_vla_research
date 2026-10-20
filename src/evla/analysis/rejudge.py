"""ログからの把持閾値の再判定（感度分析）。docs/track2/38_gazebo_eval.md 4.3

オンラインより「厳しい」閾値だけが判定できる。厳しい閾値で掴めなかった場合に方策がやり直して成功した
可能性は考えないので、再判定の成功率は下限（悲観的な推定）になる。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from evla.analysis.stats import wilson_ci
from evla.grasp.judge import GraspGeometry, GraspThresholds, check_grasp


@dataclass(frozen=True)
class ThresholdScale:
    position: float = 1.0      # 上下・前後の許容値に掛ける
    yaw: float = 1.0
    approach: float = 1.0


def scaled(base: GraspThresholds, s: ThresholdScale) -> GraspThresholds:
    return dataclasses.replace(  # 【穴①】frozen な dataclass の一部のフィールドだけ変えたコピーを作る関数
        base,
        vertical_tol_m=base.vertical_tol_m * s.position,
        forward_tol_m=base.forward_tol_m * s.position,
        yaw_tol_deg=base.yaw_tol_deg * s.yaw,
        approach_tol_deg=None if base.approach_tol_deg is None else base.approach_tol_deg * s.approach,
    )


def rejudge_episode(events: Sequence[dict], base: GraspThresholds, scale: ThresholdScale) -> bool | None:
    """オンラインで成功したエピソードについて、実際に attach した瞬間の幾何量が、
    より厳しい閾値（scale ≤ 1）でも条件を満たすかを判定する。scale > 1 は判定できないので None。"""
    if any(v > 1.0 for v in dataclasses.astuple(scale)):  # 【穴②】緩める方向の倍率があるか
        return None
    attached = [e for e in events if e["type"] == "attached"]
    succeeded = any(e["type"] == "success" for e in events)
    if not attached or not succeeded:
        return False
    g = GraspGeometry(**attached[-1]["geometry"])  # 最後に掴んだときの幾何量（成功につながった把持）
    return check_grasp(g, scaled(base, scale)).ok  # 【穴③】厳しくした閾値で、同じ判定関数を通した結果


def sensitivity_curve(episodes: pd.DataFrame, base: GraspThresholds, scales: Sequence[float]) -> pd.DataFrame:
    """episodes: 列 config, events（エピソードごとのイベントのリスト）。
    位置の許容だけを scale 倍した成功率（Wilson 区間つき）を、設定×scale ごとに返す。"""
    rows = []
    for config, grp in episodes.groupby("config"):  # 【穴④】設定ごとに行をまとめる
        for s in scales:
            ok = [rejudge_episode(ev, base, ThresholdScale(position=s)) for ev in grp["events"]]
            k, n = sum(bool(x) for x in ok), len(ok)
            lo, hi = wilson_ci(k, n)
            rows.append({"config": config, "scale": s, "k": k, "n": n, "success_rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
