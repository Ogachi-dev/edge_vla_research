"""パレート境界。docs/track1/28_sweep_and_pareto.md 4.1"""

from __future__ import annotations

import pandas as pd

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい


def pareto_front(df: pd.DataFrame, minimize: str, maximize: str) -> pd.DataFrame:
    """minimize 列は小さいほど良く、maximize 列は大きいほど良いとして、非劣な行だけを返す（minimize の昇順）。

    方法：minimize の昇順（同じ値なら maximize の降順）に並べて上から見ていき、
    「これまでの最大の maximize」を真に超えた行だけを残す。
    """
    d = df.sort_values([minimize, maximize], ascending=___)  # 【穴①】1列目は昇順、2列目は降順
    keep = []
    best = float("-inf")
    for idx, v in d[maximize].items():
        if ___:  # 【穴②】これまでより真に良いときだけ残す条件
            keep.append(idx)
            best = v
    return d.loc[keep]
