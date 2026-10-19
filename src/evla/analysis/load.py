"""results/runs/* を1つの DataFrame に読み込む。docs/track1/28_sweep_and_pareto.md"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"policy": {"runtime": "ov"}} → {"policy.runtime": "ov"}"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(___)  # 【穴①】入れ子の辞書を、キーに「親のキー.」を付けて再帰的に平坦化
        else:
            out[key] = v
    return out


def load_runs(root: Path = Path("results/runs"), pattern: str = "*") -> pd.DataFrame:
    """config.yaml と summary.json がそろった run だけを読む（途中で落ちた run は除く）。"""
    rows = []
    for d in sorted(root.glob(pattern)):
        cfg_p, sum_p = d / "config.yaml", d / "summary.json"
        if not (cfg_p.exists() and sum_p.exists()):
            continue
        cfg = yaml.safe_load(cfg_p.read_text())
        summary = json.loads(sum_p.read_text())
        rows.append({"run_id": d.name, **flatten(cfg), **flatten(summary, prefix="result.")})
    return pd.DataFrame(rows)
