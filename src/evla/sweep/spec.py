"""掃引の定義（YAML）を設定のリストに展開する。docs/track1/28_sweep_and_pareto.md 4.1"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.common.run_id import config_hash


@dataclass(frozen=True)
class SweepItem:
    overrides: dict[str, Any]      # "a.b": value
    key: str                       # 設定のハッシュ（再開と結合に使う）


@dataclass(frozen=True)
class SweepSpec:
    name: str
    mode: str
    base: Path
    items: list[SweepItem]
    rounds: int = 1
    cooldown_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def expand_grid(block: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """{"a": [1, 2], "b": [3]} → [{"a": 1, "b": 3}, {"a": 2, "b": 3}]（直積）"""
    keys = list(block)
    return [dict(zip(keys, combo, strict=True)) for combo in ___]  # 【穴①】各キーの値のリストの直積をとる関数


def matches(overrides: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    """rule のキーと値がすべて overrides と一致するか（exclude の判定）。"""
    return all(k in overrides and overrides[k] == v for k, v in rule.items())


def expand(spec_path: Path) -> SweepSpec:
    """grid（リストの各要素ごとに直積をとり、全体を連結）を展開し、exclude に合うものを除く。"""
    raw = yaml.safe_load(spec_path.read_text())
    combos: list[dict[str, Any]] = []
    for block in raw["grid"]:
        combos.extend(___)  # 【穴②】ブロックごとの直積
    excludes = raw.get("exclude", [])
    kept = [c for c in combos if ___]  # 【穴③】どの除外ルールにも当てはまらない
    items = [SweepItem(overrides=c, key=config_hash(c)) for c in kept]
    return SweepSpec(name=raw["name"], mode=raw["mode"], base=Path(raw["base"]), items=items,
                     rounds=raw.get("rounds", 1), cooldown_s=raw.get("cooldown_s", 0.0))


def policy_key(cfg: Mapping[str, Any]) -> str:
    """policy.* の部分だけから計算したハッシュ。モードAの結果とモードBの結果を結合するキーになる。
    cfg は平坦化した辞書（"policy.runtime": ...）。"""
    return config_hash({k: v for k, v in cfg.items() if ___})  # 【穴④】"policy." で始まるキーだけ
