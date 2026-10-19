#!/usr/bin/env python3
"""段階的な掃引を実行する。docs/track1/28_sweep_and_pareto.md

例: python scripts/sweep.py configs/sweep/stage1.yaml --dry_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.sweep.runner import run_sweep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("spec", type=Path)
    p.add_argument("--dry_run", action=___)  # 【穴①】書けば True、書かなければ False になるフラグの指定
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--sec_per_item", type=float, default=300.0, help="見積もり用の1実行あたりの秒数")
    a = p.parse_args()
    run_sweep(a.spec, resume=not a.no_resume, dry_run=a.dry_run, sec_per_item=a.sec_per_item)


if __name__ == "__main__":
    main()
