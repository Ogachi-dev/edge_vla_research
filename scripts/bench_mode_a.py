#!/usr/bin/env python3
"""モードA（推論単体）を1設定だけ測る。docs/track1/26_bench_mode_a.md

例: python scripts/bench_mode_a.py --config configs/bench/torch_cpu.yaml --set policy.n_threads=8
"""

from __future__ import annotations

import os

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.common.cli import base_parser
from evla.common.config import load_config, parse_override, set_by_path


def set_thread_env(n_threads: int | None) -> None:
    """torch / OpenMP を import する「前」に呼ぶ（import 後に変えても効かないライブラリがある）。"""
    if n_threads is None:
        return
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VLA_N_THREADS"):
        ___[k] = str(n_threads)  # 【穴①】環境変数の辞書（os モジュール。値は文字列）


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    # 先に n_threads だけ読んで環境変数を設定してから、重いモジュールを import する
    import yaml

    raw = yaml.safe_load(args.config.read_text())
    for item in args.set:
        set_by_path(raw, *parse_override(item))
    set_thread_env(raw.get("policy", {}).get("n_threads"))

    from evla.bench.mode_a import ModeAConfig, run_mode_a
    from evla.common.config import to_plain
    from evla.common.run_id import make_run_dir, new_run_id

    cfg = load_config(args.config, ModeAConfig, args.set)
    run_id = new_run_id("t1", "modeA", to_plain(cfg))
    run_dir = make_run_dir(run_id)
    res = run_mode_a(cfg, run_dir)
    s = res.summary
    print(f"{run_id}: p50={s['p50']:.1f}ms p90={s['p90']:.1f}ms load={res.load_s:.1f}s "
          f"rss={res.peak_rss_mb:.0f}MB")


if __name__ == "__main__":
    main()
