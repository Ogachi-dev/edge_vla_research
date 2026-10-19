#!/usr/bin/env python3
"""閉ループ評価（モードB / C）。docs/track1/27_closed_loop_b_c.md

例: python scripts/eval_closed_loop.py --config configs/closed_loop/modeC_async.yaml --set latency.ms=800
"""

from __future__ import annotations

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.analysis.stats import wilson_ci
from evla.common.cli import base_parser
from evla.common.config import load_config, to_plain
from evla.common.run_id import make_run_dir, new_run_id
from evla.sim.closed_loop import ClosedLoopConfig, run_eval


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = load_config(args.config, ClosedLoopConfig, args.set)
    mode = "modeB" if cfg.executor.mode == "lockstep" else f"modeC-{cfg.executor.mode}"
    run_dir = make_run_dir(new_run_id("t1", mode, to_plain(cfg)))
    df = run_eval(cfg, run_dir)
    k, n = int(df["success"].sum()), len(df)
    lo, hi = ___  # 【穴①】成功率の95%信頼区間
    print(f"{run_dir.name}: success {k}/{n} = {k / n:.2f}  95%CI [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
