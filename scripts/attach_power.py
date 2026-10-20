#!/usr/bin/env python3
"""results/runs/* の summary.json に、HWiNFO ログから集計した電力・温度・クロックを追記する。docs/common/40"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from evla.analysis.power import load_hwinfo_csv, summarize_window


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hwinfo", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    a = p.parse_args()
    df = load_hwinfo_csv(a.hwinfo)
    for run in a.runs:
        s = pd.read_parquet(run / "samples.parquet")
        s = s[~s["is_warmup"]]
        t0 = int(s["t_start_ns"].min())
        t1 = int(s["t_start_ns"].max() + s["latency_ms"].iloc[-1] * 1e6)  # 【穴①】最後の推論の長さ（ms → ns）
        ps = summarize_window(df, t0, t1, latency_ms=float(s["latency_ms"].median()))
        summ_path = run / "summary.json"
        summ = json.loads(summ_path.read_text())
        summ["power"] = dataclasses.asdict(ps)
        summ_path.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
        print(f"{run.name}: {ps.mean_package_w:.1f}W, {ps.energy_per_infer_j}J/infer, throttled={ps.throttled}")


if __name__ == "__main__":
    main()
