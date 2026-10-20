#!/usr/bin/env python3
"""reports/ に表と図を出力する。docs/track1/28_sweep_and_pareto.md 5、docs/common/41"""

from __future__ import annotations

from pathlib import Path

from evla.analysis.load import load_runs

REPORTS = Path("reports")


def benchmark_table() -> None:
    df = load_runs(pattern="*_t1_modeA_*")
    if df.empty:
        print("モードAの run がない")
        return
    cols = ["policy.runtime", "policy.precision", "policy.device", "policy.n_threads",
            "result.summary.p50", "result.summary.p90", "result.peak_rss_mb"]
    t = (df[[c for c in cols if c in df.columns]]
         .groupby(["policy.runtime", "policy.precision", "policy.device", "policy.n_threads"], dropna=False)
         .median()  # 【穴①】同じ設定の周ごとの値をまとめる（外れ値に強い代表値）
         .reset_index()
         .sort_values("result.summary.p50"))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark_table.md").write_text(t.to_markdown(index=False, floatfmt=".1f"))
    print(t.to_string(index=False))


def main() -> None:
    benchmark_table()
    # TODO(you): モードBの成功率と結合してパレート図（evla.analysis.plots.pareto_plot）、
    #            遅延許容曲線（latency_tolerance_plot）、閾値の感度（38章）を出す


if __name__ == "__main__":
    main()
