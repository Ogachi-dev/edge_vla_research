#!/usr/bin/env python3
"""reports/ に表と図を出力する。docs/track1/28_sweep_and_pareto.md 5、docs/common/41"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from evla.analysis.load import flatten, load_runs
from evla.analysis.plots import latency_tolerance_plot, pareto_plot
from evla.analysis.stats import wilson_ci

REPORTS = Path("reports")
RUNS = Path("results/runs")


def run_label(row: pd.Series) -> str:
    """表と図に出す設定の名前。tag があればそれ、無ければランタイムごとに効く設定から作る。
    掃引が付ける tag（<掃引の名前>_r<周>）は設定を区別しないので使わない。
    非同期（executor.mode=async）には /async と、次の推論を始めるキューの残り（async_threshold）を付ける。
    例：torch/fp32、ov/GPU/fp16、ov/CPU/fp32/w8a16、ov/GPU/fp16/async/th20、vlacpp/sycl"""
    tag = row.get("tag")
    if isinstance(tag, str) and tag and not re.fullmatch(r".+_r\d+", tag):
        return tag
    runtime = str(row.get("policy.runtime"))
    if runtime == "vlacpp":  # デバイスと精度は vla-server 側で決まる
        backend = row.get("server.env.VLA_BACKEND")
        parts = [f"vlacpp/{backend if isinstance(backend, str) else 'cpu'}"]
    else:
        keys = ("policy.device", "policy.precision") if runtime == "ov" else ("policy.precision",)
        parts = [runtime] + [str(row.get(k)) for k in keys if isinstance(row.get(k), str)]
        model = row.get("policy.model")
        # 重みの量子化（公開の w8a16、scripts/quantize_ov.py の int8・int4）
        parts += [q for q in ("w8a16", "int8", "int4") if isinstance(model, str) and q in model]
    if row.get("executor.mode") == "async":
        th = row.get("executor.async_threshold")
        parts.append("async" if pd.isna(th) else f"async/th{int(th)}")
    return "/".join(parts)


def benchmark_table() -> pd.DataFrame:
    df = load_runs(pattern="*_t1_modeA_*")
    if df.empty:
        print("モードAの run がない")
        return df
    df["label"] = df.apply(run_label, axis=1)
    cols = ["label", "policy.n_threads", "result.summary.p50", "result.summary.p90",
            "result.load_s", "result.peak_rss_mb", "result.peak_rss_server_mb"]
    t = (df[[c for c in cols if c in df.columns]]
         .groupby(["label", "policy.n_threads"], dropna=False)
         .median()  # 【穴①】同じ設定の周ごとの値をまとめる（外れ値に強い代表値）
         .reset_index()
         .sort_values("result.summary.p50"))
    t.insert(2, "n_runs", df.groupby(["label", "policy.n_threads"], dropna=False).size().reindex(
        pd.MultiIndex.from_frame(t[["label", "policy.n_threads"]])).to_numpy())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark_table.md").write_text(t.to_markdown(index=False, floatfmt=".1f"))
    print(t.to_string(index=False))
    return t


def load_closed_loop(pattern: str) -> pd.DataFrame:
    """閉ループの run ごとに、成功率・Wilson 区間・閉ループ中の推論レイテンシの中央値を1行にまとめる。"""
    rows = []
    for d in sorted(RUNS.glob(pattern)):
        ep_p, cfg_p = d / "episodes.parquet", d / "config.yaml"
        if not (ep_p.exists() and cfg_p.exists()):
            continue
        cfg = flatten(yaml.safe_load(cfg_p.read_text()))
        ep = pd.read_parquet(ep_p)
        k, n = int(ep["success"].sum()), len(ep)
        lo, hi = wilson_ci(k, n)
        rows.append({
            "run_id": d.name, **cfg, "n_episodes": n, "n_success": k,
            "success_rate": k / n, "ci_lo": lo, "ci_hi": hi,
            "p50_ms": float(ep["lat_p50_ms"].median()), "wall_min": float(ep["wall_s"].sum() / 60),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["label"] = df.apply(run_label, axis=1)
    return df


def success_table_and_pareto() -> None:
    # 同期（モードB）に、実測の推論時間を遅延にした非同期（モードC、latency.type=measured）を並べる
    df = pd.concat([load_closed_loop("*_t1_modeB_*"), _mode_c("measured")], ignore_index=True)
    if df.empty:
        print("モードBの run がない")
        return
    cols = ["label", "executor.n_exec", "n_episodes", "n_success", "success_rate", "ci_lo", "ci_hi",
            "p50_ms", "wall_min", "run_id"]
    t = df[[c for c in cols if c in df.columns]].sort_values("p50_ms")
    (REPORTS / "success_table.md").write_text(t.to_markdown(index=False, floatfmt=".2f"))
    print(t.to_string(index=False))
    # パレート図：横軸＝閉ループ中の推論 p50、縦軸＝成功率。凡例はランタイム（label）× n_exec
    plot_df = df.assign(runtime=df["label"], precision="n_exec=" + df["executor.n_exec"].astype(str))
    pareto_plot(plot_df, REPORTS / "figures" / "pareto")


def _mode_c(latency_type: str) -> pd.DataFrame:
    df = load_closed_loop("*_t1_modeC-*")
    if df.empty or "latency.type" not in df.columns:
        return df
    return df[df["latency.type"] == latency_type]


def latency_tolerance() -> None:
    df = _mode_c("constant")  # 遅延許容曲線は仮想の一定の遅延の run だけ
    if df.empty or "latency.ms" not in df.columns:
        print("モードCの run（latency.type=constant）がない")
        return
    curve = df.rename(columns={"executor.mode": "mode", "executor.n_exec": "n_exec", "latency.ms": "latency_ms"})
    cols = ["label", "mode", "n_exec", "latency_ms", "n_episodes", "success_rate", "ci_lo", "ci_hi", "run_id"]
    (REPORTS / "latency_tolerance.md").write_text(
        curve[cols].sort_values(["mode", "n_exec", "latency_ms"]).to_markdown(index=False, floatfmt=".2f"))
    latency_tolerance_plot(curve, REPORTS / "figures" / "latency_tolerance")


def main() -> None:
    benchmark_table()
    success_table_and_pareto()
    latency_tolerance()


if __name__ == "__main__":
    main()
