"""報告書の図。docs/track1/28_sweep_and_pareto.md 5、docs/track2/38_gazebo_eval.md 4.4"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面なしで PNG/SVG を書く
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evla.analysis.pareto import pareto_front  # noqa: E402


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def pareto_plot(df: pd.DataFrame, out: Path, x: str = "p50_ms", y: str = "success_rate") -> None:
    """横軸 p50、縦軸 成功率（Wilson 区間のエラーバー）。マーカーの形＝ランタイム、色＝精度。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    shapes = "os^Dv<>pPXh*"  # ランタイムがこれより多いときは、形を繰り返して使う
    runtimes = sorted(df["runtime"].unique())
    markers = {rt: shapes[i % len(shapes)] for i, rt in enumerate(runtimes)}
    for (rt, prec), g in df.groupby(["runtime", "precision"]):
        yerr = np.vstack([g[y] - g["ci_lo"], g["ci_hi"] - g[y]])
        ax.errorbar(g[x], g[y], yerr=yerr, fmt=markers[rt], label=f"{rt}/{prec}", capsize=3)  # 【穴①】縦方向のエラーバー（下側と上側の長さ）を渡す引数
    front = pareto_front(df, minimize=x, maximize=y)
    ax.plot(front[x], front[y], "k--", lw=1, label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("latency p50 [ms]")
    ax.set_ylabel("success rate")
    ax.legend(fontsize=7)
    save(fig, out)


def latency_tolerance_plot(curve: pd.DataFrame, out: Path) -> None:
    """横軸 遅延、縦軸 成功率、線＝(mode, n_exec)、帯＝信頼区間。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (mode, n_exec), g in curve.groupby(["mode", "n_exec"]):
        g = g.sort_values("latency_ms")
        ax.plot(g["latency_ms"], g["success_rate"], marker="o", label=f"{mode}, n_exec={n_exec}")
        ax.fill_between(g["latency_ms"], g["ci_lo"], g["ci_hi"], alpha=0.2)  # 【穴②】2本の線の間を塗る matplotlib の関数
    ax.set_xlabel("inference latency [ms]")
    ax.set_ylabel("success rate")
    ax.legend(fontsize=7)
    save(fig, out)


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(x))
    return xs, np.arange(1, len(xs) + 1) / len(xs)  # 【穴③】1/n, 2/n, ..., 1 の階段


def grasp_ecdf_plot(df: pd.DataFrame, out: Path, col: str = "abs_offset_mm") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for config, g in df.groupby("config"):
        xs, ys = ecdf(g[col].to_numpy())
        ax.step(xs, ys, where="post", label=str(config))
    ax.set_xlabel(col)
    ax.set_ylabel("ECDF")
    ax.legend(fontsize=7)
    save(fig, out)
