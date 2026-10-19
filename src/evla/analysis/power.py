"""Windows 側の HWiNFO ログ（CSV）と run の時間区間の突き合わせ。docs/common/40_metrics_measurement.md 4.2"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい


@dataclass(frozen=True)
class PowerSummary:
    mean_package_w: float
    p95_package_w: float
    mean_temp_c: float
    max_temp_c: float
    mean_clock_mhz: float
    throttled: bool
    idle_package_w: float
    energy_per_infer_j: float | None


def find_col(df: pd.DataFrame, *keywords: str) -> str:
    """列名にすべてのキーワードを含む最初の列（HWiNFO の列名は版やCPUで揺れる）。"""
    for c in df.columns:
        if ___:  # 【穴①】大文字小文字を無視して、全キーワードが列名に含まれる
            return c
    raise KeyError(f"列が見つからない: {keywords}")


def load_hwinfo_csv(path: Path, tz: str = "Asia/Tokyo") -> pd.DataFrame:
    """HWiNFOのCSVを読み、UTC の時刻を index にする。【要確認】日付の書式は実物を見て合わせる。"""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    df = df[pd.to_numeric(df.iloc[:, 2], errors="coerce").notna()]  # 末尾の「列名の繰り返し」行などを捨てる
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M:%S.%f")
    df.index = ___  # 【穴②】ローカル時刻として解釈し、UTC に変換する
    return df.apply(pd.to_numeric, errors="coerce")


def _ts(t_ns: int) -> pd.Timestamp:
    return pd.Timestamp(t_ns, unit="ns", tz="UTC")


def summarize_window(df: pd.DataFrame, t0_ns: int, t1_ns: int, idle_window_s: float = 30.0,
                     latency_ms: float | None = None) -> PowerSummary:
    """[t0, t1] の区間を集計する。t0 の直前 idle_window_s 秒をアイドルの基準とする。"""
    pkg, temp, clk = find_col(df, "package", "power"), find_col(df, "package", "[°c]"), find_col(df, "clock", "effective")
    w = df.loc[___]  # 【穴③】時刻の index で区間を切り出すスライス
    idle = df.loc[_ts(t0_ns) - pd.Timedelta(seconds=idle_window_s):_ts(t0_ns)]
    if w.empty:
        raise ValueError("区間にログが無い（時計のずれ、タイムゾーンを確認）")
    p_idle = float(idle[pkg].mean())
    energy = None
    if latency_ms is not None:
        energy = ___  # 【穴④】(平均電力 − アイドル電力) × 1推論の秒数 = J
    throttled = bool((w[temp] >= 95).mean() > 0.1 or w[clk].min() < 0.7 * idle[clk].median())
    return PowerSummary(
        mean_package_w=float(w[pkg].mean()), p95_package_w=float(w[pkg].quantile(0.95)),
        mean_temp_c=float(w[temp].mean()), max_temp_c=float(w[temp].max()),
        mean_clock_mhz=float(w[clk].mean()), throttled=throttled, idle_package_w=p_idle,
        energy_per_infer_j=energy,
    )
