import pandas as pd
import pytest

from evla.analysis.power import load_hwinfo_csv, summarize_window


def make_csv(path):
    rows = ["Date,Time,CPU Package Power [W],CPU Package [°C],Core Effective Clocks [MHz]"]
    for s in range(60):
        busy = 30 <= s < 50
        rows.append(f"15.10.2026,12:00:{s:02d}.000,{25.0 if busy else 5.0},{70 if busy else 45},{3000}")
    path.write_text("\n".join(rows), encoding="utf-8-sig")


def test_summarize(tmp_path):
    p = tmp_path / "h.csv"
    make_csv(p)
    df = load_hwinfo_csv(p)
    t0 = pd.Timestamp("2026-10-15 12:00:30", tz="Asia/Tokyo").value
    t1 = pd.Timestamp("2026-10-15 12:00:49", tz="Asia/Tokyo").value
    s = summarize_window(df, t0, t1, idle_window_s=20, latency_ms=1000)
    assert s.mean_package_w == pytest.approx(25.0)
    assert s.idle_package_w == pytest.approx(5.0, abs=1.1)     # 区間の端の行が入るかどうかで少し変わる
    assert s.energy_per_infer_j == pytest.approx(20.0, abs=1.1)
    assert not s.throttled
