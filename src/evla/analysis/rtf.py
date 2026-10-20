"""RTF（実時間比）。docs/common/40_metrics_measurement.md 4.3"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def rtf_series(t_sim_s: npt.ArrayLike, t_wall_ns: npt.ArrayLike, window_s: float = 5.0) -> npt.NDArray:
    """各時点について、直前 window_s 秒（壁時計）の sim 時刻の増分 ÷ 壁時計の増分。"""
    ts = np.asarray(t_sim_s, dtype=np.float64)
    tw = np.asarray(t_wall_ns, dtype=np.float64) / 1e9
    j = np.searchsorted(tw, tw - window_s)  # 【穴①】ソート済みの配列で「window_s 秒前」の位置を二分探索する関数
    dw = tw - tw[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dw > 0, (ts - ts[j]) / dw, np.nan)  # 【穴②】sim 時刻の増分 ÷ 壁時計の増分
