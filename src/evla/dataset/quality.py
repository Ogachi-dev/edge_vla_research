"""生エピソードの品質チェック。docs/track2/34_episode_recorder.md 6

D5 の判定ゲートで使う。どれか1つでも失敗したエピソードは学習に使わない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from evla.dataset.raw_episode import RawEpisode, list_complete_episodes, load_raw_episode

FRAME_DT = 1.0 / 30


@dataclass
class EpisodeQuality:
    name: str
    n_frames: int
    success: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_episode(ep: RawEpisode, black_thresh: float = 5.0, n_image_samples: int = 5) -> EpisodeQuality:
    q = EpisodeQuality(ep.dir.name, ep.n_frames, bool(ep.meta.get("success", False)))
    t = ep.arrays["t_sim"]
    dt = np.diff(t)  # 【穴①】隣り合う要素の差（フレーム間隔）
    if dt.size and not np.allclose(dt, FRAME_DT, atol=1e-6):  # 【穴②】すべての間隔が 1/30 秒に（誤差 1e-6 以内で）等しいか
        q.problems.append(f"frame_gap: dt の最大 {dt.max():.4f}s")
    for key in ("state_rad", "action_rad"):
        a = ep.arrays[key]
        if not np.all(np.isfinite(a)):  # 【穴③】NaN でも inf でもない要素が True になる関数
            q.problems.append(f"nonfinite: {key}")
        if len(a) != ep.n_frames:
            q.problems.append(f"length: {key} {len(a)} != {ep.n_frames}")
    for cam in ("top", "wrist"):
        n_png = len(list((ep.dir / cam).glob("*.png")))
        if n_png != ep.n_frames:
            q.problems.append(f"missing_images: {cam} {n_png}/{ep.n_frames}")
            continue
        for i in np.linspace(0, ep.n_frames - 1, n_image_samples).astype(int):
            if ep.load_image(cam, int(i)).mean() < black_thresh:  # 【穴④】画素の平均が閾値より暗い＝黒画像
                q.problems.append(f"black_image: {cam}[{i}]")
                break
    return q


def check_dataset(raw_root: Path) -> pd.DataFrame:
    rows = []
    for ep_dir in list_complete_episodes(raw_root):
        q = check_episode(load_raw_episode(ep_dir))
        rows.append({"name": q.name, "n_frames": q.n_frames, "success": q.success, "ok": q.ok,
                     "problems": "; ".join(q.problems)})
    return pd.DataFrame(rows)


def gate_summary(df: pd.DataFrame, last_n: int = 50) -> dict[str, float]:
    """D5 判定ゲートの数値（docs/00_schedule.md）：成功数、直近 last_n 本の成功率、品質の合格率。"""
    return {
        "n_success": int(df["success"].sum()),
        "recent_success_rate": float(df["success"].tail(last_n).mean()),  # 【穴⑤】最後の last_n 行の成功率（True/False の平均）
        "quality_ok_rate": float(df.loc[df["success"], "ok"].mean()) if df["success"].any() else float("nan"),
    }
