#!/usr/bin/env python3
"""閉ループの動画を横に並べ、設定の名前と成功・失敗を書き込んだ1本の動画にする。

例: python scripts/make_video_grid.py --out results/videos/task0_sim.mp4 \\
        "ov/GPU=results/videos/modeB_ov_gpu/task0_ep0_ok.mp4" "torch=results/videos/modeB_torch/task0_ep0_ok.mp4"
--wallclock を付けると、各フレームの実時間（*_t.npy。推論の待ちを含む。最初の推論のウォームアップは除く）に合わせて並べる。
ロックステップでは推論中にシミュレーションが止まるので、実時間の動画では推論の遅い設定ほど止まって見える。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEADER = 40  # 名前と時刻を書く帯の高さ [px]


def _panel(frame: np.ndarray, label: str, sub: str, badge: str | None) -> np.ndarray:
    h, w, _ = frame.shape
    img = Image.new("RGB", (w, h + HEADER), "white")
    img.paste(Image.fromarray(frame), (0, HEADER))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    d.text((6, 3), label, fill="black", font=font)
    d.text((6, 21), sub, fill="dimgray", font=font)
    if badge:  # エピソードが終わったあとは、最後のフレームに結果を重ねる
        color = "seagreen" if badge == "SUCCESS" else "firebrick"
        d.rectangle((0, HEADER + h - 26, w, HEADER + h), fill=color)
        d.text((6, HEADER + h - 23), badge, fill="white", font=font)
    return np.asarray(img)


def _warm(times: np.ndarray) -> np.ndarray:
    """最初の推論の待ちを、2回目以降の推論の待ちの中央値に縮める。
    vla.cpp の OV 版は GPU 向けのコンパイルを最初の推論で行う（約 90 秒）。physicalai の OV は読み込みで済ませるので、
    そのままだと比べ方がそろわない。"""
    gaps = np.diff(times)
    infer = gaps[gaps > 5 * np.median(gaps)]  # 推論を挟んだステップの間隔
    if len(infer) > 1:
        gaps[0] = min(gaps[0], float(np.median(infer[1:])))
    return np.concatenate([[0.0], np.cumsum(gaps)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="名前=動画のパス（ファイル名の _ok / _fail で成功・失敗を判断する）")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--wallclock", action="store_true")
    a = p.parse_args()

    clips = []
    for spec in a.inputs:
        label, path = spec.split("=", 1)
        frames = iio.imread(path)
        # 実時間は録画時に保存した *_t.npy、シミュレーション時間は 1ステップ = 1/fps 秒
        times = _warm(np.load(path.replace(".mp4", "_t.npy"))) if a.wallclock else np.arange(len(frames)) / a.fps
        badge = "SUCCESS" if "_ok" in Path(path).stem else "FAIL"
        clips.append((label, frames, times, badge))

    end = max(c[2][-1] for c in clips) + 1.5  # 終わったあとに結果を 1.5 秒見せる
    out = []
    for k in range(int(end * a.fps)):
        t = k / a.fps
        row = []
        for label, frames, times, badge in clips:
            i = int(np.searchsorted(times, t, side="right")) - 1  # t の時点で最後に描かれたフレーム
            i = min(max(i, 0), len(frames) - 1)
            done = t > times[-1]
            sub = f"{'t' if a.wallclock else 'sim t'}={min(t, times[-1]):5.1f}s  step {i}"
            row.append(_panel(frames[i], label, sub, badge if done else None))
        out.append(np.concatenate(row, axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(a.out, np.stack(out), fps=a.fps, codec="libx264", macro_block_size=8)
    print(f"{a.out}: {len(out)} フレーム、{len(out) / a.fps:.1f} 秒")


if __name__ == "__main__":
    main()
