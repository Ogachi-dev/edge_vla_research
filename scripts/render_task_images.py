#!/usr/bin/env python3
"""報告書用に、各タスクの初期状態の画像と指示文を reports/figures/tasks/ に保存する。

例: MUJOCO_GL=glfw python scripts/render_task_images.py --suite libero_object
出力: task<番号>.png（初期状態 0）、tasks_grid.png（全タスクを並べたもの）、tasks.json（番号と指示文）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evla.sim.libero_env import LiberoConfig, LiberoTaskEnv

COLS = 5
CAPTION = 24  # タスク番号を書く帯の高さ [px]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--n_tasks", type=int, default=10)  # LIBERO の Spatial・Object・Goal・10 は各 10 タスク
    p.add_argument("--out", type=Path, default=Path("reports/figures/tasks"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tasks, frames = [], []
    for t in range(a.n_tasks):
        env = LiberoTaskEnv(LiberoConfig(suite=a.suite, task_id=t))
        try:
            obs = env.reset(init_state_index=0, seed=0)
            frame = env.render_frame()
        finally:
            env.close()
        Image.fromarray(frame).save(a.out / f"task{t}.png")
        tasks.append({"task_id": t, "instruction": obs.task})
        frames.append(frame)
        print(f"task {t}: {obs.task}")

    # 全タスクを 2 段 × 5 列に並べ、上に番号を書く
    h, w, _ = frames[0].shape
    rows = -(-len(frames) // COLS)
    grid = Image.new("RGB", (w * COLS, (h + CAPTION) * rows), "white")
    d = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=16)
    for i, f in enumerate(frames):
        x, y = (i % COLS) * w, (i // COLS) * (h + CAPTION)
        d.text((x + 6, y + 3), f"task {i}", fill="black", font=font)
        grid.paste(Image.fromarray(np.asarray(f)), (x, y + CAPTION))
    grid.save(a.out / "tasks_grid.png")
    (a.out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
