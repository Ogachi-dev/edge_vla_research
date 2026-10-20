#!/usr/bin/env python3
"""scene.yaml と乱数シードから world SDF を作る（見た目のランダム化）。docs/track2/31 Step 2

例: python3 ros2_ws/src/evla_sim/scripts/render_world.py --seed 0 --out /tmp/evla_world_0.sdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jinja2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent.parent


def cube_inertia(mass: float, size: float) -> float:
    return mass * size**2 / 6  # 【穴①】一様な立方体の慣性モーメント m·s²/6


def box_walls(inner_xy: list[float], wall_h: float, t: float) -> list[dict]:
    ix, iy = inner_xy
    ox, oy = ix + 2 * t, iy + 2 * t
    return [
        {"name": "bottom", "pose": f"0 0 {t / 2} 0 0 0", "size": f"{ox} {oy} {t}"},
        {"name": "wall_xp", "pose": f"{ix / 2 + t / 2} 0 {wall_h / 2} 0 0 0", "size": f"{t} {oy} {wall_h}"},
        {"name": "wall_xn", "pose": f"{-(ix / 2 + t / 2)} 0 {wall_h / 2} 0 0 0", "size": f"{t} {oy} {wall_h}"},
        {"name": "wall_yp", "pose": f"0 {iy / 2 + t / 2} {wall_h / 2} 0 0 0", "size": f"{ox} {t} {wall_h}"},  # 【穴②】内寸の半分＋壁厚の半分（壁の中心の位置）
        {"name": "wall_yn", "pose": f"0 {-(iy / 2 + t / 2)} {wall_h / 2} 0 0 0", "size": f"{ox} {t} {wall_h}"},
    ]


def render(scene: dict, seed: int) -> str:
    rng = np.random.default_rng(seed)
    ap = scene["appearance_ranges"]
    top = dict(scene["cameras"]["top"])
    pose = np.array(top["pose_xyzrpy"], dtype=float)
    pose[:3] += rng.uniform(-ap["camera_jitter_m"], ap["camera_jitter_m"], 3)
    pose[3:] += rng.uniform(-ap["camera_jitter_rad"], ap["camera_jitter_rad"], 3)
    top["pose"] = " ".join(f"{v:.5f}" for v in pose)
    color = ap["cube_colors"][int(rng.integers(len(ap["cube_colors"])))]
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(HERE / "worlds"), undefined=jinja2.StrictUndefined)  # 【穴③】テンプレートに未定義の変数があればエラーにする設定（タイポの早期発見）
    return env.get_template("evla_pickplace.sdf.jinja").render(
        **scene, top=top,
        light=round(float(rng.uniform(*ap["light_intensity"])), 3),
        table_color=" ".join(f"{v:.2f}" for v in rng.uniform(0.5, 0.9, 3)),
        cube_color=" ".join(str(c) for c in color),
        cube_inertia=cube_inertia(scene["cube"]["mass_kg"], scene["cube"]["size_m"]),
        box_walls=box_walls(scene["box"]["inner_xy_m"], scene["box"]["wall_h_m"], scene["box"]["wall_t_m"]),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, default=HERE / "config" / "scene.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    scene = yaml.safe_load(a.scene.read_text())
    a.out.write_text(render(scene, a.seed))
    print(f"world → {a.out}")


if __name__ == "__main__":
    main()
