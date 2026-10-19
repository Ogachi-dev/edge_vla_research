"""生エピソード群 → LeRobotDataset v3（.venv-bench で実行）。docs/track2/34_episode_recorder.md 4.2

LeRobotDataset の API は 0.6 系を想定。【要確認】の行は入れた版のソースで確かめる。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
from evla.dataset.conventions import LEROBOT_NAMES, URDF_TO_LEROBOT, rad_vector_to_lerobot
from evla.dataset.raw_episode import list_complete_episodes, load_raw_episode

FPS = 30
CAMERAS = ("top", "wrist")


def build_features(image_hw: tuple[int, int] = (480, 640)) -> dict[str, dict]:
    vec = {"dtype": "float32", "shape": (6,), "names": list(LEROBOT_NAMES)}
    feats: dict[str, dict] = {"observation.state": vec, "action": dict(vec)}
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {
            "dtype": ___,  # 【穴①】画像を動画（mp4）として保存する dtype の名前
            "shape": (*image_hw, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def convert(raw_root: Path, out_root: Path, repo_id: str, task_text: str,
            only_success: bool = True, image_writer_threads: int = 4) -> dict[str, Any]:
    """生エピソードを順に読み、add_frame → save_episode。最後に finalize()。
    追加メタデータを out_root/meta/evla/ に書く。変換の要約を返す。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 【要確認】import の場所

    ds = ___(  # 【穴②】空のデータセットを新しく作るクラスメソッド
        repo_id=repo_id, fps=FPS, root=out_root, robot_type="so101_follower",
        features=build_features(), use_videos=True, image_writer_threads=image_writer_threads,
    )
    extra_rows, n_skipped = [], 0
    for ep_dir in list_complete_episodes(raw_root):
        ep = load_raw_episode(ep_dir)
        if only_success and not ep.meta.get("success", False):
            n_skipped += 1
            continue
        state = rad_vector_to_lerobot(ep.arrays["state_rad"])  # (T, 6) deg・グリッパ0〜100
        action = rad_vector_to_lerobot(ep.arrays["action_rad"])
        for i in range(ep.n_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                **{f"observation.images.{c}": ep.load_image(c, i) for c in CAMERAS},
                "task": task_text,  # 【要確認】版によっては add_frame(frame, task=...) の形
            }
            ___  # 【穴③】1フレーム追加するメソッド
        ___  # 【穴④】ここまでのフレームを1エピソードとして確定するメソッド
        extra_rows.append({"source": ep_dir.name, **ep.meta})
    ds.finalize()  # 【要確認】v3 で必要な最後の書き出し（呼び忘れるとメタデータが欠ける）

    meta_dir = out_root / "meta" / "evla"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes_extra.jsonl", "w") as f:
        for i, row in enumerate(extra_rows):
            f.write(json.dumps({"episode_index": i, **row}, ensure_ascii=False) + "\n")
    (meta_dir / "robot.json").write_text(json.dumps({
        "joint_map_urdf_to_lerobot": URDF_TO_LEROBOT, "units": "deg", "gripper": "0(closed)-100(open)",
    }, indent=2))
    # TODO(you): cameras.json、thresholds.json、generation.json（34章 4.4 の表）
    for name in ("thresholds.json",):
        src = raw_root / name
        if src.exists():
            shutil.copy(src, meta_dir / name)
    return {"n_episodes": len(extra_rows), "n_skipped": n_skipped}
