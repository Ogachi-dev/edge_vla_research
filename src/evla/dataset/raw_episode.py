"""生エピソードの書き込みと読み込み（ROS 側＝NumPy 1.26 でも動く）。docs/track2/34_episode_recorder.md 4.1

data/raw/<name>/ep_000123/
├── top/000000.png ...  wrist/000000.png ...
├── arrays.npz          # t_sim (T,), state_rad (T,6), action_rad (T,6), cube_pose_w (T,7)
├── events.jsonl
└── meta.json           # これが最後に書かれる。meta.json がある＝完全なエピソード
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ARRAY_KEYS = ("t_sim", "state_rad", "action_rad", "cube_pose_w")


class _ImageWriter:
    """PNG を別スレッドで書く（cv2.imwrite の待ちで、シミュレーションの step が遅れないように）。"""

    def __init__(self) -> None:
        self.q: queue.Queue[tuple[Path, NDArray[np.uint8]] | None] = queue.Queue(maxsize=256)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        import cv2

        while True:
            item = self.q.get()  # 【穴①】キューから1つ取り出す（空なら来るまで待つ）
            if item is None:  # 終了の合図
                break
            path, rgb = item
            try:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 【穴②】OpenCV は BGR の順で保存するので、RGB から並べ替える指定
                if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f"書き込み失敗: {path}")
            except BaseException as e:  # noqa: BLE001  エラーはメインスレッドで finish() のときに投げ直す
                self.error = e

    def put(self, path: Path, rgb: NDArray[np.uint8]) -> None:
        self.q.put((path, rgb))

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()  # 【穴③】スレッドが全部書き終わって終了するまで待つ
        if self.error is not None:
            raise self.error


class RawEpisodeWriter:
    def __init__(self, ep_dir: Path, cameras: tuple[str, ...] = ("top", "wrist")) -> None:
        if (ep_dir / "meta.json").exists():
            raise FileExistsError(f"既に完全なエピソードがある: {ep_dir}")
        self.dir = ep_dir
        self.cameras = cameras
        for c in cameras:
            (ep_dir / c).mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list[NDArray]] = {k: [] for k in ARRAY_KEYS}
        self.events: list[dict[str, Any]] = []
        self._img = _ImageWriter()
        self.n = 0

    def add_frame(self, t_sim: float, images: Mapping[str, NDArray[np.uint8]], state_rad: NDArray,
                  action_rad: NDArray, cube_pose_w: NDArray) -> None:
        for c in self.cameras:
            self._img.put(self.dir / c / f"{self.n:06d}.png", images[c])  # 【穴④】6桁ゼロ埋めのファイル名（000123.png）
        self.rows["t_sim"].append(np.float64(t_sim))
        self.rows["state_rad"].append(np.asarray(state_rad, np.float64))
        self.rows["action_rad"].append(np.asarray(action_rad, np.float64))
        self.rows["cube_pose_w"].append(np.asarray(cube_pose_w, np.float64))
        self.n += 1

    def add_event(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))

    def finish(self, meta: Mapping[str, Any]) -> None:
        """画像を書き終えるのを待ってから、配列・イベント・meta.json の順に書く。"""
        self._img.close()
        np.savez_compressed(self.dir / "arrays.npz", **{k: np.stack(v) for k, v in self.rows.items()})
        with open(self.dir / "events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")  # 【穴⑤】1行に1つのJSON（JSON Lines）
        (self.dir / "meta.json").write_text(json.dumps({**meta, "n_frames": self.n}, indent=2, ensure_ascii=False))


@dataclass(frozen=True)
class RawEpisode:
    dir: Path
    arrays: dict[str, NDArray]
    meta: dict[str, Any]
    events: list[dict[str, Any]]

    @property
    def n_frames(self) -> int:
        return len(self.arrays["t_sim"])

    def image_path(self, camera: str, i: int) -> Path:
        return self.dir / camera / f"{i:06d}.png"

    def load_image(self, camera: str, i: int) -> NDArray[np.uint8]:
        import cv2

        bgr = cv2.imread(str(self.image_path(camera, i)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.image_path(camera, i))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_raw_episode(ep_dir: Path) -> RawEpisode:
    with np.load(ep_dir / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((ep_dir / "meta.json").read_text())
    events = [json.loads(line) for line in (ep_dir / "events.jsonl").read_text().splitlines() if line]
    return RawEpisode(ep_dir, arrays, meta, events)


def list_complete_episodes(raw_root: Path) -> list[Path]:
    """meta.json があるエピソードだけを、名前順に返す（途中で落ちたものは除く）。"""
    return sorted(p.parent for p in raw_root.glob("ep_*/meta.json"))  # 【穴⑥】ワイルドカードでファイルを探す Path のメソッド
