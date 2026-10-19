#!/usr/bin/env python3
"""生エピソード → LeRobotDataset v3（.venv-bench で実行）。docs/track2/34_episode_recorder.md"""

from __future__ import annotations

import argparse
from pathlib import Path

from evla.dataset.to_lerobot import convert


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw/so101_pickplace_v1"))
    p.add_argument("--out", type=Path, default=Path("data/lerobot/so101_pickplace_v1"))
    p.add_argument("--repo_id", default="local/so101_pickplace_v1")
    p.add_argument("--task", default="Pick up the cube and put it in the box.")
    p.add_argument("--include_failures", action="store_true")
    a = p.parse_args()
    print(convert(a.raw, a.out, a.repo_id, a.task, only_success=not a.include_failures))


if __name__ == "__main__":
    main()
