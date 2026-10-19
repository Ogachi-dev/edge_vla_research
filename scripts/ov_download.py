#!/usr/bin/env python3
"""Intel 公開の OpenVINO 版 SmolVLA を data/models/ov/ に取得する。docs/track1/24_runtime_openvino.md"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい

REPOS = ("OpenVINO/smolvla-libero-fp16-ov", "OpenVINO/smolvla-libero-bf16-ov", "OpenVINO/smolvla-libero-w8a16-ov")


def main() -> None:
    for repo in REPOS:
        out = Path("data/models/ov") / repo.split("/")[1]
        path = ___(repo_id=repo, local_dir=out)  # 【穴①】HFのリポジトリを丸ごとダウンロードする huggingface_hub の関数
        files = sorted(p.name for p in Path(path).iterdir())
        print(f"{repo} → {out}: {files}")


if __name__ == "__main__":
    main()
