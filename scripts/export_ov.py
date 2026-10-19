#!/usr/bin/env python3
"""LeRobot のチェックポイント → OpenVINO IR（.venv-export で実行）。docs/track2/36_model_conversion.md

例: python scripts/export_ov.py --ckpt HuggingFaceVLA/smolvla_libero --out data/models/ov_check/libero_fp16
    python scripts/export_ov.py --ckpt outputs/train/so101_v1/checkpoints/last/pretrained_model \
        --out data/models/so101_v1/ov_int8 --int8
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい


def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):  # 1MB ずつ読む（大きなファイルを一度にメモリに載せない）
            ___  # 【穴①】ハッシュに少しずつデータを足すメソッド
    return h.hexdigest()


def export_openvino(ckpt: str, out_dir: Path, int8_weights: bool) -> Path:
    from physicalai.export import compress_weights_openvino_int8_sym
    from physicalai.inference import InferenceModel
    from physicalai.policies import SmolVLA

    policy = SmolVLA(pretrained_name_or_path=___)  # 【穴②】HFのrepoかローカルのディレクトリを渡す引数
    policy.eval()
    hooks = [compress_weights_openvino_int8_sym] if int8_weights else []
    policy.export(str(out_dir), backend=___, post_export_hooks=hooks)  # 【穴③】書き出し先の形式
    m = InferenceModel(str(out_dir), backend="openvino", device="CPU")  # 読み込めることの確認
    print("exported:", sorted(p.name for p in out_dir.iterdir()))
    print("model:", m)
    (out_dir / "convert_log.json").write_text(json.dumps({
        "ckpt": ckpt,
        "ckpt_sha256": sha256_of(Path(ckpt) / "model.safetensors"),
        "int8_weights": int8_weights,
        "physicalai": metadata.version("physicalai"),
    }, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--int8", action="store_true")
    a = p.parse_args()
    export_openvino(a.ckpt, a.out, a.int8)


if __name__ == "__main__":
    main()
