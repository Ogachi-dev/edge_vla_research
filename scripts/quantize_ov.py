#!/usr/bin/env python3
"""OpenVINO IR（physicalai の書き出し）の重みを NNCF で量子化し、同じ形式のディレクトリに保存する。

例: python scripts/quantize_ov.py --src data/models/ov/smolvla-libero-fp16-ov \\
        --out data/models/ov/smolvla-libero-int4-ov --mode int4_asym
重みだけを量子化する（活性は f16/f32 のまま。データを使わない）。NNCF の既定どおり、埋め込みと最後の層は INT8 で残る。
manifest.json・config.json・トークナイザはそのままコピーするので、physicalai の InferenceModel でそのまま読める。
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib import metadata
from pathlib import Path

COPY = ("manifest.json", "config.json", "tokenizer.xml", "tokenizer.bin", "README.md")


def main() -> None:
    import nncf
    import openvino as ov

    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", default="int4_asym", choices=["int4_asym", "int4_sym", "int8_asym", "int8_sym"])
    # INT4 だけで使う（INT8 は出力チャネルごと）。SmolVLA の線形層の入力チャネル数（960・480・320）を割り切る 32 にする
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--ratio", type=float, default=1.0)      # INT4 にする層の割合（残りは INT8）
    a = p.parse_args()

    model = ov.Core().read_model(a.src / "smolvla.xml")
    mode = nncf.CompressWeightsMode[a.mode.upper()]
    kw = {"group_size": a.group_size, "ratio": a.ratio} if mode.name.startswith("INT4") else {}
    compressed = nncf.compress_weights(model, mode=mode, **kw)

    a.out.mkdir(parents=True, exist_ok=True)
    ov.save_model(compressed, a.out / "smolvla.xml")
    for name in COPY:
        if (a.src / name).exists():
            shutil.copy2(a.src / name, a.out / name)
    (a.out / "convert_log.json").write_text(json.dumps({
        "src": str(a.src), "mode": a.mode, **kw,
        "nncf": metadata.version("nncf"), "openvino": metadata.version("openvino"),
    }, indent=2))
    size = (a.out / "smolvla.bin").stat().st_size / 2**20
    print(f"{a.out}: smolvla.bin {size:.0f} MB")


if __name__ == "__main__":
    main()
