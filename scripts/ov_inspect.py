#!/usr/bin/env python3
"""OpenVINO IR の入出力（名前・形・型）と、演算の種類を表示する。docs/track1/24_runtime_openvino.md Step 2

例: python scripts/ov_inspect.py data/models/ov/smolvla-libero-fp16-ov/smolvla.xml
"""

from __future__ import annotations

import sys
from collections import Counter

import openvino as ov

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい


def main(xml: str) -> None:
    model = ___  # 【穴①】IR（.xml/.bin）を読み込む（まだコンパイルしない）
    print("== inputs")
    for p in model.inputs:
        print(f"  {p.get_any_name():40s} {p.get_partial_shape()}  {p.get_element_type()}")
    print("== outputs")
    for p in model.outputs:
        print(f"  {p.get_any_name():40s} {p.get_partial_shape()}  {p.get_element_type()}")
    ops = ___(op.get_type_name() for op in model.get_ordered_ops())  # 【穴②】要素ごとの出現回数を数える collections のクラス
    print("== ops (top 25)")
    for name, n in ops.most_common(25):
        print(f"  {name:30s} {n}")
    rand = [k for k in ops if "Random" in k]
    print("== 乱数演算:", rand or "なし（ノイズを外から渡せるか、入力を確認）")


if __name__ == "__main__":
    main(sys.argv[1])
