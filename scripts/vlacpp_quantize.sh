#!/usr/bin/env bash
# 公開 GGUF から量子化版を作る（vla.cpp の変換用 venv で実行）。docs/track1/25_runtime_vlacpp.md Step 4
set -euo pipefail
VLA=third_party/vla.cpp
SRC=data/models/gguf/smolvla-libero.gguf
for t in Q8_0 Q4_0; do
  lower=$(echo "$t" | tr 'A-Z' 'a-z')
  python "$VLA/scripts/quantize_gguf.py" --in "$SRC" --out "data/models/gguf/smolvla-libero-${lower}.gguf" --type "$t"  # 【穴①】量子化の種類を指定する引数
  python "$VLA/scripts/quantize_gguf.py" --in "$SRC" --out "data/models/gguf/smolvla-libero-${lower}-vision.gguf" --type "$t" --vision
done
ls -la data/models/gguf/
