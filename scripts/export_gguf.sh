#!/usr/bin/env bash
# LeRobot のチェックポイント → GGUF（BF16、Q8_0、Q4_0）。vla.cpp の変換用 venv で実行。docs/track2/36
# 例: bash scripts/export_gguf.sh outputs/train/so101_v1/checkpoints/last/pretrained_model so101_v1
set -euo pipefail
CKPT=$1; NAME=$2
VLA=third_party/vla.cpp
OUT=data/models/$NAME
mkdir -p "$OUT"
python "$VLA/scripts/convert_smolvla_to_gguf.py" --ckpt "$CKPT" --out "$OUT/gguf_bf16.gguf"  # 【穴①】変換元のチェックポイントを渡す引数
for t in Q8_0 Q4_0; do
  python "$VLA/scripts/quantize_gguf.py" --in "$OUT/gguf_bf16.gguf" --out "$OUT/gguf_$(echo $t | tr A-Z a-z).gguf" --type "$t"
done
sha256sum "$CKPT/model.safetensors" "$OUT"/*.gguf | tee "$OUT/sha256.txt"
