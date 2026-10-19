#!/usr/bin/env bash
# vla-bench を公開ベンチと同じ手順（ウォームアップ3回＋計測20回）× 3プロセスで回す。docs/track1/25 Step 5
# 例: BUILD=build-sycl bash scripts/vlabench_reference.sh
set -euo pipefail
BUILD=${BUILD:-build}
OUT=results/vlabench/$(date +%Y%m%d-%H%M%S)_${BUILD}
mkdir -p "$OUT"
for i in 1 2 3; do
  "third_party/vla.cpp/$BUILD/vla-bench" -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown \
    | tee "$OUT/run$i.md"
  sleep ___  # 【穴①】プロセス間の冷却時間[秒]
done
echo "→ $OUT"
