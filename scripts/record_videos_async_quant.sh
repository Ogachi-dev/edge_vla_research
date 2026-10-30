#!/usr/bin/env bash
# 量子化と非同期の推論動画を撮る（レポート用。scripts/record_videos.sh の続き）
#   モードB（ロックステップ）：OV GPU f16 の重み fp16・w8a16（8bit）・int4（4bit）
#   モードC（非同期、実測の推論時間を遅延にする）：OV GPU fp16 の async_threshold 10・20、8bit・4bit の threshold 20
# 初期状態 0 と 1 を撮り、並べた動画には初期状態 1 を使う（タスク 4・7 では設定ごとの成否が分かれた）。
# 並べた動画（results/videos/）
#   quant_task<番号>.mp4     fp16 / 8bit / 4bit をシミュレーション時間で並べる（行動の違いを比べる）
#   realtime_task<番号>.mp4  同期の 8bit（推論の待ちを含む実時間）と、非同期の threshold 10・20（シミュレーション時間
#                            ＝実時間に相当）を並べる。実時間で動かしたときの見え方を比べる
#   async_quant_task<番号>.mp4  非同期（threshold 20）の fp16 / 8bit / 4bit をシミュレーション時間で並べる
# 撮影済みの設定（全タスクの動画がそろっているもの）は撮り直さない。撮り直すときは FORCE=1 を付ける
# 使い方: scripts/record_videos_async_quant.sh [タスク番号 ...]   （省くと 4 と 7。約 20 分）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
unset PYTHONPATH AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source .venv-bench/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(4 7)
TASK_JSON="[$(IFS=,; echo "${TASKS[*]}")]"
OUT=results/videos
EP=1   # 並べた動画に使う初期状態

record() {  # record <名前> <設定ファイル> [--set ...]
  local name=$1 cfg=$2
  shift 2
  if [ "${FORCE:-0}" != 1 ] && recorded "$name"; then
    echo "=== $name は撮影済み（撮り直すときは FORCE=1）"
    return
  fi
  echo "=== $(date -Is) $name"
  python scripts/eval_closed_loop.py --config "$cfg" "$@" \
    --set save_video='"all"' --set task_ids="$TASK_JSON" --set episodes_per_task=$((EP + 1)) | tee "$OUT/$name.log"
  local run_id
  run_id=$(tail -1 "$OUT/$name.log" | cut -d: -f1)
  rm -rf "${OUT:?}/$name"
  mv "results/runs/$run_id" "$OUT/$name"
}
recorded() {  # 全タスクの、使う初期状態の動画がそろっているか
  local t
  for t in "${TASKS[@]}"; do
    ls "$OUT/$1/videos/task${t}_ep${EP}_"*.mp4 >/dev/null 2>&1 || return 1
  done
}
mkdir -p "$OUT"
B=configs/closed_loop/modeB_ov_gpu.yaml
C=configs/closed_loop/modeC_async_measured.yaml
record q_fp16 "$B"
record q_w8a16 "$B" --set policy.model=data/models/ov/smolvla-libero-w8a16-ov
record q_int4 "$B" --set policy.model=data/models/ov/smolvla-libero-int4-ov
record async_th10 "$C" --set executor.async_threshold=10
record async_th20 "$C" --set executor.async_threshold=20
record async_th20_w8a16 "$C" --set executor.async_threshold=20 --set policy.model=data/models/ov/smolvla-libero-w8a16-ov
record async_th20_int4 "$C" --set executor.async_threshold=20 --set policy.model=data/models/ov/smolvla-libero-int4-ov

for t in "${TASKS[@]}"; do
  v() { ls "$OUT/$1/videos/task${t}_ep${EP}_"*.mp4; }
  python scripts/make_video_grid.py --out "$OUT/quant_task${t}.mp4" \
    "B fp16=$(v q_fp16)" "B 8bit (w8a16)=$(v q_w8a16)" "B 4bit (int4)=$(v q_int4)"
  python scripts/make_video_grid.py --out "$OUT/realtime_task${t}.mp4" --wallclock \
    "B sync 8bit=$(v q_w8a16)" "C async th10=$(v async_th10)|sim" "C async th20=$(v async_th20)|sim"
  python scripts/make_video_grid.py --out "$OUT/async_quant_task${t}.mp4" \
    "C async fp16=$(v async_th20)" "C async 8bit (w8a16)=$(v async_th20_w8a16)" "C async 4bit (int4)=$(v async_th20_int4)"
done
echo "=== done $(date -Is)"
