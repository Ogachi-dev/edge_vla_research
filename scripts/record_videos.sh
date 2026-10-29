#!/usr/bin/env bash
# 各モードの推論動画を撮る（レポート用）。同じタスク・同じ初期状態で、閉ループの設定を1エピソードずつ回す
#   モードB（ロックステップ）：OV GPU、vla.cpp OV、PyTorch CPU
#   モードC（非同期、仮想の遅延 500 ms）：OV GPU
# モードAは固定した観測に推論するだけでシミュレーションを動かさないので、動画はない。
# run は results/runs ではなく results/videos/<名前>/ に移す（表と図に少ないエピソードの行を混ぜない）。
# 最後に、横に並べた動画を results/videos/task<番号>_{sim,wallclock}.mp4 に作る。
# 使い方: scripts/record_videos.sh [タスク番号 ...]   （省くと 0 と 4。約 15 分）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
unset PYTHONPATH AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source .venv-bench/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(0 4)
TASK_JSON="[$(IFS=,; echo "${TASKS[*]}")]"
OUT=results/videos

record() {  # record <名前> <設定ファイル>
  local name=$1 cfg=$2
  echo "=== $(date -Is) $name"
  python scripts/eval_closed_loop.py --config "$cfg" \
    --set save_video='"all"' --set task_ids="$TASK_JSON" --set episodes_per_task=1 | tee "$OUT/$name.log"
  local run_id
  run_id=$(tail -1 "$OUT/$name.log" | cut -d: -f1)
  rm -rf "${OUT:?}/$name"
  mv "results/runs/$run_id" "$OUT/$name"
}
mkdir -p "$OUT"
record modeB_ov_gpu configs/closed_loop/modeB_ov_gpu.yaml
record modeB_vlacpp_ov configs/closed_loop/modeB_vlacpp_ov.yaml
record modeB_torch configs/closed_loop/modeB_lockstep.yaml
record modeC_ov_gpu configs/closed_loop/modeC_async.yaml

for t in "${TASKS[@]}"; do
  v() { ls "$OUT/$1/videos/task${t}_ep0_"*.mp4; }
  args=("B ov/GPU fp16=$(v modeB_ov_gpu)" "B vla.cpp OV=$(v modeB_vlacpp_ov)" "B torch fp32=$(v modeB_torch)")
  python scripts/make_video_grid.py --out "$OUT/task${t}_sim.mp4" "${args[@]}" "C async ov/GPU 500ms=$(v modeC_ov_gpu)"
  # モードCの遅延は仮想時間なので、実時間の動画には入れない
  python scripts/make_video_grid.py --out "$OUT/task${t}_wallclock.mp4" --wallclock "${args[@]}"
done
echo "=== done $(date -Is)"
