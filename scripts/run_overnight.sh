#!/usr/bin/env bash
# 夜間の本番の実行をまとめて順に回す（重要なものから）。docs/notes/2026-10-28.md
#   ov_gpu  モードB：OV GPU（50エピソード、約 30 分）
#   vlacpp  モードB：vla.cpp OV（50エピソード、約 45 分）
#   sweep   掃引の段1（モードA、10 設定 × 3 周、約 3 時間）
#   torch   モードB：PyTorch CPU の基準（50エピソード、約 1.5〜3 時間）
#   sycl    掃引の段1のうち vla.cpp SYCL 版だけ（3 周、約 20 分）。WSL ごと止まったことがあるので最後に単独で回す
#   tables  表と図を作る
# 使い方: setsid scripts/run_overnight.sh [工程 ...] > results/overnight.log 2>&1 &
#         工程を省くと全部を上の順で回す。例：scripts/run_overnight.sh sweep torch sycl tables
#         止めるときは kill -- -<表示された PGID>
# 1つが失敗しても次へ進む。掃引は完了の印があるものを飛ばすので、途中で止まっても同じコマンドで続きから回せる
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# ROS のシェルから起動しても、ROS の Python パスを読まない（venv と混ざる）
unset PYTHONPATH AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source .venv-bench/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-glfw}"   # この端末では egl・osmesa が動かず、glfw（WSLg）だけ動いた
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(ov_gpu vlacpp sweep torch sycl tables)
echo "PGID=$(ps -o pgid= $$ | tr -d ' ')  start=$(date -Is)  steps=${STEPS[*]}"

run() {
  echo "=== $(date -Is) $*"
  "$@" || echo "!!! 失敗 rc=$? : $*"
  sleep 60   # 前の工程のメモリ（GPU の確保を含む）がホストに返るのを待つ
}
for s in "${STEPS[@]}"; do
  case "$s" in
    ov_gpu) run python scripts/eval_closed_loop.py --config configs/closed_loop/modeB_ov_gpu.yaml ;;
    vlacpp) run python scripts/eval_closed_loop.py --config configs/closed_loop/modeB_vlacpp_ov.yaml ;;
    sweep)  run python scripts/sweep.py configs/sweep/stage1.yaml ;;
    torch)  run python scripts/eval_closed_loop.py --config configs/closed_loop/modeB_lockstep.yaml ;;
    sycl)   run python scripts/sweep.py configs/sweep/stage1_sycl.yaml ;;
    tables) python scripts/make_report_tables.py ;;
    *) echo "!!! 知らない工程: $s" ;;
  esac
done
echo "=== done $(date -Is)"
