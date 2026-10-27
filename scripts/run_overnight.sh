#!/usr/bin/env bash
# 夜間の本番の実行をまとめて順に回す（重要なものから）。docs/notes/2026-10-28.md
#   1. モードB：OV GPU（50エピソード、約 1.5 時間）
#   2. モードB：vla.cpp OV（50エピソード、約 1.5 時間）
#   3. 掃引の段1（モードA、9 設定 × 3 周、約 3 時間）
#   4. モードB：PyTorch CPU の基準（50エピソード、約 3 時間）
# 使い方: setsid scripts/run_overnight.sh > results/overnight.log 2>&1 &
#         止めるときは kill -- -<表示された PGID>
# 1つが失敗しても次へ進む。終わったら python scripts/make_report_tables.py で表と図を作る
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# ROS のシェルから起動しても、ROS の Python パスを読まない（venv と混ざる）
unset PYTHONPATH AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
source .venv-bench/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-glfw}"   # この端末では egl・osmesa が動かず、glfw（WSLg）だけ動いた
echo "PGID=$(ps -o pgid= $$ | tr -d ' ')  start=$(date -Is)"

run() {
  echo "=== $(date -Is) $*"
  "$@" || echo "!!! 失敗 rc=$? : $*"
}
run python scripts/eval_closed_loop.py --config configs/closed_loop/modeB_ov_gpu.yaml
run python scripts/eval_closed_loop.py --config configs/closed_loop/modeB_vlacpp_ov.yaml
run python scripts/sweep.py configs/sweep/stage1.yaml
run python scripts/eval_closed_loop.py --config configs/closed_loop/modeB_lockstep.yaml
run python scripts/make_report_tables.py
echo "=== done $(date -Is)"
