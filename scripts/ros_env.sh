#!/usr/bin/env bash
# ROS の環境を読み込む。使い方: source scripts/ros_env.sh （docs/env/11_ros2_gazebo.md 4.2）
# venv が有効なら止める（ROS はシステムの Python で動かす）
if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
REPO="$(git rev-parse --show-toplevel)"
source /opt/ros/jazzy/setup.bash
source "$REPO/ros2_ws/install/setup.bash" 2>/dev/null || true
# ROS のノードから evla（src/evla）を import できるようにする。
# Ubuntu 24.04 のシステム Python には pip install できない（PEP 668）ので、PYTHONPATH で通す
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   # Jazzy では ROS_LOCALHOST_ONLY は非推奨
export ROS_DOMAIN_ID=42                          # TODO(you): 他の機器と被らない値
