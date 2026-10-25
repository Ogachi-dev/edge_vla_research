#!/usr/bin/env bash
# ROS の環境を読み込む。使い方: source scripts/ros_env.sh （docs/env/11_ros2_gazebo.md 4.2）
# venv が有効なら止める（ROS はシステムの Python で動かす）
if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
REPO="$(git rev-parse --show-toplevel)"
source /opt/ros/humble/setup.bash
source "$REPO/ros2_ws/install/setup.bash" 2>/dev/null || true
# ROS のノードから evla（src/evla）を import できるようにする。
# システムの Python に pip install すると apt の版と混ざるので、PYTHONPATH で通す
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# ~/.local（pip install --user）の NumPy 2 が apt の NumPy 1.21 より先に読まれると、
# cv_bridge などの C 拡張が import で落ちる。ROS のシェルではユーザーの site-packages を読まない
export PYTHONNOUSERSITE=1
# WSLg の GPU 描画（D3D12）は OpenGL 4.1 までで、Gazebo の ogre2 が落ちる（ogre は表示がちらつく）。
# ソフトウェア描画（llvmpipe、OpenGL 4.5）にする。ROS のシェルで起動する Gazebo と RViz に効く（docs/env/11 Step 2）
export LIBGL_ALWAYS_SOFTWARE=1
export ROS_LOCALHOST_ONLY=1                      # Humble には ROS_AUTOMATIC_DISCOVERY_RANGE がない
export ROS_DOMAIN_ID=42                          # TODO(you): 他の機器と被らない値
