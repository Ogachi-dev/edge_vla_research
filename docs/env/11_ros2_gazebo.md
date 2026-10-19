# env/11. ROS 2 Jazzy＋Gazebo Harmonic＋lerobot_ws

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章に穴埋めはありません。コマンドを順に実行し、各 Step の「確かめる」をすべてチェックしてから次へ進みます
> - ゴール：SO-101 が Gazebo と MoveIt で動き、関節1〜6の対応と符号の表ができている（34章・37章の根拠になる）

## 1. 目的と計画書の対応

- Ubuntu 24.04（WSL2）上に ROS 2 Jazzy と Gazebo Harmonic を入れ、SO-101 を Gazebo と MoveIt で動かせる状態にする。
- 計画書の対応箇所：「環境構成（OS / ROS、ロボットモデル）」、日程D1（トラック2）

## 2. 前提知識・事前に読む資料

- ROS 2 Jazzyのインストール（deb）：<https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>
- ROS 2とGazeboの組み合わせ：<https://gazebosim.org/docs/harmonic/ros_installation/>
- ros_gz：<https://github.com/gazebosim/ros_gz/tree/jazzy>
- gz_ros2_control：<https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html>
- MoveIt 2（Jazzy）：<https://moveit.picknik.ai/main/index.html>
- lerobot_ws：<https://github.com/Ar-Ray-code/lerobot_ws>（ブランチ `sts_servo`）

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `ros2_ws/src/lerobot_ws/` | 外部リポジトリ（clone のみ。変更しない） |
| `ros2_ws/src/evla_sim/` | 自作パッケージの骨組み（中身は [31](../track2/31_gazebo_scene.md) で作る） |
| `scripts/ros_env.sh` | ROS環境を読み込むシェル（`source` 用） |
| `docs/notes/env_setup.md` | インストールのログ、`glxinfo -B` の結果、RTF |

## 4. インタフェース仕様

### 4.1 lerobot_ws の構成（調査で確認した事実）

| 項目 | 値 |
| --- | --- |
| パッケージ | `lerobot_description`、`lerobot_controller`、`lerobot_moveit` |
| URDF | `lerobot_description/urdf/so101.urdf.xacro`（本体は `so101_base.xacro`） |
| リンク | `world` → `base` → `shoulder` → `upper_arm` → `lower_arm` → `wrist` → `gripper` → `jaw` |
| 関節名 | **`"1"`〜`"6"`**（数字の文字列） |
| 関節の対応 | 1=shoulder_pan、2=shoulder_lift、3=elbow_flex、4=wrist_flex、5=wrist_roll、6=gripper（親子リンクから推定。**【要確認】** 実際に動かして符号も確かめる） |
| コントローラ | `arm_controller`（JointTrajectoryController、関節1〜5）、`gripper_controller`（JointTrajectoryController、関節6）、`joint_state_broadcaster` |
| 制御周期 | `update_rate: 10`（Hz）← **低すぎる**（[31](../track2/31_gazebo_scene.md) で上書きする） |
| MoveItのグループ | `arm`（base_joint＋1〜5）、`gripper`（6）。IKはKDL、`position_only_ik: False` |
| Gazebo起動 | `empty.sdf` を `-r` で起動し、`/clock` だけをbridgeする |

```bash
# 起動順（READMEより）
ros2 launch lerobot_description so101_gazebo.launch.py
ros2 launch lerobot_controller so101_follower_controller.launch.py   # 【要確認】READMEとファイル名が異なる
ros2 launch lerobot_moveit so101_moveit.launch.py
```

**【要確認】** READMEには `so101_controller.launch.py` とありますが、リポジトリにあるのは `so101_follower_controller.launch.py` です。`ros2 launch lerobot_controller <TAB>` で実際の名前を確認してください。

### 4.2 `scripts/ros_env.sh`（リポジトリに用意済み）

ROS のノードは `evla`（`src/evla`）を import するので、`PYTHONPATH` に `src/` を通す行が要ります（Ubuntu 24.04 のシステム Python には `pip install` できないため）。下は、この行を足す前の雛形です。

```bash
#!/usr/bin/env bash
# 使い方: source scripts/ros_env.sh
# venvが有効なら止める（ROSはシステムPythonで動かす）
if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
source /opt/ros/jazzy/setup.bash
source "$(git rev-parse --show-toplevel)/ros2_ws/install/setup.bash" 2>/dev/null || true
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST   # Jazzyでは ROS_LOCALHOST_ONLY は非推奨
export ROS_DOMAIN_ID=42                          # TODO(you): 他の機器と被らない値
# TODO(you): Gazeboの描画で問題が出たら、ここで環境変数を調整する（7節を参照）
```

## 5. ハンズオン

各 Step は「**実行する → 出力を確かめる → メモに残す**」の1サイクルです。出力は `docs/notes/env_setup.md` に貼っていきます。ROS のシェルでは venv を有効にしません（`source scripts/ros_env.sh` が venv を止める）。

### Step 1：ROS 2 Jazzy（1時間）

1. 公式手順に従い、aptのリポジトリを追加して `ros-jazzy-desktop` と `ros-dev-tools` を入れる。
2. 追加パッケージ：

```bash
sudo apt install -y \
  ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-moveit ros-jazzy-moveit-py ros-jazzy-xacro \
  python3-zmq python3-msgpack python3-opencv
```

3. `ros-jazzy-ros-gz` を入れると、Gazebo Harmonic（`gz sim`）も一緒に入る。

**確かめる**

- [ ] `source /opt/ros/jazzy/setup.bash && ros2 doctor --report | head` がエラーなく動く
- [ ] `gz sim --version` が 8.x
- [ ] `python3 -c "import zmq, msgpack, cv2, numpy; print(numpy.__version__)"` が `1.26.x`（ROS 側は NumPy 1.26。[01](../01_repository_design.md) のプロセス分離の前提）

### Step 2：WSLgでの描画確認（30分〜）

1. `sudo apt install mesa-utils` → `glxinfo -B` を実行し、`OpenGL renderer string` に `D3D12 (Intel(R) Iris(R) Xe Graphics)` が出ることを確認する。`llvmpipe` と出る場合はソフトウェア描画になっている。
2. `gz sim shapes.sdf` を実行し、GUIが開くことを確認する。

**確かめる**

- [ ] `glxinfo -B` の `OpenGL renderer string` に `D3D12 (Intel(R) Iris(R) Xe Graphics)` が出る（`llvmpipe` ならソフトウェア描画。7節の落とし穴）
- [ ] `gz sim shapes.sdf` の右下の RTF が 1.0 付近

### Step 3：lerobot_ws のビルド（1時間）

```bash
mkdir -p ros2_ws/src && cd ros2_ws/src
git clone -b sts_servo https://github.com/Ar-Ray-code/lerobot_ws.git
cd .. && rosdep update && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

- `lerobot_ws` はリポジトリ全体が colcon ワークスペースです。`ros2_ws/src/lerobot_ws/src/*` の下にパッケージがありますが、colcon はサブディレクトリまで探索するので、このままでビルドできます。
- **ルート側の `.gitignore` に `ros2_ws/src/lerobot_ws/` を追加する**か、git submodule として固定する（**【判断】** submoduleを推奨。コミットハッシュを固定でき、再現性が保てる）。

**確かめる**

- [ ] `colcon build` が `Summary: N packages finished` で終わり、失敗がない
- [ ] `source scripts/ros_env.sh && ros2 pkg list | grep lerobot` で lerobot_ws のパッケージが見える
- [ ] lerobot_ws のコミット（`git -C ros2_ws/src/lerobot_ws rev-parse HEAD`）をメモに残した

### Step 4：起動確認（1時間）

1. 3つのlaunchを順番に起動し、RVizのMotionPlanningパネルからplan＆executeを1回行う。
2. `ros2 topic hz /joint_states` を実行する（10Hz前後なら、コントローラの `update_rate` が効いている）。
3. `ros2 control list_controllers` で3つのコントローラが `active` になっていることを確認する。
4. Gazeboの統計を確認する：`gz topic -e -t /stats -n 1`（**【要確認】** world名を付けた `/world/empty/stats` の場合もある。`gz topic -l | grep stats` で探す）。

**確かめる**

- [ ] RViz の MotionPlanning で plan＆execute した動きが、Gazebo 上のロボットにも反映される
- [ ] `ros2 control list_controllers` で3つのコントローラが `active`
- [ ] カメラなしの RTF（`real_time_factor`）をメモに残した（31章のカメラ付きの値と比べる）

### Step 5：関節の対応と符号の確認（1時間）

1. `ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory ...` で、関節を1つずつ+0.3rad動かす。
2. どの関節がどう動いたかを表にまとめる（例：関節1が+で上から見て反時計回り）。
3. この表は、[34](../track2/34_episode_recorder.md) でLeRobotの関節名と符号の規約を決めるときの根拠になる。

**確かめる**

- [ ] 関節1〜6それぞれについて「+0.3rad でどちらに動いたか」と、対応する LeRobot の関節名（`shoulder_pan` など）の表がメモにある
- [ ] グリッパ（関節6）は、角度が大きいほど開くのか閉じるのかが分かっている（32章の `gripper_closed_rad`、34章の `GRIPPER_*_DEG` に使う）

## 6. ベストプラクティス

- **ROSのシェルとvenvのシェルを分ける**：ターミナルを「ROS用」と「推論用」に分け、プロンプトの色や表示で区別する。混ぜるとNumPyのABI問題が起きる（[01](../01_repository_design.md)）。
- **外部パッケージはオーバーレイで変更する**：lerobot_wsのURDFやコントローラ設定を変えたいときは、`evla_sim` 側にxacroを作って `xacro:include` し、launch引数で差し替える。
- **起動をスクリプトにまとめる**：3つのlaunchを手で順に起動すると、データ生成を一晩回すときに再現できない。[31](../track2/31_gazebo_scene.md) で1つのlaunchにまとめる。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| GazeboのGUIが真っ黒、または落ちる | WSLgのOpenGLとogre2の相性 | `--render-engine ogre` を試す。最後の手段は `LIBGL_ALWAYS_SOFTWARE=1`（遅い） |
| メッシュが表示されない | `GZ_SIM_RESOURCE_PATH` に `share` ディレクトリが入っていない | lerobot_wsのlaunchは設定している。自作launchでも同様に設定する |
| `ros2 topic list` に他ノードが見えない | DDSの探索範囲、またはドメインIDが違う | 全シェルで `scripts/ros_env.sh` を読む |
| MoveItの実行がタイムアウトする | sim時刻とMoveItの実行時間監視がずれている | 全ノードで `use_sim_time:=true` にする。`trajectory_execution.allowed_execution_duration_scaling` を大きくする |
| RTFが低い（0.3など） | カメラ描画の負荷。WSLgの描画が遅い | カメラなしで測り直して切り分ける。計画書の基準（RTF 0.5未満ならロックステップ主体）に従う |

## 8. ヒント

<details><summary>L1：関節の符号を確かめる方法</summary>

ROSのトピックで1関節ずつ小さく動かし、RVizとGazeboで動きを目視する。TFで子リンクの姿勢がどう変わったかを数値で見ると、より確実。
</details>

<details><summary>L2：使うコマンド</summary>

`ros2 topic pub --once`、`ros2 run tf2_ros tf2_echo base gripper`、`ros2 control list_controllers`、`ros2 param get /controller_manager update_rate`、`gz topic -l`、`gz topic -e -t <topic> -n 1`、`gz service -l`
</details>

<details><summary>L3：JointTrajectory を1発送るコマンドの形</summary>

```
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: ['1','2','3','4','5'], points: [{positions: [0.3,0,0,0,0], time_from_start: {sec: 2}}]}"
```
</details>

## 9. 完了条件（DoD）

- [ ] `glxinfo -B` の結果がメモにあり、ハードウェア描画かどうか分かっている
- [ ] SO-101がGazeboに表示され、RVizのMoveItからplan＆executeでき、Gazebo上でも動く
- [ ] `ros2 control list_controllers` で3コントローラが active
- [ ] 関節1〜6の対応と符号の表がメモにある
- [ ] Gazeboの、カメラなしのRTFを記録した

## 10. 理解度チェック・発展課題

- `gz_ros2_control` は、Gazeboの物理ステップとros2_controlの `update_rate` をどう対応させているか。`update_rate` を物理ステップより高くするとどうなるか。
- `robot_state_publisher` が出すTFと、Gazeboが持っている真の姿勢は、どういう場合にずれるか。
- 発展：5自由度の腕で、任意の位置・任意の姿勢に手先を置けないのはなぜか。pick&placeのどの場面で問題になるか（→ [33](../track2/33_moveit_demo_gen.md)）。
