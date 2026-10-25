# env/11. ROS 2 Humble＋Gazebo Fortress＋lerobot_ws

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章に穴埋めはありません。コマンドを順に実行し、各 Step の「確かめる」をすべてチェックしてから次へ進みます
> - ゴール：SO-101 が Gazebo と MoveIt で動き、関節1〜6の対応と符号の表ができている（34章・37章の根拠になる）

## 1. 目的と計画書の対応

- Ubuntu 22.04（WSL2）上に ROS 2 Humble と Gazebo Fortress を入れ、SO-101 を Gazebo と MoveIt で動かせる状態にする。
- **【計画書との差分】** 計画書は Ubuntu 24.04＋Jazzy＋Harmonic だが、検証端末の WSL（22.04＋Humble 導入済み）に合わせて、Humble の apt の標準の組み合わせ（Fortress）にした。理由と影響は [decisions](../decisions.md)。
- 計画書の対応箇所：「環境構成（OS / ROS、ロボットモデル）」、日程D1（トラック2）

## 2. 前提知識・事前に読む資料

- ROS 2 Humbleのインストール（deb）：<https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>
- ROS 2とGazeboの組み合わせ：<https://gazebosim.org/docs/fortress/ros2_integration/>
- ros_gz：<https://github.com/gazebosim/ros_gz/tree/humble>
- gz_ros2_control：<https://control.ros.org/humble/doc/gz_ros2_control/doc/index.html>
- MoveIt 2（Humble）：<https://moveit.picknik.ai/humble/index.html>
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

lerobot_ws は Jazzy＋Harmonic 向けに書かれています。Humble＋Fortress で使う部品は次のとおり確かめました（2026-10、ソースで確認）。

| 部品 | Humble＋Fortress での扱い |
| --- | --- |
| `gz_ros2_control`（`ros-humble-gz-ros2-control` 0.7.x） | URDF の書き方（`gz_ros2_control-system`、`gz_ros2_control/GazeboSimSystem`）は Jazzy と同じ。Fortress 向けにビルドされている |
| `ros_gz_sim` の `gz_sim.launch.py` | `gz_args` をそのまま使える。Fortress では中で `ign gazebo` を起動する |
| SDF のプラグイン名（`gz-sim-*-system`、`gz::sim::systems::*`） | Fortress（ign-gazebo6）はファイル名の `gz-sim` を `ignition-gazebo` に読み替え、クラス名にも `gz::sim::` の別名を登録しているので、そのまま読める |
| `ros_gz_bridge` の型名（`gz.msgs.*`） | `ignition.msgs.*` と `gz.msgs.*` のどちらも受け付ける |
| CLI | `gz sim` / `gz topic` / `gz service` / `gz sdf` ではなく、**`ign gazebo` / `ign topic` / `ign service` / `ign sdf`** |

**【要確認】** lerobot_ws 自体（launch・MoveIt の設定）が Humble でそのままビルド・起動できるかは、Step 3〜4 で確かめる。

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

ROS のノードは `evla`（`src/evla`）を import するので、`PYTHONPATH` に `src/` を通す行が要ります（システムの Python に `pip install` すると apt の版と混ざるため）。下は、この行を足す前の雛形です。

```bash
#!/usr/bin/env bash
# 使い方: source scripts/ros_env.sh
# venvが有効なら止める（ROSはシステムPythonで動かす）
if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
source /opt/ros/humble/setup.bash
source "$(git rev-parse --show-toplevel)/ros2_ws/install/setup.bash" 2>/dev/null || true
export PYTHONNOUSERSITE=1                       # ~/.local の NumPy 2 を読まない（7節の落とし穴）
export ROS_LOCALHOST_ONLY=1                      # Humble には ROS_AUTOMATIC_DISCOVERY_RANGE がない
export ROS_DOMAIN_ID=42                          # TODO(you): 他の機器と被らない値
# TODO(you): Gazeboの描画で問題が出たら、ここで環境変数を調整する（7節を参照）
```

## 5. ハンズオン

各 Step は「**実行する → 出力を確かめる → メモに残す**」の1サイクルです。出力は `docs/notes/env_setup.md` に貼っていきます。ROS のシェルでは venv を有効にしません（`source scripts/ros_env.sh` が venv を止める）。

### Step 1：ROS 2 Humble の追加パッケージ（1時間）

1. `ros-humble-desktop` はこの端末に入っている（`dpkg -l ros-humble-desktop`）。入っていなければ、公式手順で apt のリポジトリを追加して `ros-humble-desktop` と `ros-dev-tools` を入れる。
2. 追加パッケージ：

```bash
sudo apt install -y \
  ros-humble-ros-gz ros-humble-gz-ros2-control ros-humble-ros2-controllers \
  ros-humble-moveit ros-humble-moveit-py ros-humble-xacro \
  python3-zmq python3-msgpack python3-opencv
```

3. `ros-humble-ros-gz` を入れると、Gazebo Fortress（`ign gazebo`）も一緒に入る。
4. `pip install --user` で入れた NumPy 2 が `~/.local` にあると、apt の NumPy 1.21 より先に読まれる（`python3 -c "import numpy; print(numpy.__file__)"` で確かめる）。ROS のシェルでは `scripts/ros_env.sh` が `PYTHONNOUSERSITE=1` にして読まないようにしている。

**確かめる**

- [ ] `source scripts/ros_env.sh && ros2 doctor --report | head` がエラーなく動く
- [ ] `ign gazebo --version` が 6.x
- [ ] `source scripts/ros_env.sh && python3 -c "import zmq, msgpack, cv2, numpy; print(numpy.__version__, numpy.__file__)"` が `1.21.x` で、場所が `/usr/lib/python3/dist-packages`（ROS 側は NumPy 1.21。[01](../01_repository_design.md) のプロセス分離の前提）

### Step 2：WSLgでの描画確認（30分〜）

1. `sudo apt install mesa-utils` → `glxinfo -B` を実行し、`OpenGL renderer string` に `D3D12 (Intel(R) Iris(R) Xe Graphics)` が出ることを確認する。`llvmpipe` と出る場合はソフトウェア描画になっている。
2. `Max core profile version` を見る。WSLg の D3D12 ドライバ（Mesa 23.2）は **4.1** までで、Fortress の既定の描画エンジン ogre2（ogre-next 2.2）が使う機能（4.3 以降）がない。そのまま起動すると、GUI が白いまま `Ogre::UnimplementedException ... GL3PlusTextureGpu::copyTo` で落ちる。カメラのセンサも同じ理由で落ちる。
3. 描画エンジンを ogre（1.9）にする。GUI の既定は `~/.ignition/gazebo/6/gui.config`（初回の起動で作られる）の `<engine>` なので、ここを書き換える（lerobot_ws の launch のように、起動オプションを変えられない場合にも効く）。

```bash
cp ~/.ignition/gazebo/6/gui.config ~/.ignition/gazebo/6/gui.config.bak-ogre2
sed -i 's#<engine>ogre2</engine>#<engine>ogre</engine>#' ~/.ignition/gazebo/6/gui.config
```

4. `ign gazebo shapes.sdf` を実行し、GUIが開くことを確認する（一度だけ試すなら `--render-engine ogre` を付けてもよい）。`evla_sim` の world と `sim.launch.py` は ogre を指定してある（31章）。

**確かめる**

- [ ] `glxinfo -B` の `OpenGL renderer string` に `D3D12 (Intel(R) Iris(R) Xe Graphics)` が出る（`llvmpipe` ならソフトウェア描画。7節の落とし穴）
- [ ] `glxinfo -B` の `Max core profile version` をメモに残した（4.1 なら ogre を使う）
- [ ] `ign gazebo shapes.sdf` の右下の RTF が 1.0 付近

### Step 3：lerobot_ws のビルド（1時間）

```bash
mkdir -p ros2_ws/src && cd ros2_ws/src
git clone -b sts_servo https://github.com/Ar-Ray-code/lerobot_ws.git
cd .. && rosdep update && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

- lerobot_ws は Jazzy＋Harmonic 向けです。Humble でビルドや起動に失敗したら、エラーと直し方をメモに残す。直すときは lerobot_ws を書き換えず、`evla_sim` 側のオーバーレイ（xacro の include、launch の引数）で吸収する（[01](../01_repository_design.md) 6節）。
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
4. Gazeboの統計を確認する：`ign topic -e -t /stats -n 1`（**【要確認】** world名を付けた `/world/empty/stats` の場合もある。`ign topic -l | grep stats` で探す）。

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
| GazeboのGUIが白いまま `Ogre::UnimplementedException`（`GL3PlusTextureGpu::copyTo`）で落ちる | WSLg の OpenGL が 4.1 までで、ogre2 が使う機能がない（この端末で確認） | ogre にする（Step 2）。`--render-engine ogre`、または `~/.ignition/gazebo/6/gui.config` の `<engine>ogre</engine>`。最後の手段は `LIBGL_ALWAYS_SOFTWARE=1`（遅い） |
| メッシュが表示されない | リソースパスに `share` ディレクトリが入っていない | lerobot_wsのlaunchは `GZ_SIM_RESOURCE_PATH` を設定している。Fortress が読むのは `IGN_GAZEBO_RESOURCE_PATH` なので、見えなければ同じ値を `IGN_GAZEBO_RESOURCE_PATH` にも入れる（**【要確認】**） |
| ROS のノードで `numpy.core.multiarray failed to import` | `~/.local` の NumPy 2 を、apt の NumPy 1.21 でビルドされた C 拡張（cv_bridge など）が読んだ | `source scripts/ros_env.sh`（`PYTHONNOUSERSITE=1`）で起動する |
| `colcon build` で `No module named 'catkin_pkg'` | venv を有効にしたシェルでビルドし、CMake が venv の Python を拾った | `deactivate` して `build/<pkg>` を消してからビルドし直す |
| `ros2 topic list` に他ノードが見えない | DDSの探索範囲、またはドメインIDが違う | 全シェルで `scripts/ros_env.sh` を読む |
| MoveItの実行がタイムアウトする | sim時刻とMoveItの実行時間監視がずれている | 全ノードで `use_sim_time:=true` にする。`trajectory_execution.allowed_execution_duration_scaling` を大きくする |
| RTFが低い（0.3など） | カメラ描画の負荷。WSLgの描画が遅い | カメラなしで測り直して切り分ける。計画書の基準（RTF 0.5未満ならロックステップ主体）に従う |

## 8. ヒント

<details><summary>L1：関節の符号を確かめる方法</summary>

ROSのトピックで1関節ずつ小さく動かし、RVizとGazeboで動きを目視する。TFで子リンクの姿勢がどう変わったかを数値で見ると、より確実。
</details>

<details><summary>L2：使うコマンド</summary>

`ros2 topic pub --once`、`ros2 run tf2_ros tf2_echo base gripper`、`ros2 control list_controllers`、`ros2 param get /controller_manager update_rate`、`ign topic -l`、`ign topic -e -t <topic> -n 1`、`ign service -l`
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
