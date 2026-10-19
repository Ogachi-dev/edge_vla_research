# track2/31. Gazeboシーン：world・物体・カメラ・TCP・衝突マスク・bridge

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`ros2_ws/src/evla_sim/CMakeLists.txt`（1）、`ros2_ws/src/evla_sim/config/bridge.yaml`（3）、`ros2_ws/src/evla_sim/config/controllers.yaml`（1）、`ros2_ws/src/evla_sim/urdf/so101_evla.urdf.xacro`（4）、`ros2_ws/src/evla_sim/worlds/evla_pickplace.sdf.jinja`（1）、`ros2_ws/src/evla_sim/scripts/render_world.py`（3）、`ros2_ws/src/evla_sim/launch/sim.launch.py`（3）
> - テスト：なし（「9. 完了条件」で確認）
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

## 1. 目的と計画書の対応

- SO-101のpick&place用のGazeboシーン（机、キューブ、箱、カメラ2台、TCPフレーム、DetachableJoint）を作り、ROSとつなぐ。
- lerobot_wsの設定を、オーバーレイで上書きする（制御周期、コントローラ）。
- 計画書の対応箇所：「環境構成（ロボットモデル：TCPフレーム）」「データセット仕様（カメラ）」「DetachableJointによる固定（注意点の表）」、日程D2「Gazeboシーン構築」

## 2. 前提知識・事前に読む資料

- SDFの仕様：<http://sdformat.org/spec>（world、model、link、collision の `surface/contact/collide_bitmask`、sensor）
- URDFの `<gazebo>` 拡張と、固定関節の統合（lumping）：<http://sdformat.org/tutorials?tut=sdformat_urdf_extensions>
- Gazeboのセンサ：<https://gazebosim.org/docs/harmonic/sensors/>
- DetachableJoint：<https://gazebosim.org/api/sim/8/detachablejoints.html>
- ros_gz_bridge のYAML形式：<https://github.com/gazebosim/ros_gz/tree/jazzy/ros_gz_bridge>

### 2.1 調査で確認した事実

| 対象 | 事実 |
| --- | --- |
| SO-101のグリッパ | 固定側の爪は `gripper` リンクのメッシュの一部。可動側の爪が `jaw` リンク（関節 "6"、可動範囲 −0.1745〜1.745 rad） |
| DetachableJoint（gz-sim8のソース） | パラメータは `parent_link`、`child_model`、`child_link`、`attach_topic`、`detach_topic`、`output_topic`、`suppress_child_warning`。状態は `"attached"` / `"detached"` の文字列を出す。**起動時は接続済み**。再接続は**その時点の相対姿勢のまま**固定する |
| DARTの制約 | 接続された初期状態で、親と子が衝突していてはいけない。接触中の再接続はサポートされない |
| UserCommands（gz-sim8） | `/world/<w>/set_pose`、`set_pose_vector`、`light_config`、`visual_config`、`material_color`、`enable_collision`、`disable_collision` などのサービスとトピックがある |
| PosePublisher | `publish_model_pose`、`publish_link_pose`、`use_pose_vector_msg`、`update_frequency`、`topic`。既定のトピックは `<スコープ名>/pose` |
| ros_gz_bridge（jazzy） | `ControlWorld`、`SetEntityPose`、`SpawnEntity`、`DeleteEntity` のサービスをbridgeできる |

## 3. 成果物

```
ros2_ws/src/evla_sim/
├── package.xml, CMakeLists.txt          # ament_cmake（share にインストールするだけ）
├── urdf/so101_evla.urdf.xacro           # lerobot_wsのxacroをinclude＋TCP・手首カメラ・DetachableJoint・衝突マスク
├── config/
│   ├── controllers.yaml                 # update_rate 300 などの上書き
│   ├── bridge.yaml                      # ros_gz_bridge
│   ├── scene.yaml                       # 寸法・配置範囲・カメラ（世界の唯一の正）
│   └── grasp.yaml                       # 把持判定の閾値（32章）
├── worlds/evla_pickplace.sdf.jinja      # world のテンプレート（見た目のランダム化用）
├── models/{table,box,cube}/model.sdf
├── launch/sim.launch.py
└── scripts/render_world.py              # scene.yaml＋乱数 → world SDF を生成
```

## 4. インタフェース仕様

### 4.1 `config/scene.yaml`（シーンの寸法の唯一の正）

```yaml
world_name: evla
physics: {max_step_size: 0.0016666666666666668, real_time_factor: 1.0}   # 1/600秒（30章の4.4）
cube: {size_m: 0.03, mass_kg: 0.02}            # TODO(you): グリッパの開き幅から決める（32章）
box:  {inner_xy_m: [0.12, 0.12], wall_h_m: 0.05, wall_t_m: 0.005}
table: {height_m: 0.0}                          # ロボットの台座を基準にするなら0
spawn_ranges:                                   # ロボットのbase座標系での配置範囲
  cube_xy_m: {x: [0.15, 0.28], y: [-0.12, 0.12]}
  cube_yaw_deg: [-45, 45]
  box_xy_m: {x: [0.15, 0.25], y: [0.15, 0.22]}
  min_cube_box_dist_m: 0.10
cameras:
  top:   {pose_xyzrpy: [0.25, 0.0, 0.55, 0.0, 1.5708, 0.0], hfov_rad: 1.0, width: 640, height: 480, rate_hz: 30}
  wrist: {parent: gripper, pose_xyzrpy: [TODO], hfov_rad: 1.2, width: 640, height: 480, rate_hz: 30}
appearance_ranges:                              # 33章の「見た目の多様化」
  light_intensity: [0.6, 1.2]
  cube_colors: [[1,0,0], [0,0.6,0], [0,0,1], [1,0.8,0]]
  camera_jitter_m: 0.01
  camera_jitter_rad: 0.03
```

### 4.2 URDFオーバーレイ（雛形。中身は自分で埋める）

```xml
<?xml version="1.0"?>
<robot name="so101" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find lerobot_description)/urdf/so101_base.xacro"/>
  <xacro:include filename="$(find lerobot_description)/urdf/so101_ros2_control.xacro"/>
  <!-- so101_gazebo.xacro は include しない（コントローラYAMLのパスを差し替えるため） -->

  <!-- TCP：両爪先端の中間（閉じた状態）。値は5節 Step 3 で求める -->
  <link name="tcp"/>
  <joint name="tcp_joint" type="fixed">
    <parent link="gripper"/><child link="tcp"/>
    <origin xyz="TODO(you)" rpy="TODO(you)"/>
  </joint>

  <!-- 手首カメラ -->
  <link name="wrist_camera_link"/>
  <joint name="wrist_camera_joint" type="fixed">
    <parent link="gripper"/><child link="wrist_camera_link"/>
    <origin xyz="TODO(you)" rpy="TODO(you)"/>
  </joint>
  <gazebo reference="wrist_camera_joint"><preserveFixedJoint>true</preserveFixedJoint></gazebo>
  <gazebo reference="wrist_camera_link">
    <sensor name="wrist_camera" type="camera">
      <!-- TODO(you): update_rate, topic, camera_info_topic, camera/horizontal_fov, image/width,height,format -->
    </sensor>
  </gazebo>

  <!-- 衝突マスク：爪とキューブだけ衝突しないようにする（4.4） -->
  <!-- TODO(you): gripper / jaw の collision に collide_bitmask を設定する方法を確かめる（【要確認】5節 Step 4） -->

  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin">
      <parameters>$(find evla_sim)/config/controllers.yaml</parameters>
    </plugin>
    <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
      <parent_link>gripper</parent_link>
      <child_model>cube</child_model>
      <child_link>cube_link</child_link>
      <attach_topic>/grasp/attach</attach_topic>
      <detach_topic>/grasp/detach</detach_topic>
      <output_topic>/grasp/state</output_topic>
    </plugin>
  </gazebo>
</robot>
```

### 4.3 `config/bridge.yaml`（雛形）

```yaml
- ros_topic_name: /clock
  gz_topic_name: /clock
  ros_type_name: rosgraph_msgs/msg/Clock
  gz_type_name: gz.msgs.Clock
  direction: GZ_TO_ROS
- ros_topic_name: /camera/top/image
  gz_topic_name: /camera/top/image
  ros_type_name: sensor_msgs/msg/Image
  gz_type_name: gz.msgs.Image
  direction: GZ_TO_ROS
# TODO(you): /camera/top/camera_info、/camera/wrist/{image,camera_info}
# TODO(you): /grasp/attach, /grasp/detach（ROS_TO_GZ, std_msgs/msg/Empty ↔ gz.msgs.Empty）
# TODO(you): /grasp/state（GZ_TO_ROS, std_msgs/msg/String ↔ gz.msgs.StringMsg）
# TODO(you): /model/cube/pose（GZ_TO_ROS, tf2_msgs/msg/TFMessage ↔ gz.msgs.Pose_V）
```

サービスは、YAMLの代わりに `parameter_bridge` の引数で指定します（ドキュメントにYAMLでの例がないため）。

```
/world/evla/control@ros_gz_interfaces/srv/ControlWorld
/world/evla/set_pose@ros_gz_interfaces/srv/SetEntityPose
```

### 4.4 衝突マスクの設計

SDFの `collide_bitmask` は、**2つのcollisionのビットANDが0なら衝突しない**という規則です。

| collision | bitmask | キューブ（0x02）との関係 |
| --- | --- | --- |
| キューブ | `0x02` | — |
| 机、箱 | 既定（`0xFFFF`） | 0x02 & 0xFFFF ≠ 0 → 衝突する |
| `gripper`、`jaw` | `0x01` | 0x01 & 0x02 = 0 → 衝突しない |
| その他のロボットリンク | 既定 | 衝突する（腕でキューブを押すことはありうる） |

**【判断】** 爪とキューブの衝突は**常に無効**にする（固定中だけ無効にするのではなく）。

- 理由：実行中に衝突設定を切り替えるのは難しく、DARTの「接触中の再接続は不可」という制約にも当たりやすい。
- 代償：爪がキューブをすり抜けるので、「押して動かす」動きは再現されない。把持判定（32章）の許容範囲で、「掴める位置にいたか」を厳密に判定することで補う。
- 代替案：gz-sim8 の `/world/<w>/disable_collision` サービスで、動的に切り替えられる可能性がある（**【要確認】**）。

### 4.5 カメラの内部・外部パラメータ（34章のメタデータで使う）

- 内部：`fx = fy = (W / 2) / tan(hfov / 2)`、`cx = W / 2`、`cy = H / 2`（正方画素、歪みなし）。**【要確認】** Gazeboが出す `camera_info` の値と照合し、`(W-1)/2` などの違いがないか確かめる。
- 外部：Gazeboのカメラ座標系は**X前方・Y左・Z上**で、OpenCVの光学座標系（**Z前方・X右・Y下**）と違う。メタデータには両方の規約を明記し、OpenCV規約に変換した値も保存する。

## 5. 実装手順

### Step 1：パッケージとlaunchの骨組み（1時間）

1. `ros2 pkg create evla_sim --build-type ament_cmake`。`urdf config worlds models launch scripts` を `share` にインストールする。
2. `sim.launch.py`：lerobot_wsの `so101_gazebo.launch.py` を参考に（コピーではなく理解して）、world と URDF を自分のものに差し替える。
3. `controllers.yaml`：lerobot_wsのものをコピーし、`update_rate: 300` にする。

### Step 2：world と物体（2時間）

1. `models/cube/model.sdf`：リンク名 `cube_link`、慣性はキューブの公式（`I = m·s²/6`）で計算する。collisionに `collide_bitmask 0x02` を設定する。PosePublisherを付ける。
2. `models/box/model.sdf`：底板＋4枚の壁（薄い直方体）。内寸は `scene.yaml` から生成する。
3. world：Physics（`max_step_size`）、UserCommands、SceneBroadcaster、Sensors（`render_engine` は ogre2）の各systemと、光源、地面、机、箱、キューブを置く。
4. `render_world.py`：`scene.yaml` と乱数シードから、Jinjaテンプレートで world SDF を作る（光の強さ、カメラの微小なずれ、机の色など）。

### Step 3：TCPの位置を求める（1〜2時間）

1. RViz で `gripper` と `jaw` を表示し、グリッパを閉じた状態（関節6を閉じ側の限界）にする。
2. `.venv-bench` で `trimesh` を使い、爪のメッシュ（STL）を読み込む。リンク座標系での爪の先端（最も遠い頂点群）の座標を求め、両爪の先端の中点を計算する。
3. その値を `tcp_joint` の `origin` に書き、RVizでTCPの座標軸が爪の間に表示されることを目視で確かめる。
4. TCP座標系の軸の向きを決めて記録する（例：Z＝接近方向、Y＝爪の開閉方向）。**この規約は32章の判定で使う**。

### Step 4：衝突マスクの確認（1〜2時間）

1. URDFの `<gazebo reference="gripper">` で、collisionの `collide_bitmask` を設定できるか試す。`xacro ... > /tmp/so101.urdf && gz sdf -p /tmp/so101.urdf` でSDFに変換し、`collide_bitmask` が入っているか確認する。
2. 入らない場合は、SDFに変換してから編集する方式（`render_world.py` でSDFを後処理する）に切り替える。
3. 動作確認：グリッパを開いた状態でキューブの真上から降ろし、**爪がキューブをすり抜け、キューブは机の上に残る**ことを確かめる。

### Step 5：カメラ（1.5時間）

1. 上方カメラはworldに固定、手首カメラはURDFの `wrist_camera_link` に付ける。
2. `ros2 run rqt_image_view rqt_image_view` で2つの画像を確認する。
3. `ros2 topic hz /camera/top/image` が30Hz（sim時刻）で、`camera_info` の `K` が4.5の式と合うことを確かめる。
4. カメラ付きのRTFを測り、記録する（計画書の基準：0.5未満ならロックステップを主体にする）。

### Step 6：DetachableJointとサービス（1時間）

1. 起動直後の状態が `attached` であることを `ros2 topic echo /grasp/state` で確かめる。
2. `ros2 topic pub --once /grasp/detach std_msgs/msg/Empty {}` で外し、`detached` になることを確かめる。
3. `set_pose` サービスでキューブを移動できることを確かめる。

## 6. ベストプラクティス

- **寸法は1つのYAMLにだけ書く**：キューブの大きさが、SDF、判定の閾値、メタデータにばらばらに書かれていると、必ずずれる。`scene.yaml` から生成・読み込みする。
- **物理パラメータを現実的にする**：質量と慣性が極端だと、DARTが不安定になる。キューブ20g、一辺3cmなどの現実的な値にする。
- **見た目のランダム化はworld生成時に行う**：実行中の変更（`visual_config` など）は手軽だが、反映のタイミングが分かりにくい。エピソードのまとまり（例：20エピソード）ごとにworldを作り直す方式が確実。
- **TCPは数値で確かめる**：目視でずれていても気づかないことがある。TFで `tcp` の位置を出し、閉じた爪の先端の位置と比べる。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 起動直後、キューブが腕と一緒に浮く | DetachableJointは起動時に接続済み | エピソード開始時、腕を動かす前に detach する |
| キューブがはじけ飛ぶ | 初期状態でキューブと爪が重なっている（DARTの制約）。衝突マスクが効いていない | キューブの初期位置を爪から離す。Step 4で確認する |
| 手首カメラが表示されない | 固定関節の統合（lumping）でリンクが消え、センサが行方不明になった | `preserveFixedJoint`。またはセンサを `gripper` に直接付け、姿勢で位置を調整する |
| DetachableJointが `child model not found` | ロボットのspawn時点でキューブが存在しない | キューブはworld SDFに入れておく（後からspawnしない） |
| カメラ画像が真っ黒 | Sensors systemがない、またはrender engineが使えない | worldに `gz-sim-sensors-system` を入れる。ogre2が駄目ならogreを試す |
| `/clock` が進まない | worldをpause状態で起動した（`-r` なし） | 仕様。生成時は episode_runner が step する（34章） |

## 8. ヒント

<details><summary>L1：爪の先端をメッシュからどう求めるか</summary>

メッシュの頂点をリンク座標系に変換し（URDFのcollisionの `origin` を適用する）、爪が伸びる方向の座標が最大の頂点群を「先端」とみなす。その重心を先端の点とする。
</details>

<details><summary>L2：使うツール</summary>

`trimesh.load(path)`、`mesh.apply_transform(T)`、`tf_transformations` 相当の行列計算（`scipy.spatial.transform.Rotation.from_euler("xyz", rpy)`）、`gz sdf -p`（URDF→SDFの変換確認）、`gz topic -e -t /grasp/state`、`gz service -s /world/evla/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --req 'name: "cube", position: {x: 0.2, y: 0, z: 0.02}'`
</details>

<details><summary>L3：render_world.py の流れ</summary>

```
scene = yaml.safe_load(scene.yaml); rng = default_rng(seed)
params = {
  "step": scene.physics.max_step_size,
  "light": rng.uniform(*scene.appearance_ranges.light_intensity),
  "top_cam_pose": jitter(scene.cameras.top.pose_xyzrpy, rng, ...),
  ...
}
sdf = jinja2.Template(template_text).render(**params)
write(out_dir / f"world_{seed}.sdf", sdf); write(out_dir / f"world_{seed}.json", params)   # 使った値を必ず残す
```
</details>

## 9. 完了条件（DoD）

- [ ] `ros2 launch evla_sim sim.launch.py` 1本で、ロボット・机・箱・キューブ・2カメラが起動し、3コントローラが active になる
- [ ] TFに `tcp` があり、閉じた爪の間に位置する（数値で確認した）
- [ ] 爪がキューブをすり抜け、机と箱はキューブと衝突する
- [ ] `/grasp/state` が起動時に `attached` で、detach で `detached` になる
- [ ] `/camera/{top,wrist}/image` が30Hz（sim時刻）、`camera_info` が4.5の式と一致する
- [ ] `set_pose` と `control`（pause / multi_step）のサービスがROSから呼べる
- [ ] カメラ付きのRTFを記録した

## 10. 理解度チェック・発展課題

- 物理ステップを1/600秒にした理由を、30Hzのフレームとの関係で説明せよ。
- 衝突マスクを常に無効にすることで、データセットの「物理的なもっともらしさ」に何が失われるか。それは評価結果の解釈にどう影響するか。
- 発展：Gazeboのカメラ座標系からOpenCVの光学座標系への回転行列を書け。外部パラメータのメタデータにはどちらを保存するのがよいか。
