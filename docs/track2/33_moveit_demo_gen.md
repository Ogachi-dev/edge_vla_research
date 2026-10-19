# track2/33. MoveItでのpick&placeデモ自動生成とランダム化

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**18個**（`src/evla/planning/` の `ik_so101.py` 7、`randomize.py` 5、`episode_plan.py` 1、`build.py` 3、`ros2_ws/src/evla_nodes/evla_nodes/demo_generator.py` 2）
> - ゴール：`pytest tests/planning` が通り、Gazebo で固定配置の pick & place が1回成功する。20エピソードの試運転で、200エピソードにかかる時間を見積もれる
> - 進め方：Step 1〜3 は**ROS なし**（pytest）。Step 4 以降は ROS 側。Step 5〜6 は34章の `episode_runner` を使うので、34章と行き来する

## 1. 目的と計画書の対応

- MoveIt 2で、ランダムな配置のキューブを箱に入れる軌道を自動生成する。
- 配置・軌道・見た目のランダム化と、立て直しの動き（ノイズ注入）を入れる。
- 一晩無人で回しても止まらないようにする。
- 計画書の対応箇所：「データセット仕様（生成方法）」「データの多様化」、リスク「MoveIt生成デモの単調さ」、日程D3〜D4

## 2. 前提知識・事前に読む資料

- MoveIt 2のPython API（MoveItPy）：<https://moveit.picknik.ai/main/doc/examples/motion_planning_python_api/motion_planning_python_api_tutorial.html>
- MoveItの時間パラメータ化（速度・加速度のスケーリング）：<https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html>
- pick_ik（自由度が6未満の腕に向いたIK）：<https://github.com/PickNikRobotics/pick_ik>
- 5自由度アームの逆運動学（幾何的な解法）：平面2リンクの逆運動学＋ベース回転＋手首

### 2.1 5自由度の腕で何ができて、何ができないか（実装前に理解すること）

SO-101の腕は、shoulder_pan・shoulder_lift・elbow_flex・wrist_flex・wrist_rollの5関節です。手先の姿勢（6自由度）のうち、**1自由度は自由に選べません**。

- shoulder_panで、腕が動く**鉛直面の向き**が決まる。
- shoulder_lift・elbow_flex・wrist_flexで、その面内での位置（2）と手先の傾き（1）が決まる。
- wrist_rollで、接近軸まわりの回転が決まる。

→ **接近方向が、ベースの軸を含む鉛直面の中にある姿勢**だけが到達できます。**真上からの把持**（接近方向＝鉛直下向き）は、どの鉛直面にも含まれるので、常に条件を満たします。キューブのヨーには wrist_roll で合わせます。

lerobot_wsのMoveIt設定は `position_only_ik: False`（6自由度の姿勢を要求）のKDLなので、**到達できない姿勢を目標にすると、IKがタイムアウトして失敗**します。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/planning/ik_so101.py` | SO-101の幾何的な逆運動学（真上からの把持用）。ROS非依存 |
| `src/evla/planning/episode_plan.py` | `EpisodePlan`：エピソードを「区間」の列として表すデータ型 |
| `src/evla/planning/randomize.py` | 配置・軌道パラメータのサンプリング |
| `src/evla/planning/build.py` | 区間の列（`EpisodePlan`）を組み立てる。ROS 非依存（関節空間の移動は差し替え可能な関数として受け取る） |
| `evla_nodes/demo_generator.py` | MoveIt で関節空間の移動を計画する（`build.py` に渡す。実行はしない） |
| `configs/datagen/default.yaml` | ランダム化の範囲、区間のパラメータ、エピソード数 |
| `tests/planning/test_ik_so101.py` | IK→FKの往復テスト |

## 4. インタフェース仕様

### 4.1 「計画」と「実行」を分ける

**【判断】** demo_generatorは**軌道を計画するだけ**で、実行は episode_runner（34章）が行う。

- 理由：記録はロックステップ（worldをpauseして少しずつstepする）で行う。MoveItの `execute` はsim時刻の進行を前提に完了を待つので、時計を持つノードと実行するノードを分けると、競合や待ち合わせが複雑になる。
- 計画は、前の区間の終了状態を次の区間の開始状態にして、順に作る（simでは追従誤差が小さいので、これで十分）。

```python
# src/evla/planning/episode_plan.py
@dataclass(frozen=True)
class ArmSegment:
    name: str                                  # "approach", "descend", "lift", "transfer", "lower", "retreat", "home"
    trajectory: "JointTrajectoryData"          # 関節名、時刻[s]、位置[rad]（時間パラメータ化済み）

@dataclass(frozen=True)
class GripperSegment:
    name: str                                  # "close", "open"
    target_rad: float
    duration_s: float

@dataclass(frozen=True)
class WaitSegment:
    name: str
    duration_s: float

Segment = ArmSegment | GripperSegment | WaitSegment

@dataclass(frozen=True)
class EpisodePlan:
    episode_seed: int
    scene: "SceneSample"                       # キューブ・箱の姿勢、色、初期姿勢など
    params: "TrajectoryParams"                 # 接近高さ、速度、ノイズなど
    segments: tuple[Segment, ...]
    perturbations: tuple[dict[str, Any], ...]  # 注入したノイズの記録（34章のメタデータへ）
```

```python
# src/evla/planning/randomize.py
@dataclass(frozen=True)
class SceneSample:
    cube_xy_m: tuple[float, float]; cube_yaw_rad: float; cube_color: tuple[float, float, float]
    box_xy_m: tuple[float, float]
    q_init_rad: tuple[float, ...]              # 腕の初期姿勢（home の周りに小さな乱数）

@dataclass(frozen=True)
class TrajectoryParams:
    approach_height_m: float                   # キューブの上、何mから降りるか
    vel_scale: float; acc_scale: float         # MoveItの速度・加速度のスケーリング
    planner_seed: int
    grasp_offset_m: tuple[float, float, float] # 立て直し用：わざとずらす量（0なら注入なし）
    gripper_close_rad: float                   # キューブ幅相当で止める閉じ角

def sample_scene(rng: np.random.Generator, ranges: Mapping) -> SceneSample: ...
def sample_params(rng: np.random.Generator, ranges: Mapping) -> TrajectoryParams: ...
```

```python
# src/evla/planning/ik_so101.py
@dataclass(frozen=True)
class So101Geometry:            # URDFから読み取ったリンク長とオフセット（31章のTCPを含む）
    ...

def ik_top_down(p_base_tcp: ArrayLike, yaw_rad: float, geo: So101Geometry,
                elbow: Literal["up", "down"] = "up") -> NDArray[np.float64] | None:
    """真上からの把持姿勢（接近方向＝鉛直下向き、開閉軸がyaw_radを向く）の関節角[rad]（5関節）を返す。
    到達できなければ None。yaw は wrist_roll の可動範囲に入るよう90°の倍数だけずらしてよい。"""

def fk_tcp(q: ArrayLike, geo: So101Geometry) -> NDArray[np.float64]:
    """順運動学（テスト用）。ROSのTFと一致することを結合テストで確かめる。"""
```

### 4.2 1エピソードの区間の並び

| # | 区間 | 内容 |
| --- | --- | --- |
| 0 | 初期化 | 腕を `q_init` へ（記録しない） |
| 1 | approach | キューブの真上 `approach_height` の位置へ（関節空間で計画） |
| 2 | descend | 把持の高さまで真下へ降りる（直線に近い軌道。デカルト経路か、細かい中間点） |
| 3 | close | グリッパを `gripper_close_rad` まで閉じる（→ 32章の判定で attach） |
| 4 | lift | 真上へ持ち上げる |
| 5 | transfer | 箱の上へ |
| 6 | lower | 箱の中へ少し降ろす |
| 7 | open | 開く（→ 解放） |
| 8 | retreat | 上へ離れる |
| 9 | wait | 成功判定の静止を待つ |

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章の穴埋め版では、**IK を自作し**（lerobot_ws の KDL は真上からの姿勢で失敗しやすい。2.1節）、MoveIt には関節空間の目標だけを渡します。区間の組み立ても ROS から切り離しています。だから MoveIt が動かなくても、`joint_interp`（関節空間の直線補間）で先に進めます。

### Step 0：準備（10分）

```bash
source .venv-bench/bin/activate
grep -n "【穴\|TODO(you)\|【要確認】" src/evla/planning/*.py ros2_ws/src/evla_nodes/evla_nodes/demo_generator.py configs/datagen/*.yaml
pytest tests/planning -q        # まだ失敗する
```

- [ ] 穴18個が表示され、テストが失敗する
- [ ] `configs/datagen/so101_geometry.yaml` の値が**仮の値**であることを確認した（Step 1 の後半で実物に置き換える）

### Step 1：真上からの把持の IK ― `ik_so101.py`（穴①〜⑦、2〜3時間）

**読む**：ファイル冒頭の docstring（「腕の平面」のモデル）と4.1節。**紙に図を描く**：真上から見た図（パンの角度 ψ）と、腕の平面の側面図（肩 → 肘 → 手首 → TCP、TCP は真下を向く）。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 真上から見た目標の方位角。`atan2` の引数の順 |
| ② | 工具が真下を向くので、手首の軸は TCP の真上 L3 にある。高さの基準を肩（`base_height_m`）にとる |
| ③ | 2リンク（L1, L2）の平面 IK。肩から手首までの距離に余弦定理を使って cos(θ2) を求める |
| ④ | 肩 → 手首の線と、上腕のなす角（`atan2` で、L2 の成分から求める） |
| ⑤ | 工具のピッチ φ = θ1 + θ2 + θ3 = −90° から θ3 を決める |
| ⑥ | 爪の向き（ヨー）＝ ψ + roll なので、roll を逆算する |
| ⑦ | 全関節が可動範囲に入っているか（`np.all` と2つの比較） |

**動かす**

```bash
pytest tests/planning/test_ik_so101.py -v
python -c "
import numpy as np, yaml
from evla.planning.ik_so101 import So101Geometry, ik_top_down, fk_tcp
g = So101Geometry(**{k: tuple(v) if isinstance(v, list) else v for k, v in yaml.safe_load(open('configs/datagen/so101_geometry.yaml')).items()})
q = ik_top_down([0.2, -0.05, 0.015], 0.3, g); print(np.round(q, 3)); print(np.round(fk_tcp(q, g), 4))
print(ik_top_down([1.0, 0.0, 0.0], 0.0, g))
"
```

**確かめる**

- [ ] 3つのテスト（`up` / `down` のパラメータ化を含めて4件）が `PASSED`
- [ ] FK の出力が `[0.2, -0.05, 0.015, -1.5708, 0.3]`（φ = −π/2）。1m 先は `None`（届かない）

**URDF の値に置き換える（`TODO(you)`）**：`so101_geometry.yaml` のリンク長・`sign`・`offset_rad`・関節の範囲を、URDF（xacro 展開後）の joint origin と limit から読み取って書きます。そのあと ROS 側で、**FK と TF が一致する**ことを確かめます。

```bash
source scripts/ros_env.sh
ros2 launch evla_sim sim.launch.py world:=/tmp/evla_world_0.sdf
# 別の端末で：適当な関節角に動かし、その角度で fk_tcp を計算した値と比べる
ros2 run tf2_ros tf2_echo base tcp          # 【要確認】ベースのリンク名
```

- [ ] FK と TF の位置の差が 1mm 未満（ずれるなら、リンク長か `sign` / `offset_rad` が違う）

### Step 2：区間の組み立て ― `episode_plan.py`（穴①）と `build.py`（穴①〜③）（1.5時間）

**読む**：4節の `EpisodePlan`（区間の列）。`build_plan` を上から読み、9つの区間（approach → descend → close → lift → transfer → lower → open → retreat → wait）が何をするかを書き出す。

**埋める**

| 穴 | ねらい |
| --- | --- |
| `episode_plan` ① | 区間の種類で長さの持ち方が違う。腕の区間だけは軌道の最後の時刻が長さになる（`isinstance`） |
| `build` ① | 補間の進み具合 s ∈ [0,1] を、始点と終点で速度0になる3次式（smoothstep：3s² − 2s³）に通す |
| `build` ② | p0 → p1 の直線を n 等分した k 番目の点 |
| `build` ③ | キューブ幅で止まる閉じ角。`params` に入っている |

**動かす**

```bash
pytest tests/planning/test_build.py -v
python -c "
import yaml
from evla.planning.build import PlanConfig, build_plan, joint_interp
from evla.planning.ik_so101 import So101Geometry
from evla.planning.randomize import SceneSample, TrajectoryParams
g = So101Geometry(**{k: tuple(v) if isinstance(v, list) else v for k, v in yaml.safe_load(open('configs/datagen/so101_geometry.yaml')).items()})
sc = SceneSample(cube_xy_m=(0.2, -0.05), cube_yaw_rad=0.3, cube_color=(1, 0, 0), box_xy_m=(0.2, 0.15), q_init_rad=(0, 0, 0, 0, 0))
pr = TrajectoryParams(approach_height_m=0.08, vel_scale=0.5, acc_scale=0.5, planner_seed=0, grasp_offset_m=(0.01, 0, 0), gripper_close_rad=0.1)
plan = build_plan(0, sc, pr, g, PlanConfig(), joint_interp)
print([s.name for s in plan.segments], round(plan.duration_s, 1), plan.perturbations)
"
```

**確かめる**

- [ ] 4つのテストが `PASSED`
- [ ] ずれ（1cm）を入れたので、`descend` の後に `correct` の区間が入り、`perturbations` に `grasp_offset` が記録されている。エピソードの長さ（秒）をメモする

### Step 3：ランダム化 ― `randomize.py`（穴①〜⑤、1時間）

**読む**：`configs/datagen/default.yaml`（ランダム化の範囲）。エピソードごとのシードは `base_seed + episode_index`。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `[lo, hi)` の一様乱数（`rng` のメソッド） |
| ② | キューブと箱の距離（`math` の関数）。近すぎる配置はやり直す（棄却サンプリング） |
| ③ | 色のリストから1つ選ぶ添字 |
| ④ | 初期姿勢に小さなノイズ。home と同じ形の正規乱数 |
| ⑤ | 確率 `perturb_prob` で立て直しを注入する。一様乱数と比べる |

**動かす**

```bash
pytest tests/planning/test_randomize.py -v
```

- [ ] 3つのテストが `PASSED`（同じシードなら同じ配置、範囲と最小距離、注入の割合が約30%）

### Step 4：MoveIt で関節空間の移動を計画する ― `demo_generator.py`（穴①②、1〜2時間）

**読む**：MoveItPy のチュートリアル（2節のリンク）。このクラスは `build_plan` の `plan_joint_move` に**差し替えて渡す**だけのもの（`joint_interp` と同じ形の関数）であることを確認する。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `MoveItPy` から計画グループ（`ARM_GROUP`）を操作するオブジェクトを得るメソッド |
| ② | 目標を**関節角の状態**で与える。開始状態と同じ作り方（`self._state`） |

**動かす**：MoveIt を起動した状態で、home から適当な姿勢への計画を1回試します。

```bash
source scripts/ros_env.sh
python3 -c "
import numpy as np
from evla_nodes.demo_generator import MoveItJointPlanner
traj = MoveItJointPlanner()(np.zeros(5), np.array([0.3, -0.5, 0.8, 0.5, 0.0]), 0.5)
print(traj.joint_names, len(traj.times_s), round(traj.duration_s, 2))
"
```

- [ ] 関節名・点の数・所要時間が表示される。動かなければ【要確認】の行（robot 名、パッケージ名、グループ）を実物に合わせる
- [ ] **それでも動かなければ**、`planner:=interp`（`joint_interp`）で先に進む。理由を `docs/decisions.md` に残す

### Step 5：固定配置で1回成功させる（34章の Step 4 と同時に、2時間）

34章の `episode_runner` ができたら、ランダム化の範囲を**1点に潰した**設定で1エピソード回します。

```bash
cp configs/datagen/default.yaml configs/datagen/fixed.yaml
# fixed.yaml を編集：cube_xy_m・box_xy_m の範囲を1点に、cube_yaw_deg を [0, 0]、perturb_prob を 0、n_episodes を 1
ros2 run evla_nodes episode_runner --ros-args -p use_sim_time:=true \
  -p datagen_config:=configs/datagen/fixed.yaml -p out_dir:=data/raw/fixed_test -p planner:=interp
cat data/raw/fixed_test/ep_*/meta.json
```

- [ ] `meta.json` の `success` が `true`（32章の判定で `success` が出た）
- [ ] 失敗したら `failure_reason`（`grasp_rejected:<条件>` など）を見て、TCP・IK・閾値のどれを直すか決める

### Step 6：試運転と見積もり（1時間）

```bash
cp configs/datagen/default.yaml configs/datagen/trial20.yaml     # n_episodes: 20 に変える
ros2 run evla_nodes episode_runner --ros-args -p use_sim_time:=true \
  -p datagen_config:=configs/datagen/trial20.yaml -p out_dir:=data/raw/trial20 -p planner:=interp
cat data/raw/trial20/generation_stats.json
```

- [ ] 生成の成功率（失敗の内訳つき）と、1エピソードの所要時間（`meta.json` の `wall_s`）が分かる
- [ ] 200エピソードにかかる時間を見積もり、D4の夜に間に合うか判断した
- [ ] キューブ位置の散布図（`meta.json` の `plan.scene`）で、範囲全体がカバーされている
- [ ] 6節の頑健化の項目（IK 失敗時のやり直し、タイムアウトなど）のうち、試運転で問題になったものに対処した

### Step 7：仕上げ（10分）

- [ ] 5ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/planning` を通す
- [ ] `pytest tests/planning` がすべて通る
- [ ] コミットする（`configs/datagen/fixed.yaml`・`trial20.yaml` は残しておくと、後で再現できる）

## 6. ベストプラクティス

- **IKは往復テストで検証する**：`fk_tcp(ik_top_down(p, yaw)) ≈ p` を、作業範囲内のランダムな点で多数回確かめる。さらに、`fk_tcp` がROSのTF（`world → tcp`）と一致することを結合テストで確かめる。
- **失敗を前提に作る**：計画の失敗、IKの解なし、把持の判定に落ちる、などは必ず起きる。失敗したエピソードは捨てて次へ進み、失敗の理由を数える。
- **一晩回すための対策**：
  - 区間ごとのタイムアウト（計画5秒、実行は軌道の時間×3）
  - Nエピソードごと（例：50）にGazeboを再起動する（メモリリーク対策。見た目のランダム化のためのworldの作り直しと兼ねる）
  - 進捗をファイルに書き、途中から再開できるようにする（完了したエピソードのシードを記録する）
  - 生成の成功率が一定以下（例：50%）に落ちたら止めて通知する（何かが壊れている）
- **単調さを数値で確かめる**：生成したデータのキューブ位置の散布図、軌道の時間長のヒストグラムを作る。偏りがあれば範囲を調整する。
- **公開データの作り方を参考にする**：SmolVLAの論文用データ（`lerobot/svla_so100_pickplace`）は、5つのキューブ位置×各10エピソードで、25エピソードでは足りなかったと報告されている。「各変化に十分な数のデモ」を意識して範囲と数を決める。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| IKが頻繁に失敗する | 5自由度で到達できない6自由度の姿勢を要求している | 2.1を読み直す。真上からの姿勢にするか、IKを自作する |
| wrist_rollが大きく回り込む | キューブのヨーに合わせるとき、90°の倍数の選び方が悪い | 現在の角度に最も近い等価なヨー（90°周期）を選ぶ |
| 降下中にキューブを押し飛ばす | 爪以外のリンク（手首）がキューブに当たる。接近の高さが低すぎる | 接近の高さの下限を上げる。MoveItの計画シーンにキューブを入れる |
| 閉じた後に判定が通らない | 降下の高さがずれている（TCPの定義と目標の不一致） | 32章の `grasp_candidate` のログで、どの軸がずれたか見る |
| 一晩で数エピソードしか進まない | RTFが低い。どこかで待ちが発生している | 試運転で1エピソードの時間の内訳を測る |
| 計画の結果が毎回同じ | OMPLのシードを固定していない／逆に毎回変わって再現できない | シードをエピソードのシードから決め、記録する |

## 8. ヒント

<details><summary>L1：真上からの把持のIKを幾何的に解くには</summary>

ベースの真上から見て、目標点の方向がshoulder_panの角度になる。その鉛直面に落とすと、手首の位置（TCPから接近方向に、手首からTCPまでの長さだけ戻った点）を、肩と肘の2リンクで届かせる問題になる（余弦定理）。手先が真下を向くように wrist_flex を決め、最後に wrist_roll でヨーを合わせる。URDFの関節の原点のオフセット（リンクが一直線でない）に注意。
</details>

<details><summary>L2：使うAPI</summary>

MoveItPy：`MoveItPy(node_name=...)`、`get_planning_component("arm")`、`set_start_state(robot_state=...)`、`set_goal_state(robot_state=...)`、`plan()` → `.trajectory`、`RobotState.set_joint_group_positions("arm", q)`。MoveItPyの設定には `moveit_configs_utils.MoveItConfigsBuilder` が必要（**【要確認】** lerobot_moveitのパッケージでそのまま使えるか確かめる。使えなければ `move_group` の `/plan_kinematic_path` サービスを呼ぶ方式に切り替える）。
</details>

<details><summary>L3：demo_generator の1エピソードの疑似コード</summary>

```
scene = sample_scene(rng, ranges); params = sample_params(rng, ranges)
q = scene.q_init; segs = []
for name, target in [("approach", above(scene.cube, params.approach_height)), ("descend", at(scene.cube) + params.grasp_offset_m), ...]:
    q_goal = ik_top_down(target.p, target.yaw, geo)
    if q_goal is None: return Failure("ik", name)
    traj = moveit_plan(start=q, goal=q_goal, vel=params.vel_scale, seed=params.planner_seed)
    if traj is None: return Failure("plan", name)
    segs.append(ArmSegment(name, traj)); q = traj.positions[-1]
    if name == "descend": segs.append(GripperSegment("close", params.gripper_close_rad, 1.0))
    ...
return EpisodePlan(seed, scene, params, tuple(segs), perturbations)
```
</details>

## 9. 完了条件（DoD）

- [ ] `pytest tests/planning` が通る（IK→FKの往復、到達できない点で None）
- [ ] FKとROSのTFの一致を確認した（誤差 < 1mm）
- [ ] 固定配置で、計画→実行→32章の判定で `success` が出る
- [ ] ランダム化した20エピソードの試運転で、生成の成功率と1エピソードの所要時間が分かっている
- [ ] 200エピソード生成にかかる時間の見積もりがある（D4の夜に間に合うか判断できる）
- [ ] キューブ位置の散布図で、範囲全体がカバーされている

## 10. 理解度チェック・発展課題

- 5自由度の腕で、キューブを斜め45°から掴むことはできるか。できるならどんな条件か。
- 立て直しの注入で記録される「わざとずれた動き」も、模倣学習の教師データになってしまう。これは問題か。どう緩和できるか。
- 発展：MoveItを使わずに、自作IK＋関節空間の補間（最小躍度軌道など）だけでデモを作ると、何が得られ、何が失われるか。
