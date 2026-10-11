# track2/33. MoveItでのpick&placeデモ自動生成とランダム化

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
| `evla_nodes/demo_generator.py` | MoveItで各区間を計画し、`EpisodePlan` を作る（実行はしない） |
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

## 5. 実装手順

1. **固定配置で1回成功させる（3時間）**：まずランダム化なしで、キューブを決まった位置に置き、区間1〜9を手書きの目標で計画・実行して、32章の判定で `success` が出るところまで行く。この段階では、MoveItの `execute` をそのまま使ってよい（実時間で動かす）。
2. **IKの問題を解決する（2〜3時間）**：次のどれかを選び、`docs/decisions.md` に理由を書く。
   - (a) **幾何的なIKを自作する**（学習効果が最も高い）。IKで目標の関節角を求め、MoveItには**関節空間の目標**として渡す（衝突回避と時間パラメータ化はMoveItに任せる）。
   - (b) `pick_ik` を入れ、位置と姿勢の重みを調整する。
   - (c) KDLのまま、真上からの把持姿勢（2.1で到達可能と分かっている姿勢）だけを目標にする。
3. **計画と実行の分離（2時間）**：`demo_generator` が `EpisodePlan` を作って返すだけにする。実行は、34章の `episode_runner` ができるまでは、仮の実行関数（FollowJointTrajectoryアクションに送る）で行う。
4. **ランダム化（2時間）**：`sample_scene`、`sample_params`。エピソードごとのシードは `base_seed + episode_index` にする。
5. **立て直しの注入（2時間）**：確率 p（例：0.3）で `grasp_offset_m` を1〜2cm入れる。descendの後で「ずれた位置」から「正しい位置」へ小さく動く区間を挟む。注入したかどうかを `perturbations` に記録する。
6. **頑健化（2時間）**：6節の項目を実装する。
7. **試運転（1時間）**：20エピソードを回し、成功率と1エピソードあたりの所要時間を測る。200エピソードにかかる時間を見積もる。

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
