# track2/32. 把持判定：純粋関数のコアとROSノード、DetachableJoint連携

## 1. 目的と計画書の対応

- 計画書の把持条件1〜4、解放条件、タスク成功条件を、**ROSに依存しない純粋関数**として実装し、pytestで境界値を検証する。
- それをROSノードで包み、DetachableJointのattach/detachを操作する。
- 判定に使った幾何量を**すべてログに残し**、後から閾値を変えて再判定（感度分析、38章）できるようにする。
- 計画書の対応箇所：「把持判定仕様とDetachableJoint実装」全体、評価指標「把持精度」、日程D3

## 2. 前提知識・事前に読む資料

- 剛体変換（同次変換行列、`T_a_b` の記法、逆変換、合成）
- 回転の表現（回転行列、クォータニオン、オイラー角）と、角度の折り返し（wrap）
- tf2（Python）：<https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html>
- 31章で決めたTCP座標系の軸の規約（例：Z＝接近方向、Y＝爪の開閉方向）

### 2.1 記法

- `T_w_tcp`：world座標系から見たTCPの姿勢（4×4）。`T_w_cube` も同様。
- `T_tcp_cube = inv(T_w_tcp) @ T_w_cube`：TCPから見たキューブ。位置 `p = T_tcp_cube[:3, 3]` がずれ量になる。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/grasp/geometry.py` | 変換、ヨーのずれ、接近角の計算（NumPyのみ） |
| `src/evla/grasp/judge.py` | `GraspThresholds`、`check_grasp`、`check_release`、`check_success`、状態機械 |
| `src/evla/grasp/events.py` | イベントのデータ型とJSON変換 |
| `tests/grasp/test_geometry.py`、`test_judge.py` | 境界値・不変条件のテスト |
| `ros2_ws/src/evla_nodes/evla_nodes/grasp_judge_node.py` | ROSノード（I/Oだけ） |
| `ros2_ws/src/evla_sim/config/grasp.yaml` | 閾値（ノードのパラメータ） |

**`evla.grasp` はNumPy 1.26でも動くコードにする**（ROSのプロセスから import するため。[01](../01_repository_design.md)）。

## 4. インタフェース仕様

```python
# src/evla/grasp/geometry.py
def pose_to_matrix(pos: ArrayLike, quat_xyzw: ArrayLike) -> NDArray[np.float64]: ...
def relative_pose(T_w_a: NDArray, T_w_b: NDArray) -> NDArray:
    """T_a_b = inv(T_w_a) @ T_w_b。逆行列は回転の転置で計算する（np.linalg.inv を使わない）。"""
def yaw_misalignment_deg(R_tcp_cube: NDArray, approach_axis: int = 2, open_axis: int = 1) -> float:
    """キューブの面の向きと爪の開閉軸のずれ[deg]を、[-45, 45) に折り返して返す（立方体は90°周期）。
    接近軸まわりの回転だけを見る。"""
def approach_angle_deg(R_w_tcp: NDArray, approach_axis: int = 2) -> float:
    """TCPの接近軸と、鉛直下向き(0,0,-1)のなす角[deg]。"""
def jaw_opening_m(q_gripper_rad: float, table: "OpeningTable") -> float:
    """グリッパ角から爪の開き幅を求める（事前に作った表を線形補間）。"""
```

```python
# src/evla/grasp/judge.py
@dataclass(frozen=True)
class GraspThresholds:
    cube_size_m: float
    vertical_tol_m: float            # TCPの接近軸方向
    forward_tol_m: float             # もう1つの軸
    yaw_tol_deg: float = 15.0
    approach_tol_deg: float | None = 30.0    # None で条件4を使わない
    gripper_closed_rad: float = 0.0  # これ以下を「閉じている」とみなす（TODO: 31章の符号の確認に従う）
    release_open_rad: float = 0.5    # これ以上開いたら解放
    # 開閉方向の許容値は、そのときの開き幅から計算する： (opening - cube_size) / 2

@dataclass(frozen=True)
class GraspGeometry:                 # ログに残す（再判定の材料）
    offset_m: tuple[float, float, float]    # TCP座標系でのキューブ中心のずれ (x, y, z)
    yaw_err_deg: float
    approach_deg: float
    opening_m: float
    q_gripper_rad: float
    close_commanded: bool

@dataclass(frozen=True)
class GraspDecision:
    ok: bool
    failed: tuple[str, ...]          # 満たさなかった条件の名前（"open_axis", "vertical", "forward", "yaw", "approach", "not_closing"）

def compute_geometry(T_w_tcp, T_w_cube, q_gripper_rad, close_commanded, opening_table) -> GraspGeometry: ...
def check_grasp(g: GraspGeometry, th: GraspThresholds) -> GraspDecision: ...
def check_release(q_gripper_rad: float, th: GraspThresholds) -> bool: ...

@dataclass(frozen=True)
class SuccessCriteria:
    box_inner_xy_m: tuple[float, float]
    box_center_xy_m: tuple[float, float]
    rim_z_m: float
    margin_m: float = 0.0            # 壁からの余裕（キューブの半分など）
    still_frames: int = 15           # N：連続して静止しているフレーム数（30Hzで0.5秒）
    still_tol_m: float = 0.001       # 1フレームあたりの移動量の上限

def cube_in_box(p_w_cube: ArrayLike, sc: SuccessCriteria) -> bool: ...

class GraspState(Enum):
    FREE = auto(); ATTACH_PENDING = auto(); ATTACHED = auto(); DETACH_PENDING = auto(); RELEASED = auto()

class GraspStateMachine:
    """入力（幾何量、gzからの接続状態）を受けて、出すべき指令（attach/detach）とイベントを返す。ROSを知らない。"""
    def __init__(self, th: GraspThresholds, sc: SuccessCriteria) -> None: ...
    def reset(self) -> None: ...
    def update(self, t_sim: float, g: GraspGeometry, p_w_cube: NDArray, gz_attached: bool) -> "StepOutput": ...

@dataclass(frozen=True)
class StepOutput:
    command: Literal["attach", "detach"] | None
    events: tuple[dict[str, Any], ...]       # "grasp_candidate", "attached", "released", "success" など
    success: bool | None                     # 確定したら True / False
```

**イベントの種類**（`/grasp/event` にJSONで出し、JSONLにも保存する）

| type | いつ | 主な内容 |
| --- | --- | --- |
| `grasp_candidate` | 閉じ指令が出た（または閉じ角に達した）各判定時刻。**判定結果によらず出す** | `GraspGeometry` 全部、`GraspDecision`、使った閾値 |
| `attach_sent` / `attached` | attachを送った / gzが `attached` を返した | 確定時の相対姿勢（`T_tcp_cube`） |
| `released` | 解放条件を満たし、`detached` を確認した | キューブの位置 |
| `success` / `failure` | 静止を確認した / タイムアウト | 判定に使った値 |

## 5. 実装手順

1. **幾何の関数とテスト（2時間）**
   - `relative_pose`：ランダムな姿勢で `relative_pose(T, T) == I` になること、合成の結合則が成り立つことをテストする。
   - `yaw_misalignment_deg`：0°→0、14.9°→14.9、44°→44、46°→−44、90°→0、−91°→−1、をテストする。「90°足しても結果が同じ」ことを、ランダムな角度で多数回確かめる（プロパティテスト）。
2. **開き幅の表（1時間）**：31章で爪の先端のフレームを追加しておけば、関節6を少しずつ動かしてTFで先端間の距離を測り、`(q, opening)` の表を作れる。`config/opening_table.csv` に保存する。
3. **`check_grasp`（1時間）**：条件1（軸ごとの箱、開閉方向は開き幅から計算）、条件2、条件3、条件4（任意）。テストで、各条件の境界±εを確かめる。
4. **`cube_in_box` と静止判定（1時間）**：箱の内側、縁の高さ未満、Nフレーム連続の静止。
5. **状態機械（2時間）**：`FREE →（条件成立で attach を指令）→ ATTACH_PENDING →（gzが attached）→ ATTACHED →（解放条件で detach を指令）→ DETACH_PENDING →（gzが detached）→ RELEASED →（成功/失敗の判定）`。テストでは、gzの状態を手で与えて遷移を確かめる。
6. **ROSノード（2時間）**
   - 購読：`/joint_states`（関節6の角度）、`/model/cube/pose`、`/grasp/state`、グリッパへの指令（`/gripper_controller/joint_trajectory` か、コントローラの `controller_state` の reference）
   - TF：`tf2_ros.Buffer` と `TransformListener` で `world → tcp` を取得する
   - 30Hz（sim時刻）のタイマーで、幾何量を計算 → `state_machine.update` → 指令をpublish → イベントをpublish・保存
   - サービス `~/reset`（`std_srvs/Trigger`）：エピソード開始時に状態機械をリセットし、detachを送る
7. **結合テスト（1時間）**：Gazeboで、手動（またはMoveIt）でキューブの位置までグリッパを動かして閉じ、`attached` になることを確認する。ずれた位置で閉じたら、`grasp_candidate` は出るが attach されないことも確認する。

## 6. ベストプラクティス

- **判定と記録を分ける**：判定が通らなかった場合も、`grasp_candidate` に幾何量を全部残す。これが閾値の感度分析（38章）の材料になる。
- **閾値は幾何から決め、根拠を残す**：開閉方向の許容値は `(開き幅 − キューブ幅) / 2` で自動的に決まる。上下・前後は、爪の長さやキューブの大きさから決め、`grasp.yaml` にコメントで根拠を書く。データ生成と評価で**同じファイル**を使う。
- **ROSノードは薄く**：ノードの中に `if` の連鎖で判定を書かない。`GraspStateMachine.update` を呼ぶだけにする。
- **attachの確定を待つ**：bridge経由なので、指令から接続まで遅れがある。gzが `attached` を返した時点の相対姿勢を「把持時のずれ」として記録する（計画書の注意点の表のとおり）。
- **単位と座標系をイベントに書く**：`"offset_m"`、`"frame": "tcp"` のように、読み手が迷わないようにする。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| ヨーのずれが ±90° 付近で暴れる | 折り返しを ±180° でしている | 90°周期で [-45, 45) に折り返す |
| TCPとキューブの位置が常に数cmずれる | TFの `world` とGazeboのworldの原点が違う（ロボットのspawn位置） | ロボットはworld原点にspawnする。またはbaseの姿勢を合成する |
| `lookup_transform` が例外になる | sim時刻とTFのタイムスタンプの不整合、まだTFが届いていない | `Time()`（最新）で取得する。`use_sim_time:=true`。例外は捕捉してスキップ（ログは出す） |
| attachを連打してしまう | `ATTACH_PENDING` の状態がない | 状態機械で、確定待ちの間は再送しない |
| 解放直後に再び掴む | 解放後も把持条件を満たしている | `RELEASED` からは把持に戻らない（エピソード中1回限り）。再把持を許すかは【判断】として決め、記録する |
| 成功判定が出ない | 静止判定が厳しすぎる（物理の微小な振動） | `still_tol_m` を実際の振動の大きさから決める |

## 8. ヒント

<details><summary>L1：ヨーのずれをどう定義するか</summary>

TCPから見たキューブの回転のうち、接近軸まわりの成分だけを取り出す。キューブの1つの面の法線を接近軸に垂直な平面に射影し、爪の開閉軸とのなす角を求める。立方体なので、どの面を使っても90°の倍数しか違わない → 90°周期で折り返す。
</details>

<details><summary>L2：使う関数</summary>

`scipy` は使わずNumPyだけで書けるが、テストでは `scipy.spatial.transform.Rotation` を答え合わせに使ってよい。`np.arctan2`、`np.mod(angle + 45, 90) - 45`、`np.arccos(np.clip(dot, -1, 1))`。ROS側：`tf2_ros.Buffer.lookup_transform("world", "tcp", rclpy.time.Time())`、`json.dumps(event)`。
</details>

<details><summary>L3：check_grasp の疑似コード</summary>

```
failed = []
x, y, z = g.offset_m                         # 例：y=開閉軸、z=接近軸（31章の規約に合わせる）
open_tol = max(0.0, (g.opening_m - th.cube_size_m) / 2)
if abs(y) > open_tol: failed.append("open_axis")
if abs(z) > th.vertical_tol_m: failed.append("vertical")
if abs(x) > th.forward_tol_m: failed.append("forward")
if abs(g.yaw_err_deg) > th.yaw_tol_deg: failed.append("yaw")
if th.approach_tol_deg is not None and g.approach_deg > th.approach_tol_deg: failed.append("approach")
if not (g.close_commanded or g.q_gripper_rad <= th.gripper_closed_rad): failed.append("not_closing")
return GraspDecision(ok=not failed, failed=tuple(failed))
```

注意：開き幅は「閉じ始める直前」の値を使うのか、「現在」の値を使うのかで結果が変わる。どちらにしたかを決めて記録する。
</details>

## 9. 完了条件（DoD）

- [ ] `pytest tests/grasp` が通り、5節のヨーの境界値とプロパティテストを含む
- [ ] 開き幅の表（`opening_table.csv`）がある
- [ ] Gazeboで、正しい位置で閉じると `attached`、ずれた位置では attach されずに `grasp_candidate` だけが出る
- [ ] 箱に入れて静止すると `success` イベントが出る
- [ ] `grasp.yaml` に閾値と根拠のコメントがある
- [ ] イベントのJSONLが `results/` に保存される

## 10. 理解度チェック・発展課題

- 球ではなく軸ごとの箱で判定する理由は何か。開閉方向だけ許容値が開き幅に依存するのはなぜか。
- 量子化したモデルで成功率が下がったとき、`grasp_candidate` のログから「どの条件で落ちたか」をどう集計するか。
- 発展：判定が通らなかった `grasp_candidate` について、もし閾値が緩ければ成功していたかを、後から知ることはできるか（38章の「片側の推定」を先に考えてみよ）。
