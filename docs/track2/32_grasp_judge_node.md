# track2/32. 把持判定：純粋関数のコアとROSノード、DetachableJoint連携

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**33個**（`src/evla/grasp/` の `geometry.py` 11、`judge.py` 11、`events.py` 3、`ros2_ws/src/evla_nodes/` の `evla_nodes/ros_utils.py` 2、`evla_nodes/grasp_judge_node.py` 5、`setup.py` 1）
> - ゴール：`pytest tests/grasp` が通り、Gazebo で「正しい位置で閉じると attach、ずれた位置では `grasp_candidate` だけ」が確かめられる
> - 進め方：Step 1〜5 は**ROS なし**（`.venv-bench` と pytest）でできる。Step 6〜7 は ROS 側のシェルで行う。穴の数は多いが、1つ1つは短い

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

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章の設計の要は「**判定は ROS を知らない純粋関数、ノードは I/O だけ**」です。だから Step 5 までは、Gazebo なしで境界値まで確かめられます。

### Step 0：準備（5分）

```bash
source .venv-bench/bin/activate
grep -n "【穴" src/evla/grasp/*.py ros2_ws/src/evla_nodes/evla_nodes/{ros_utils,grasp_judge_node}.py ros2_ws/src/evla_nodes/setup.py
pytest tests/grasp -q          # まだ失敗する
```

- [ ] 穴33個が表示され、テストが失敗する
- [ ] 2.1節の記法（`T_w_tcp`、`T_tcp_cube = inv(T_w_tcp) @ T_w_cube`）を紙に書き写しておく

### Step 1：幾何の関数 ― `geometry.py`（穴①〜⑪、2時間）

**読む**：4節の `geometry.py` の仕様と、`tests/grasp/test_geometry.py`（パラメータ化されたヨーの境界値の表がある）。

**埋める**（①〜⑤：同次変換）

| 穴 | ねらい |
| --- | --- |
| ①② | 4×4 の同次変換の、どこが回転でどこが並進か。NumPy のスライスで左辺に書く |
| ③④ | 逆変換。回転は直交行列なので逆行列＝転置。並進は「回転の逆を掛けて符号を反転」。`np.linalg.inv` を使わない理由（速さと数値の安定）も docstring で確認する |
| ⑤ | docstring の式そのもの（`@` は行列の積） |

**埋める**（⑥〜⑪：角度と開き幅）

| 穴 | ねらい |
| --- | --- |
| ⑥ | 角度の折り返し。`+half` してから剰余をとり、`-half` で戻す（立方体は90°回しても同じ形なので周期90°） |
| ⑦ | キューブの3つの軸のうち、接近軸方向の成分の**絶対値**が最も小さいもの＝最も水平な軸 |
| ⑧ | 平面上の角度。`atan2` の引数は (y, x) の順 |
| ⑨ | 回転行列の**列**は、その座標系の軸を world で表したもの |
| ⑩ | `acos` の引数を [-1, 1] に収める（丸め誤差で 1.0000001 になると NaN） |
| ⑪ | 表の線形補間（NumPy の1次元補間の関数） |

**動かす**

```bash
pytest tests/grasp/test_geometry.py -v
python -c "
import numpy as np
from evla.grasp.geometry import fold_deg, pose_to_matrix, relative_pose
print([round(fold_deg(d), 3) for d in (0, 14.9, 44, 46, 90, -91)])
T = pose_to_matrix([0.1, 0.2, 0.3], [0, 0, np.sin(0.3), np.cos(0.3)])
print(np.allclose(relative_pose(T, T), np.eye(4)))
"
```

**確かめる**

- [ ] すべてのテストが `PASSED`（`test_yaw_is_90deg_periodic`：ランダムな角度で「90°足しても同じ」というプロパティテスト）
- [ ] `[0.0, 14.9, 44.0, -44.0, 0.0, -1.0]` と `True`

### Step 2：開き幅の表を作る（`TODO(you)`、1時間）

`config/opening_table.csv` は仮の値（2行）です。31章で爪の先端のフレームを追加しておけば、関節6を少しずつ動かして TF で先端間の距離を測り、`(q, opening)` の表を作れます。

```bash
source scripts/ros_env.sh
ros2 launch evla_sim sim.launch.py world:=/tmp/evla_world_0.sdf
# 別の端末で：関節6を q に動かしてから、先端間の距離を読む（q を 0.1 rad 刻みで繰り返す）
ros2 run tf2_ros tf2_echo <固定側の先端> <可動側の先端>
```

- [ ] `ros2_ws/src/evla_sim/config/opening_table.csv` に10行程度の `q_rad,opening_m` がある（q の昇順）
- [ ] `grasp.yaml` の `opening_table_csv` に、その表のパスを書いた

### Step 3：把持と成功の条件 ― `judge.py` の穴①〜⑦（1時間）

**読む**：計画書の把持条件1〜4と、4節の `check_grasp` / `cube_in_box`。L3のヒント（8節）の疑似コードと、実際のコードの違い（`np.abs` を先に計算している）も見る。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | Step 1 の `relative_pose` を使う。どちらが「見る側」か（TCP から見たキューブ） |
| ② | 開閉方向の許容値は**固定値ではなく**、そのときの開き幅で決まる（隙間の半分）。負にならないようにする |
| ③ | ②の関数を使う。`off` は絶対値済み、`OPEN_AXIS` の成分を見る |
| ④ | ヨーは符号付きなので絶対値で比べる |
| ⑤ | `failed` が空なら成功 |
| ⑥ | 箱の中心からのずれ（xy の2成分）が、軸ごとに半幅以内。`np.all` の中身 |
| ⑦ | z が縁の高さより低い（箱の縁に乗っているだけ、を除く） |

**動かす**

```bash
pytest tests/grasp/test_judge.py -v -k "not state_machine"
```

- [ ] 5つのテスト（パラメータ化を含めると7件）が `PASSED`。境界値 `tol ± ε` で ok / ng が切り替わる

### Step 4：状態機械 ― `judge.py` の穴⑧〜⑪（1.5時間）

**読む**：`GraspStateMachine` の docstring の状態遷移。**紙に状態遷移図を描く**（FREE → ATTACH_PENDING → ATTACHED → DETACH_PENDING → RELEASED）。各矢印に「何が起きたら」を書く。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ⑧ | 「立ち上がり」の検出。前回は閉じていなくて、今回閉じた（`self.m.prev_close` を使う）。これで、30Hz で同じ候補を出し続けない |
| ⑨ | attach を送ったら、gz から返事が来るまで待つ状態に移る（連打しないため。7節の落とし穴） |
| ⑩ | gz が「外れた」と返してきた（`gz_attached` の値） |
| ⑪ | 静止のカウント。動いていなければ +1、動いたら 0 に戻す（三項演算子で1行） |

**動かす**

```bash
pytest tests/grasp/test_judge.py -v -k state_machine
```

- [ ] `test_state_machine_happy_path` と `test_state_machine_rejected_candidate_logged_once` が `PASSED`
- [ ] 紙の状態遷移図と、テストの `update` の呼び出し順を見比べて、各 `assert` がどの矢印を確かめているか言える

### Step 5：イベントの記録 ― `events.py`（穴①〜③、30分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `json` は NumPy の配列を知らない。Python のリストに変換するメソッド |
| ② | `json.dumps` が知らない型に出会ったときに呼ぶ関数を渡す引数（同じファイルの `_default`） |
| ③ | 最も多く出てきた要素。`max(..., key=...)` か `collections.Counter` |

**動かす**

```bash
pytest tests/grasp -v
```

- [ ] `tests/grasp` の**すべて**のテストが `PASSED`（ここまで ROS なし）

### Step 6：ROS ノード ― `ros_utils.py`（穴①②）、`grasp_judge_node.py`（穴①〜⑤）、`setup.py`（穴①）（2時間）

**読む**：tf2 のチュートリアル（2節のリンク）。ノードの各コールバックは「値を覚えるだけ」で、判定は `on_timer` で `self.sm.update` を呼ぶだけであることを確認する。

**埋める**

| 穴 | ねらい |
| --- | --- |
| `ros_utils` ① | ROS の時刻は `sec` と `nanosec` の2つ。整数のナノ秒1つにまとめる |
| `ros_utils` ② | z 軸まわりの回転 θ のクォータニオンは `(0, 0, sin(θ/2), cos(θ/2))`。すぐ下の `w` の行と対になる |
| `node` ① | tf2 の定番の2行。`Buffer` を作り、それに `/tf` を貯めるオブジェクトを作る（引数は buffer と node） |
| `node` ②③ | rclpy の購読と配信を作る `Node` のメソッド。すぐ下の行に同じ形の例がある |
| `node` ④ | `Buffer` から「world から見た tcp」を引く。引数の順（target, source）と、`Time()`＝最新 |
| `node` ⑤ | ノードのコールバックを処理し続ける rclpy の関数 |
| `setup.py` ① | エントリポイントは `"モジュール:関数"`。下の3行と同じ形 |

**動かす**

```bash
source scripts/ros_env.sh
cd ros2_ws && colcon build --symlink-install --packages-select evla_nodes && cd .. && source scripts/ros_env.sh
ros2 launch evla_sim sim.launch.py world:=/tmp/evla_world_0.sdf      # judge ノードも起動する
# 別の端末で
ros2 node info /grasp_judge
ros2 topic echo /grasp/event
```

- [ ] `ros2 node info` で、4節の購読・配信・サービス（`~/reset`）が揃っている
- [ ] 起動ログに `TF 待ち` が出続けない（出続けるなら `tcp` の TF が無い → 31章の Step 5）

### Step 7：Gazebo での結合テスト（1時間）

MoveIt（または `ros2 topic pub` で関節指令）で、グリッパをキューブの位置まで動かして閉じます。

- [ ] **正しい位置**で閉じると、`/grasp/event` に `grasp_candidate`（`decision.ok: true`）→ `attach_sent` → `attached` が出て、`/grasp/state` が `attached` になる
- [ ] **ずれた位置**で閉じると、`grasp_candidate`（`decision.failed` に条件名）だけが出て、attach されない
- [ ] 箱の上で開くと `released`、静止すると `success` が出る
- [ ] `/tmp/grasp_events.jsonl`（`grasp.yaml` の `event_log`）にイベントが1行1件で残っている
- [ ] `grasp.yaml` の `TODO(you)`（閾値）に、根拠をコメントで書いた（6節）

### Step 8：仕上げ（10分）

- [ ] Python の5ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/grasp ros2_ws/src/evla_nodes` を通す（他の章の穴が残るファイルは除く）
- [ ] `pytest tests/grasp` がすべて通る
- [ ] コミットする

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
