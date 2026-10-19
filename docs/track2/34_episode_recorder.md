# track2/34. ロックステップ記録とLeRobotDataset v3への変換

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**36個**（`src/evla/` の `common/stamped_buffer.py` 3、`dataset/conventions.py` 6、`dataset/raw_episode.py` 6、`dataset/quality.py` 5、`dataset/to_lerobot.py` 4、`ros2_ws/src/evla_nodes/evla_nodes/` の `lockstep.py` 4、`sim_io.py` 3、`episode_runner.py` 5）
> - ゴール：`pytest tests/common/test_stamped_buffer.py tests/dataset` が通り、ロックステップで記録した1エピソードを LeRobotDataset に変換して読み込める。D4 の夜に200エピソード以上を生成する
> - 進め方：ROS なしの部品（Step 1 前半・Step 2・Step 3）を pytest で固めてから、ROS 側（Step 1 後半・Step 4）に進む

## 1. 目的と計画書の対応

- Gazeboを**ロックステップ**（pauseして1/30秒ずつstep）で進めながら、画像2枚・関節状態・指令を、同じsim時刻で揃えて記録する。
- ROS側では「生エピソード」を書き出すだけにし、別のvenvで **LeRobotDataset v3.0** に変換する。
- 公開SO-101データと同じ関節名・順序にし、追加メタデータを `meta/` に保存する。
- 計画書の対応箇所：「データセット仕様」「追加メタデータ」、日程D3（記録ノード）、D4（デモ生成）、D5（データ品質確認）、成果物「Gazebo SO-101 pick&placeデータセット」

## 2. 前提知識・事前に読む資料

- LeRobotDataset v3：<https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3>
- ソース（読解用）：`src/lerobot/datasets/lerobot_dataset.py` の `create`、`add_frame`、`save_episode`、`finalize`
- 公開データの `meta/info.json`：<https://huggingface.co/datasets/geonmin-kim/merged_all7/raw/main/meta/info.json>
- `ros_gz_interfaces/srv/ControlWorld`（`gz.msgs.WorldControl`：`pause`、`step`、`multi_step`）

### 2.1 調査で確認した事実

**公開SO-101データ（merged_all7、v3.0）の形式**

| キー | 型・形 | 名前 |
| --- | --- | --- |
| `action` | float32 [6] | `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` |
| `observation.state` | float32 [6] | 同上 |
| `observation.images.top` | video [480, 640, 3] | av1、30fps |
| `observation.images.wrist` | video [480, 640, 3] | av1、30fps |
| `robot_type` | `so_follower` | |
| 状態の範囲（stats.json） | 最小 [−106.7, −101.7, −96.3, −3.6, −167.6, 0.9]、最大 [79.0, 79.7, 97.1, 102.7, −22.5, 70.2] | 先頭5つは度、**グリッパは0〜100の正規化値** |

**`LeRobotDataset.create` の主な引数（main）**：`repo_id, fps, features, root=None, robot_type=None, use_videos=True, image_writer_processes=0, image_writer_threads=0, video_backend=None, batch_encoding_size=1, ..., streaming_encoding=False`。`add_frame(frame)` の `frame` には **`"task"` キーが必須**。最後に **`finalize()` を必ず呼ぶ**（呼ばないとParquetのフッタが書かれず、データセットが壊れる）。

**【計画書との差分】グリッパの単位**：計画書は「度単位で統一」としていますが、公開データのグリッパ列は0〜100です。**【判断】** このデータセットでは計画書どおり**グリッパも度**で保存し、0〜100に変換する関数と、その対応（閉＝0、開＝100にする角度）をメタデータに残します。実機データとの共同学習のときに、その関数で変換します。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `ros2_ws/src/evla_nodes/evla_nodes/episode_runner.py` | 時計の所有者。リセット → 区間の実行 → フレームの記録 → 成否の判定 → 保存 |
| `ros2_ws/src/evla_nodes/evla_nodes/lockstep.py` | `LockstepClock`：pause / multi_step と、指定sim時刻のメッセージが揃うのを待つ処理 |
| `src/evla/common/stamped_buffer.py` | 時刻つきメッセージのリングバッファ（ROS 非依存。`LockstepClock` が使う） |
| `ros2_ws/src/evla_nodes/evla_nodes/sim_io.py` | `SimIO`：Gazebo とのやり取り（リセット、指令の送信、1フレーム進めて観測）。`eval_runner`（38章）と共有する |
| `src/evla/dataset/raw_episode.py` | 生エピソードの書き込み・読み込み（NumPy 1.26でも動く） |
| `src/evla/dataset/conventions.py` | 関節名の対応、rad→度、グリッパの0〜100への変換 |
| `src/evla/dataset/to_lerobot.py` | 生エピソード群 → LeRobotDataset v3（`.venv-bench` で実行） |
| `src/evla/dataset/quality.py` | 品質チェック（フレーム落ち、黒画像、統計、分布の図） |
| `scripts/convert_raw_to_lerobot.py`、`scripts/check_dataset.py` | CLI |
| `data/raw/<name>/ep_<seed>/` | 生エピソード |
| `data/lerobot/<name>/` | 変換後のデータセット |

## 4. インタフェース仕様

### 4.1 生エピソードの形式

```
data/raw/so101_pickplace_v1/ep_000123/
├── top/000000.png ...            # 480×640 RGB（PNG、圧縮レベル1：速さ優先、可逆）
├── wrist/000000.png ...
├── arrays.npz                    # t_sim[s] (T,), state_rad (T,6), action_rad (T,6), cube_pose_w (T,7)
├── meta.json                     # scene, params, perturbations, success, thresholds, world_params, versions
└── events.jsonl                  # 32章のイベント（grasp_candidate など）
```

（容量の目安：PNGで1枚100〜300KBとして、200エピソード×300フレーム×2枚で約12〜36GB。変換後に削除してよいが、**閾値の再判定（38章）に必要な `arrays.npz` と `events.jsonl` は残す**）

### 4.2 ロックステップ

```python
# evla_nodes/lockstep.py
class LockstepClock:
    """worldの時間を進める唯一の主体。"""
    def __init__(self, node: Node, world: str, physics_dt: float, frame_dt: float) -> None:
        """frame_dt / physics_dt が整数でなければ ValueError（30章の4.4）。"""
    def pause(self) -> None: ...
    def step_frame(self) -> float:
        """multi_step = frame_dt / physics_dt を要求し、新しいsim時刻を返す。"""
    def wait_for(self, t_sim: float, topics: Sequence[str], timeout_s: float = 5.0) -> dict[str, Any]:
        """全トピックについて、stamp == t_sim（許容1e-6）のメッセージが揃うまで待って返す。
        タイムアウトしたら、どのトピックが欠けたかを含めて例外にする。"""
```

```python
# src/evla/dataset/conventions.py
URDF_TO_LEROBOT: dict[str, str] = {"1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
                                   "4": "wrist_flex", "5": "wrist_roll", "6": "gripper"}   # 【要確認】env/11 Step 5の結果で確定
LEROBOT_NAMES: tuple[str, ...] = tuple(f"{n}.pos" for n in
    ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
SIGN: dict[str, float]           # URDFとLeRobot実機の符号の違い（不明なら+1にして、メタデータに「URDF規約」と書く）

def joint_state_to_vector(names: Sequence[str], positions: Sequence[float]) -> NDArray[np.float64]:
    """ROSの JointState（順序は不定）→ LeRobotの順序の (6,) [rad]。名前で並べ替える。"""
def rad_to_deg(q: NDArray) -> NDArray: ...
def gripper_deg_to_0_100(g_deg: NDArray, closed_deg: float, open_deg: float) -> NDArray: ...
```

```python
# src/evla/dataset/to_lerobot.py
def build_features(image_hw: tuple[int, int] = (480, 640)) -> dict[str, dict]:
    """observation.state / action：{"dtype": "float32", "shape": (6,), "names": list(LEROBOT_NAMES)}
    observation.images.{top,wrist}：{"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}"""

def convert(raw_root: Path, out_root: Path, repo_id: str, task_text: str,
            only_success: bool = True, image_writer_threads: int = 4) -> None:
    """生エピソードを順に読み、add_frame → save_episode。最後に finalize()。
    追加メタデータを out_root/meta/evla/ に書く。"""
```

### 4.3 「行動」の定義

**【判断】** `action[t]` は、**時刻tにコントローラが追従しようとしている目標位置**（`/arm_controller/controller_state` の `reference.positions` と、グリッパのコントローラの同じ値）とする。

- 実機のLeRobotデータの `action` は、リーダーアームの位置＝フォロワーへの指令値なので、意味が対応する。
- `observation.state[t]` は `/joint_states` の実測位置。
- 「`action[t] = state[t+1]`」で代用しない。追従の遅れがあるため、両者は一致しない。

### 4.4 追加メタデータ（`meta/evla/`）

| ファイル | 内容 |
| --- | --- |
| `cameras.json` | 各カメラの内部パラメータ（fx, fy, cx, cy, 幅, 高さ, 歪みなし）、外部パラメータ（world座標系、Gazebo規約とOpenCV規約の両方）、見た目のランダム化による変動の範囲 |
| `robot.json` | URDF（xacro展開後）のsha256、lerobot_wsのコミット、関節の対応表、単位（度）、符号の規約、グリッパの0〜100への変換の定義 |
| `episodes_extra.jsonl` | エピソードごと：シード、キューブと箱の初期姿勢、色、初期関節角、軌道パラメータ、注入したノイズ、成否、把持時のずれ（軸ごとmm、ヨー°、接近角°）、world生成のパラメータ |
| `thresholds.json` | 把持判定・成功判定の閾値（`grasp.yaml` のコピー） |
| `generation.json` | 生成コードのgitコミット、日時、生成の成功率と失敗理由の内訳 |

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章の要は「**時計を持つのは1つのノードだけ**」（4.2節）。world を止めたまま少しずつ進め、そのたびに全トピックが揃うのを待ってから記録します。

### Step 0：準備（10分）

```bash
source .venv-bench/bin/activate
grep -n "【穴" src/evla/common/stamped_buffer.py src/evla/dataset/*.py ros2_ws/src/evla_nodes/evla_nodes/{lockstep,sim_io,episode_runner}.py
pytest tests/common/test_stamped_buffer.py tests/dataset -q     # まだ失敗する
```

- [ ] 穴36個が表示され、テストが失敗する

### Step 1：時刻つきバッファとロックステップ ― `stamped_buffer.py`（穴①〜③）と `lockstep.py`（穴①〜④）（2〜3時間）

**読む**：4.2節の `LockstepClock`。「pause → `multi_step` で1フレーム分進める → `/clock` が進むのを待つ → 各トピックの、その時刻のメッセージが揃うのを待つ」の流れを図にする。

**埋める**（ROS なし）

| 穴 | ねらい |
| --- | --- |
| `stamped_buffer` ① | 順序が前後して届いても、ソート済みを保つ挿入位置（`bisect` の関数） |
| `stamped_buffer` ② | 候補（前後の2つ）のうち、目的の時刻に近い方を選ぶキー |
| `stamped_buffer` ③ | `bisect_right` の位置の1つ手前が「t 以下で最新」。手前がなければ None |

```bash
pytest tests/common/test_stamped_buffer.py -v
```

- [ ] 4つのテストが `PASSED`

**埋める**（ROS 側）

| 穴 | ねらい |
| --- | --- |
| `lockstep` ① | 1フレーム（1/30秒）あたりの物理ステップ数（1/600秒）。直前で整数か確かめている |
| `lockstep` ② | 購読のコールバックで、`stamp_of(m)` をナノ秒にしてバッファに入れる（`stamp_to_ns` を使う） |
| `lockstep` ③ | サービスを**非同期で**呼ぶ。同期呼び出しはコールバックの中で使うとデッドロックする |
| `lockstep` ④ | `ControlWorld` の止めたまま進めるフィールドの値 |

**動かす**：Gazebo を起動し、100フレーム進めて時刻を確かめます。

```bash
source scripts/ros_env.sh
cd ros2_ws && colcon build --symlink-install --packages-select evla_nodes && cd .. && source scripts/ros_env.sh
ros2 launch evla_sim sim.launch.py world:=/tmp/evla_world_0.sdf
# 別の端末で
python3 - <<'EOF'
import threading, numpy as np, rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from evla_nodes.lockstep import LockstepClock

rclpy.init(); node = Node("lockstep_try")
ex = MultiThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
clk = LockstepClock(node, "evla", 1 / 600, 1 / 30)
topics = ["/camera/top/image", "/camera/wrist/image", "/joint_states"]
for tp in topics:
    clk.track(tp, Image if "camera" in tp else JointState)
clk.pause()
ts = []
for _ in range(100):
    t = clk.step_frame(); clk.wait_for(t, topics); ts.append(t)
d = np.diff(ts); print(len(ts), round(d.min(), 6), round(d.max(), 6))
rclpy.shutdown()
EOF
```

**確かめる**

- [ ] `100 0.033333 0.033333`（時刻の差がすべて 1/30 秒）
- [ ] 途中で `TimeoutError: ... 揃わない: [...]` が出たら、欠けたトピックを `ros2 topic hz` で調べる（7節の落とし穴）

### Step 2：関節名と単位の変換 ― `conventions.py`（穴①〜⑥、1時間）

**読む**：4.1節の `conventions.py`（関節の対応、rad → 度、グリッパの 0〜100）。**記録・変換・ポリシーノードで同じ関数を使う**ことが、37章で「腕が逆方向に動く」事故を防ぎます。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `names` と `positions` を組にして辞書にする（`dict(zip(...))`） |
| ② | LeRobot の名前 → URDF の名前（`LEROBOT_TO_URDF`）→ 角度 |
| ③ | 0〜1 に正規化するときの分母（開きと閉じの差） |
| ④ | `...`（Ellipsis）を使うと、`(6,)` でも `(T, 6)` でも最後の軸の6番目を指せる |
| ⑤ | 度 → rad（NumPy の関数） |
| ⑥ | ステップごとの辞書のリストから、1つの関節の値だけを集めた配列 |

**動かす**

```bash
pytest tests/dataset/test_conventions.py -v
```

- [ ] 5つのテストが `PASSED`（特に `test_roundtrip_rad_lerobot_rad`：rad → LeRobot → rad で元に戻る）
- [ ] `URDF_TO_LEROBOT` の【要確認】を、[env/11](../env/11_ros2_gazebo.md) Step 5 の対応表と照らし合わせた

### Step 3：生エピソードの書き出し ― `raw_episode.py`（穴①〜⑥、1時間）

**読む**：`raw_episode.py` の冒頭の docstring（生エピソードのディレクトリ構成）。画像を**別スレッド**で書く理由（`cv2.imwrite` の待ちで step が遅れないように）と、`meta.json` を**最後に**書く理由（`meta.json` があるエピソード＝完全なエピソード）を確認する。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `queue.Queue` から1つ取り出す（空なら来るまで待つ） |
| ② | OpenCV は BGR の順で保存する。RGB から並べ替える `cv2.COLOR_...` の定数 |
| ③ | 書き込みスレッドが全部書き終わって終了するまで待つ（スレッドのメソッド） |
| ④ | f-string の書式で6桁ゼロ埋め |
| ⑤ | JSON Lines：1行に1つの JSON と改行 |
| ⑥ | `meta.json` を持つディレクトリだけを探す（`Path` のワイルドカード検索） |

**動かす**

```bash
pytest tests/dataset/test_raw_episode.py -v -k write_and_read
```

- [ ] `test_write_and_read` が `PASSED`（`meta.json` の無い、途中で落ちたエピソードは一覧に出ない）

### Step 4：1エピソードを記録する ― `sim_io.py`（穴①〜③）と `episode_runner.py`（穴①〜⑤）（3時間）

**読む**：3節の `episode_runner` の役割（リセット → 区間の実行 → 記録 → 成否の判定 → 保存）、4.3節の「行動」の定義、`SimIO.reset_scene` / `step_and_observe` の中身。

**埋める**

| 穴 | ねらい |
| --- | --- |
| `sim_io` ① | ROS の画像は1行のバイト数が `step`（幅×3 より大きいことがある）。`reshape` で `(高さ, step)` にしてから幅で切る |
| `sim_io` ② | `ros_gz_interfaces/msg/Entity` の「モデル」を表す定数 |
| `sim_io` ③ | 記録する action は「コントローラが今追従しようとしている目標」（`controller_state` の reference） |
| `episode_runner` ① | 区間の長さをカバーするフレーム数（切り上げ） |
| `episode_runner` ② | エピソードごとのシードから乱数生成器を作る（33章のランダム化に渡す） |
| `episode_runner` ③ | Step 3 の `RawEpisodeWriter` に1フレーム書き足す |
| `episode_runner` ④ | イベントの中に `success` が1つでもあるか（`any`） |
| `episode_runner` ⑤ | サービスの応答待ちの間も購読を受け取れる executor |

**動かす**：33章 Step 5 の `configs/datagen/fixed.yaml`（1エピソード）で記録します。

```bash
ros2 run evla_nodes episode_runner --ros-args -p use_sim_time:=true \
  -p datagen_config:=configs/datagen/fixed.yaml -p out_dir:=data/raw/fixed_test -p planner:=interp
ls data/raw/fixed_test/ep_*/ && cat data/raw/fixed_test/ep_*/meta.json | head -20
```

**確かめる**

- [ ] `ep_<seed>/` に `top/`・`wrist/`（PNG）、配列の `.npz`、`events.jsonl`、`meta.json` がある
- [ ] PNG の枚数がフレーム数と同じ。何枚か開いて、色が正しい（赤いキューブが青くない＝②が効いている）
- [ ] `meta.json` の `success` が `true`（33章の Step 5 と同じ確認）

### Step 5：LeRobotDataset に変換する ― `to_lerobot.py`（穴①〜④、2時間）

**読む**：LeRobot の `LeRobotDataset` のソース（`create`、`add_frame`、`save_episode`、`finalize` の場所）。

```bash
source .venv-bench/bin/activate
python -c "import lerobot.datasets.lerobot_dataset as m, inspect; print(inspect.getsourcefile(m))"
```

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 画像を mp4 として保存する feature の dtype の名前（公開データの `info.json` を見る） |
| ② | 空のデータセットを新しく作るクラスメソッド |
| ③④ | 1フレーム追加 → エピソードとして確定、の2つのメソッド |

**動かす**

```bash
python scripts/convert_raw_to_lerobot.py --raw data/raw/fixed_test --out data/lerobot/fixed_test --repo_id local/fixed_test
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/fixed_test', root='data/lerobot/fixed_test')
x = ds[0]; print(len(ds), {k: tuple(v.shape) for k, v in x.items() if hasattr(v, 'shape')})
"
ls data/lerobot/fixed_test/meta/evla/
```

**確かめる**

- [ ] `observation.state`・`action` が `(6,)`、画像が `(3, 480, 640)`。キーと形が公開の SO-101 データと同じ
- [ ] `meta/evla/` に4.4節の5ファイルがある（足りないものは `TODO(you)`：`cameras.json` など）

### Step 6：品質チェック ― `quality.py`（穴①〜⑤、1.5時間）

**読む**：6節の「品質チェックの項目」。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 隣り合う時刻の差（NumPy の関数） |
| ② | すべての間隔が 1/30 秒か（誤差つきの比較を全要素で） |
| ③ | NaN でも inf でもない要素が True になる関数 |
| ④ | 画素の平均が閾値より暗い＝黒画像（レンダリングの失敗） |
| ⑤ | 最後の `last_n` 行の成功率（True/False の平均） |

**動かす**

```bash
pytest tests/dataset -v
python scripts/check_dataset.py --raw data/raw/fixed_test --out reports/dataset_quality_fixed.md
```

- [ ] `tests/dataset` のすべてのテストが `PASSED`（黒画像・フレーム落ちを検出できる）
- [ ] 品質レポートが書かれ、`fixed_test` に問題がない

### Step 7：一晩の生成と D5 の判定ゲート

- [ ] D4 の夜：`configs/datagen/default.yaml`（`n_episodes` を目標に合わせる）で `episode_runner` を回す。33章の試運転の見積もりで間に合う数にする
- [ ] D5 の朝：`python scripts/check_dataset.py` でレポートを出し、`gate_summary` の数値（成功数、直近の成功率）を [00_schedule](../00_schedule.md) の判定ゲートに当てはめる
- [ ] `python scripts/convert_raw_to_lerobot.py` で成功エピソードを変換し、35章に渡す

### Step 8：仕上げ（10分）

- [ ] 8ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/common src/evla/dataset ros2_ws/src/evla_nodes` を通す
- [ ] `pytest tests/common/test_stamped_buffer.py tests/dataset` がすべて通る
- [ ] コミットする

## 6. ベストプラクティス

- **品質チェックの項目**（D5ゲートの判断材料）
  - フレーム数＝エピソードの時間×30。`t_sim` の差がすべて1/30秒（フレーム落ちなし）
  - 画像の平均輝度が極端に低くない（黒画像の検出）。2つのカメラの画像が同一でない
  - `action` と `state` の差の分布（追従の遅れが妥当な範囲か）
  - 関節ごとの値の範囲が、URDFの可動範囲の内側
  - キューブの初期位置の散布図、エピソード長のヒストグラム
  - **LeRobotDataset として読めること**：`LeRobotDataset(repo_id, root=...)` で読み込み、`dataset[0]` のキーと形が期待どおり
  - 何エピソードかを動画で目視する
- **変換は何度でもやり直せるようにする**：生エピソードを残しておけば、単位や名前の規約を後から変えても、変換だけをやり直せる。
- **「完全なエピソード」の目印を最後に書く**：途中で落ちたエピソードを、変換で拾わないようにする。
- **タスク文**：最初は全エピソードで同じ文（例：`"Pick up the cube and put it in the box."`）にする。色を変えるなら、色を入れた文（`"Put the red cube in the box."`）にする選択肢もある。どちらにしたかを記録する。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `wait_for` がタイムアウトする | カメラが1フレーム遅れて出る。stampの丸め誤差 | 許容誤差を設ける。どのトピックが欠けたかをログに出す |
| ROSの中で `import lerobot` してNumPyのエラー | 依存の衝突（[01](../01_repository_design.md)） | 変換は別venvのスクリプトで行う |
| 変換したデータセットが読めない | `finalize()` を呼んでいない | 必ず呼ぶ（例外時も `try/finally`） |
| 関節の順序がエピソードによって違う | `JointState` の名前の順序は保証されない | 名前で並べ替える（`joint_state_to_vector`） |
| `add_frame` で形のエラー | 画像がCHWになっている、またはfloat | HWCのuint8で渡す（`features` の `shape` と一致させる） |
| ディスクが一杯になる | PNGの生データ | 変換後に画像を消す。圧縮レベルを上げる |
| 記録が極端に遅い | 画像の書き込みでstepが止まっている | 書き込みを別スレッドにする |

## 8. ヒント

<details><summary>L1：同じsim時刻のメッセージを揃えるには</summary>

トピックごとに「stamp → メッセージ」の辞書を持ち、stepした後に「目的の時刻のキーが、全トピックの辞書にあるか」を確かめる。なければ少し待ってもう一度確かめる。古いキーは捨てる。`message_filters` の近似同期は、ロックステップでは必要ない（時刻が厳密に一致するので）。
</details>

<details><summary>L2：使うAPI</summary>

`ros_gz_interfaces.srv.ControlWorld`（`req.world_control.pause = True`、`req.world_control.multi_step = 20`）、`rclpy.executors.MultiThreadedExecutor`、`ReentrantCallbackGroup`、`concurrent.futures.ThreadPoolExecutor`（画像の書き込み）、`cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])`、`np.savez_compressed`、LeRobot：`LeRobotDataset.create(...)`、`ds.add_frame({...,"task": text})`、`ds.save_episode()`、`ds.finalize()`
</details>

<details><summary>L3：腕の区間を記録する疑似コード</summary>

```
pub_arm.publish(segment.trajectory.to_msg(stamp=Time(0)))   # 0 = 受信時に即開始
t_end = clock.now() + segment.duration + settle
while clock.now() < t_end:
    t = clock.step_frame()
    msgs = clock.wait_for(t, ["/camera/top/image", "/camera/wrist/image", "/joint_states", "/arm_controller/controller_state"])
    writer.add(t, images=(msgs[top], msgs[wrist]), state=to_vec(msgs[js]), action=ref_vec(msgs[ctrl], gripper_ref))
```

（JointTrajectoryの `header.stamp` が0なら、コントローラは受信した時点で開始する。pause中に送っても、stepを始めるまで動かない）
</details>

## 9. 完了条件（DoD）

- [ ] 5節 Step 1 の確認（100フレームで時刻の差がすべて1/30秒、各トピックのstampが揃う）ができた
- [ ] 1エピソードを記録・変換し、`LeRobotDataset` として読み込めて、キーと形が公開データと同じ
- [ ] `meta/evla/` に4.4の5ファイルがある
- [ ] 成功エピソード100〜200のデータセットがあり、品質レポートに問題がない（または問題とその対処が書かれている）
- [ ] グリッパの単位の扱い（度で保存し、0〜100への変換を定義）がメタデータに書かれている

## 10. 理解度チェック・発展課題

- `action[t]` をコントローラの目標値にした場合と、`state[t+1]` にした場合で、学習される方策はどう違うか。
- ロックステップを使わず実時間で記録した場合、どんなずれが生じうるか。RTFが0.3のときに何が起きるか。
- 発展：記録時に `cube_pose_w` を残したのは何のためか。38章の閾値の再判定と、将来の「状態からの学習」の両面から考えよ。
