# track2/34. ロックステップ記録とLeRobotDataset v3への変換

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`src/evla/common/stamped_buffer.py`（3）、`src/evla/dataset/conventions.py`（6）、`src/evla/dataset/raw_episode.py`（6）、`src/evla/dataset/quality.py`（5）、`src/evla/dataset/to_lerobot.py`（4）、`ros2_ws/src/evla_nodes/evla_nodes/lockstep.py`（4）、`ros2_ws/src/evla_nodes/evla_nodes/sim_io.py`（3）、`ros2_ws/src/evla_nodes/evla_nodes/episode_runner.py`（5）
> - テスト：`pytest tests/common/test_stamped_buffer.py tests/dataset`
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

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

## 5. 実装手順

1. **`LockstepClock`（2〜3時間）**
   - worldをpauseで起動し（`-r` なし）、`ControlWorld` サービスで `multi_step` を要求する。
   - 各トピックの最新メッセージを、`stamp` をキーにした小さな辞書（リングバッファ）に溜めておき、`wait_for` で目的のsim時刻のものが揃うまで待つ。
   - サービスの応答待ちと購読のコールバックが同時に動く必要があるので、`MultiThreadedExecutor` と、Callback Groupの分離を使う（[02](../02_best_practices.md)）。
   - テスト：100フレーム進めて、`t_sim` の差がすべて1/30秒であること、2つの画像と関節状態の `stamp` が一致することを確かめる。
2. **episode_runner（3時間）**
   - リセット：pause → 判定ノードの `~/reset`（detach） → `set_pose` でキューブと箱を配置 → 腕を `q_init` へ（stepしながら。記録しない） → 数フレーム待って静止させる。
   - 実行：33章の `EpisodePlan` の区間を順に処理する。腕の区間は `/arm_controller/joint_trajectory` に軌道を送り、`step_frame` → `wait_for` → 記録、を軌道の時間が終わるまで繰り返す。
   - 終了：`wait` 区間の後、判定ノードのイベントから成否を決める。成功なら保存、失敗なら理由を数えて捨てる（イベントは失敗でも別の場所に保存する）。
3. **生エピソードの書き出し（1時間）**：画像は別スレッドで書く（`cv2.imwrite` の待ちで step が遅れないように）。エピソードの最後にすべての書き込みが終わったことを確認してから、`meta.json` を書く（`meta.json` があるエピソード＝完全なエピソード、とする）。
4. **変換（2時間）**：`.venv-bench` で `convert()` を実装する。`only_success=True`。
5. **品質チェック（2時間）**：6節の項目を `quality.py` に実装し、`scripts/check_dataset.py` でレポート（Markdown＋図）を出す。
6. **一晩の生成（D4の夜）**：33章の頑健化と合わせて、200エピソード以上（成功の目標100〜200）を回す。
7. **D5の朝**：品質レポートを見て、判定ゲート（[00_schedule](../00_schedule.md)）に臨む。

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

- [ ] `LockstepClock` のテスト（100フレームで時刻の差がすべて1/30秒、各トピックのstampが一致）が通る
- [ ] 1エピソードを記録・変換し、`LeRobotDataset` として読み込めて、キーと形が公開データと同じ
- [ ] `meta/evla/` に4.4の5ファイルがある
- [ ] 成功エピソード100〜200のデータセットがあり、品質レポートに問題がない（または問題とその対処が書かれている）
- [ ] グリッパの単位の扱い（度で保存し、0〜100への変換を定義）がメタデータに書かれている

## 10. 理解度チェック・発展課題

- `action[t]` をコントローラの目標値にした場合と、`state[t+1]` にした場合で、学習される方策はどう違うか。
- ロックステップを使わず実時間で記録した場合、どんなずれが生じうるか。RTFが0.3のときに何が起きるか。
- 発展：記録時に `cube_pose_w` を残したのは何のためか。38章の閾値の再判定と、将来の「状態からの学習」の両面から考えよ。
