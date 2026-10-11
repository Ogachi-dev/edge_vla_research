# track1/21. LIBEROの導入と環境ラッパ

## 1. 目的と計画書の対応

- WSL2上でLIBERO（MuJoCo）を動かし、**ランタイムに依存しない統一形式の観測**を返す薄いラッパを作る。
- モードAで使う「固定観測セット」（実際のLIBERO画像）を保存する。
- 計画書の対応箇所：トラック1、日程D2「LIBERO起動（MuJoCo描画確認）」、リスク「WSL2でのMuJoCo・Gazeboの描画」

## 2. 前提知識・事前に読む資料

- LeRobotのLIBEROドキュメント：<https://huggingface.co/docs/lerobot/en/libero>
- LeRobotのソース（読解用）：`src/lerobot/envs/libero.py`（`LiberoEnv`）、`src/lerobot/processor/env_processor.py`（`LiberoProcessorStep`）
- MuJoCoの描画バックエンド：<https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl>（`MUJOCO_GL=egl|glfw|osmesa`）

### 2.1 調査で確認した LIBERO の仕様（LeRobot版）

| 項目 | 値 |
| --- | --- |
| 環境クラス | `lerobot.envs.libero.LiberoEnv`（gymnasium） |
| 制御周期 | `control_freq=20`（Hz） |
| 行動 | 7次元、`Box(-1, 1)`：手先の6次元の差分＋グリッパ1次元。`control_mode="relative"`（既定） |
| カメラ | `agentview_image` → `image`、`robot0_eye_in_hand_image` → `image2`。既定は256×256、HWC uint8 |
| 画像の向き | **方策に渡す前に180°回転**（高さと幅の両方を反転）。`LiberoProcessorStep` が行う |
| 状態 | 8次元＝手先位置(3)＋手先姿勢の軸角(3)＋グリッパ関節位置(2)。クォータニオンは (x, y, z, w) の順 |
| 最大ステップ数 | spatial 280、object 280、goal 300、libero_10 520、libero_90 400 |
| 開始時の待機 | `num_steps_wait=10`（物体が落ち着くまで何もしない） |
| 初期状態 | `init_states=True` で、タスクごとに固定の初期状態ファイルを使う |
| 成功判定 | `info["is_success"]`（`env.check_success()`） |

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/sim/types.py` | `Observation`、`StepResult` などの共通データ型 |
| `src/evla/sim/libero_env.py` | `LiberoTaskEnv`：LeRobotの `LiberoEnv` をラップし、`Observation` を返す |
| `scripts/libero_smoke.py` | ランダム行動で1エピソード回し、動画を保存する |
| `scripts/dump_fixed_obs.py` | 固定観測セットを `data/fixed_obs/*.npz` に保存する |
| `tests/sim/test_libero_obs.py` | 自作の変換が `LiberoProcessorStep` と一致するかのテスト |

## 4. インタフェース仕様

```python
# src/evla/sim/types.py
@dataclass(frozen=True)
class Observation:
    images: dict[str, npt.NDArray[np.uint8]]   # {"image": (H, W, 3), "image2": (H, W, 3)}。方策に渡す向き（180°回転済み）
    state: npt.NDArray[np.float32]             # (8,)。LIBEROなら [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    task: str                                  # 自然言語の指示文
    sim_step: int                              # エピソード開始からのステップ数（待機ステップを除く）

@dataclass(frozen=True)
class StepResult:
    obs: Observation
    success: bool
    done: bool          # 成功、または最大ステップ到達
    info: dict[str, Any]
```

```python
# src/evla/sim/libero_env.py
@dataclass(frozen=True)
class LiberoConfig:
    suite: str = "libero_object"
    task_id: int = 0
    image_size: int = 256          # LIBEROの描画解像度（学習データと同じ256を既定にする）
    init_state_index: int = 0      # 固定初期状態のどれを使うか（＝エピソード番号）
    control_mode: str = "relative"

class LiberoTaskEnv:
    def __init__(self, cfg: LiberoConfig) -> None: ...
    @property
    def task(self) -> str: ...
    @property
    def max_steps(self) -> int: ...
    def reset(self, init_state_index: int | None = None, seed: int | None = None) -> Observation:
        """環境をリセットし、待機ステップ後の最初の観測を返す。"""
    def step(self, action: npt.NDArray[np.float32]) -> StepResult:
        """action: (7,) float32、範囲[-1, 1]。範囲外なら ValueError（クリップで隠さない）。"""
    def render_frame(self) -> npt.NDArray[np.uint8]:
        """動画保存用の画像（人間が見やすい向き）。"""
    def close(self) -> None: ...

def quat_xyzw_to_axis_angle(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(4,) (x,y,z,w) → (3,) 軸角。角度0付近の特異点を安全に扱う。"""
```

固定観測セットの形式（`data/fixed_obs/<suite>_t<task>.npz`）：

| キー | 形 | 内容 |
| --- | --- | --- |
| `image` | `(N, 256, 256, 3)` uint8 | 方策に渡す向き |
| `image2` | `(N, 256, 256, 3)` uint8 | 同上 |
| `state` | `(N, 8)` float32 | |
| `task` | `()` str | |
| `sim_step` | `(N,)` int | 取得したステップ |

## 5. 実装手順

### Step 1：描画バックエンドの確認（30分〜2時間）

1. `.venv-bench` で、次の順に試す。成功したものを `scripts/bench_env.sh` に書いておく。

| 順 | 設定 | 備考 |
| --- | --- | --- |
| 1 | `MUJOCO_GL=egl` | GPUのオフスクリーン描画。WSL2でIntelのEGLデバイスが見えれば最速 |
| 2 | `MUJOCO_GL=glfw` | WSLgのディスプレイ（`DISPLAY`）を使う。ウィンドウは開かなくてよい |
| 3 | `MUJOCO_GL=osmesa` | CPUでのソフトウェア描画。確実に動くが遅く、**推論とCPUを取り合う** |

2. 各設定で、256×256の2カメラを100ステップ描画する時間を測り、メモに残す。

### Step 2：LeRobotの評価コマンドで素の動作を確認（1時間）

```bash
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero \
  --env.task=libero_object --env.task_ids='[0]' --eval.batch_size=1 --eval.n_episodes=1 \
  --policy.device=cpu --output_dir=results/smoke/lerobot_eval
```

- 動画が出力され、成功するかを確認する（CPUなので数分かかる）。**これが「正解の挙動」の基準**になる。
- **【要確認】** `--policy.device` などの引数名は、`lerobot-eval --help` で確認する。

### Step 3：`LiberoTaskEnv` の実装（2時間）

1. LeRobotの `LiberoEnv` を `obs_type="pixels_agent_pos"` で作る（suite は `libero.libero.benchmark` から取得する。LeRobotの `create_libero_envs` の実装を読むと手順が分かる）。
2. `reset()` / `step()` の生の観測から `Observation` を作る。
   - 画像を180°回転する（`img[::-1, ::-1]`）。**回転後は `np.ascontiguousarray` で連続メモリにする**（後段のtorchやprotobufが、負のストライドを扱えない）。
   - 状態は `[eef_pos, quat_xyzw_to_axis_angle(eef_quat), gripper_qpos]` を連結する。
3. 行動の範囲チェック、`max_steps` での打ち切りを実装する。

### Step 4：一致テスト（1時間）

- 同じ生の観測を、自作の変換と LeRobot の `LiberoProcessorStep` の両方に通し、画像は完全一致、状態は `atol=1e-6` で一致することを確かめる。

### Step 5：スモークテストと固定観測の保存（1時間）

1. `scripts/libero_smoke.py`：ランダム行動で1エピソード回し、`render_frame` で動画（mp4）を保存する。
2. `scripts/dump_fixed_obs.py`：Step 2で成功したエピソードを、PyTorchのSmolVLAで（[23](23_runtime_pytorch.md) の完成後に）回しながら、10ステップごとに観測を保存する（N=20〜30）。
   - 23章が未完成なら、ランダム行動で保存しておき、後で差し替える。画像の中身は推論時間にほとんど影響しないが、**行動の数値比較には実際の観測が必要**。

## 6. ベストプラクティス

- **既存実装との一致テストを先に書く**：前処理（回転、軸角変換）のずれは、成功率0%という分かりにくい形で現れる。LeRobotの実装を「正解」として比較テストを書けば、すぐに見つかる。
- **観測をランタイム非依存の型にする**：`Observation` は NumPy だけで表し、torch.Tensor は入れない。ランタイムごとの変換は各Policyアダプタの責任にする（[22](22_policy_interface.md)）。
- **行動をクリップで隠さない**：範囲外の行動が来たら例外にするか、少なくとも警告を出して回数を数える。量子化で行動が発散していることに気づけなくなる。
- **シードと初期状態の番号を記録する**：`init_state_index` を変えると難易度が変わる。比較する設定どうしで同じ番号の集合を使う。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `GLXBadContext`、`EGL_NOT_INITIALIZED` | WSLgのGL環境とMuJoCoの相性 | Step 1の順に切り替える |
| 方策が全く動かない（腕が明後日の方向へ行く） | 画像を回転していない | 一致テストを通す |
| `ValueError: ... negative strides` | 回転後の配列がビューのまま | `np.ascontiguousarray` |
| 初回のresetだけ極端に遅い | モデルと描画器の構築（hard reset） | 計時の対象から外す。`hard_reset=False` は報告値の再現には使わない（LeRobotのドキュメント） |
| 状態の軸角の符号が反転する | クォータニオンの (w, x, y, z) と (x, y, z, w) の取り違え | robosuiteは (x, y, z, w)。テストで確認する |
| osmesaだとレイテンシの測定値が悪化する | 描画がCPUを使い、推論と競合する | モードAは固定観測で測るので影響しない。モードCは仮想時間方式にする（[27](27_closed_loop_b_c.md)） |

## 8. ヒント

<details><summary>L1：LeRobotのLiberoEnvをどう作るか</summary>

LeRobotの `create_libero_envs` や `_make_env_fns` は、ベクトル化環境（複数並列）を作るための関数。中で `LiberoEnv(task_suite=..., task_id=..., task_suite_name=..., ...)` を呼んでいるので、その部分だけを真似して1つの環境を直接作る。
</details>

<details><summary>L2：使う関数・モジュール</summary>

`from libero.libero import benchmark` → `benchmark.get_benchmark_dict()[suite]()` でスイートを取得する。`lerobot.envs.libero.LiberoEnv`、`lerobot.processor.env_processor.LiberoProcessorStep`（テストでの比較用）、動画保存は `imageio.mimsave(path, frames, fps=20)`（`imageio[ffmpeg]` が必要）。
</details>

<details><summary>L3：軸角変換の疑似コード</summary>

```
x, y, z, w = q;  w = clip(w, -1, 1)
den = sqrt(1 - w*w)
if den < 1e-8: return zeros(3)        # 回転がほぼ0
angle = 2 * arccos(w)
return array([x, y, z]) / den * angle
```
</details>

## 9. 完了条件（DoD）

- [ ] 使う `MUJOCO_GL` と、その描画速度がメモにある
- [ ] `lerobot-eval` でSmolVLAが `libero_object` タスク0を1回成功させた動画がある
- [ ] `pytest tests/sim/test_libero_obs.py` が通る（`LiberoProcessorStep` との一致）
- [ ] `data/fixed_obs/libero_object_t0.npz` がある（N≥20）
- [ ] 範囲外の行動を `step` に渡すと `ValueError` になる

## 10. 理解度チェック・発展課題

- LIBEROの行動が「手先の差分」であることは、遅延に対する頑健性にどう影響するか。関節の絶対位置指令（トラック2のSO-101）と比べよ。
- なぜ学習データの画像は180°回転しているのか（ヒント：データ収集時の描画の向き）。
- 発展：`image_size` を256から128に下げると、SmolVLA内部の512pxへのリサイズで何が起きるか。推論時間は変わるか。
