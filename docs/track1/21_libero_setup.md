# track1/21. LIBEROの導入と環境ラッパ

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**14個**（`src/evla/sim/libero_env.py` 9、`src/evla/sim/fixed_obs.py` 2、`scripts/libero_smoke.py` 2、`scripts/dump_fixed_obs.py` 1）
> - ゴール：LIBERO が描画付きで動き、`data/fixed_obs/libero_object_t0.npz`（N≥20）ができる
> - テスト：LIBERO が必要なので pytest はなし。各 Step の「動かす」のコマンドで確かめる

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

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章のコードには **【要確認】**（LeRobot の版で変わりうる行）が多いので、穴を埋める前に、実物の観測のキーを一度見ておくのがコツです。

### Step 0：準備（5分）

```bash
source .venv-bench/bin/activate
grep -n "【穴\|【要確認】" src/evla/sim/{libero_env,fixed_obs}.py scripts/{libero_smoke,dump_fixed_obs}.py
```

- [ ] 穴14個と【要確認】の行が表示される。【要確認】の行は Step 3 で実物と照らし合わせる

### Step 1：描画バックエンドを選ぶ（30分〜2時間）

次の順に試し、動いたものを `scripts/bench_env.sh`（自分で作る）に `export MUJOCO_GL=...` として書いておきます。

| 順 | 設定 | 備考 |
| --- | --- | --- |
| 1 | `MUJOCO_GL=egl` | GPUのオフスクリーン描画。WSL2でIntelのEGLデバイスが見えれば最速 |
| 2 | `MUJOCO_GL=glfw` | WSLgのディスプレイ（`DISPLAY`）を使う。ウィンドウは開かなくてよい |
| 3 | `MUJOCO_GL=osmesa` | CPUでのソフトウェア描画。確実に動くが遅く、**推論とCPUを取り合う** |

```bash
for gl in egl glfw osmesa; do
  echo "== $gl"; MUJOCO_GL=$gl python -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); r = mujoco.Renderer(m, 256, 256); print('ok')"
done
```

- [ ] どの設定で `ok` が出たかをメモに残す（描画速度は Step 5 のスモークテストで測る）

### Step 2：LeRobot の評価コマンドで「正解の挙動」を見る（1時間）

```bash
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero \
  --env.task=libero_object --env.task_ids='[0]' --eval.batch_size=1 --eval.n_episodes=1 \
  --policy.device=cpu --output_dir=results/smoke/lerobot_eval
```

- [ ] `results/smoke/lerobot_eval/` に動画が出力され、成功している（CPUなので数分かかる）。**これが以降の基準**になる
- [ ] 引数でエラーになったら `lerobot-eval --help` で名前を確かめる（**【要確認】**）

### Step 3：軸角の変換 ― `libero_env.py` の穴①〜④（30分）

LIBERO の状態は、手先の姿勢をクォータニオンで持っています。方策の入力は軸角（3次元）なので変換します。

**読む**：4節の `quat_xyzw_to_axis_angle`。クォータニオン `(x, y, z, w) = (sin(θ/2)·軸, cos(θ/2))` の関係を紙に書く。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 成分の並び順。LIBERO（robosuite）は `xyzw` の順。`wxyz` の順を使うライブラリもあり、取り違えが典型的なバグ |
| ② | `w` から `sin(θ/2)` を求める。丸め誤差で負にならないよう、外側で `max(0, …)` している |
| ③ | 回転がほぼ0だと軸が定まらず、0 で割ってしまう。その手前で止める（しきい値は 1e-8 程度の小さな値） |
| ④ | `w = cos(θ/2)` を θ について解く |

**動かす**（LIBERO がなくても確かめられる）

```bash
python -c "
import numpy as np
from evla.sim.libero_env import quat_xyzw_to_axis_angle as f
print(f(np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])))   # z 軸まわり 90°
print(f(np.array([0, 0, 0, 1.0])))                              # 回転なし
print(f(np.array([1.0, 0, 0, 0])))                              # x 軸まわり 180°
"
```

- [ ] 1行目が `[0. 0. 1.5708...]`（π/2）、2行目が `[0. 0. 0.]`、3行目が `[3.1415... 0. 0.]`

### Step 4：環境ラッパ ― `libero_env.py` の穴⑤〜⑨（1.5時間）

**読む**：4節の `LiberoTaskEnv` と、LeRobot の `create_libero_envs` の実装（スイートの作り方が分かる）。

```bash
python -c "import lerobot.envs.libero as m, inspect; print(inspect.getsourcefile(m))"   # ソースの場所
```

**埋める**

| 穴 | ねらい |
| --- | --- |
| ⑤ | LIBERO の `benchmark` モジュールで、スイート名からスイートのクラスを引く辞書を得る |
| ⑥ | 学習データの画像は180°回転している。NumPy のスライスだけで回転できる（コピーなし＝負のストライド。だから外側で連続メモリにしている） |
| ⑦ | 位置(3)・軸角(3)・グリッパ(2) を1本の (8,) にする |
| ⑧ | 範囲外の行動は、クリップで隠さず例外にする（方策のバグを早く見つけるため）。誤差を少し許す |
| ⑨ | gymnasium の API。戻り値が5つ（`terminated` と `truncated` が分かれている）ことに注意 |

**動かす**：まず生の観測のキーを見て、【要確認】の行（`raw["pixels"][key]`、`raw["robot_state"]`）が合っているか確かめます。

```bash
export MUJOCO_GL=egl    # Step 1 で選んだもの
python -c "
from evla.sim.libero_env import LiberoConfig, LiberoTaskEnv
import numpy as np
env = LiberoTaskEnv(LiberoConfig())
raw, _ = env._env.reset(seed=0); print('raw keys:', list(raw.keys()))
obs = env.reset(seed=0)
print(env.task, obs.state.shape, {k: (v.shape, v.flags['C_CONTIGUOUS']) for k, v in obs.images.items()})
try:
    env.step(np.full(7, 1.5, np.float32))
except ValueError as e:
    print('ValueError OK:', e)
env.close()
"
```

**確かめる**

- [ ] `raw keys` が【要確認】の行と合っている（合っていなければ、その行を実物に合わせて直す）
- [ ] 指示文、`(8,)`、両カメラが `((256, 256, 3), True)` と表示される
- [ ] 範囲外の行動で `ValueError OK:` が表示される

**一致の確認**：同じ生の観測を、自作の変換と LeRobot の前処理（`LiberoProcessorStep`。**【要確認】** 名前と場所は版による）の両方に通し、画像は完全一致、状態は `atol=1e-6` で一致することを、Python で1回確かめてメモに残します（モデルが要らないので、上のコマンドに数行足せば書けます）。

### Step 5：スモークテスト ― `scripts/libero_smoke.py`（穴①②、30分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 乱数の作り方。`np.random.default_rng` の Generator のメソッドを使う（グローバルな `np.random.*` は使わない） |
| ② | `imageio` で画像のリストを mp4 にする関数 |

**動かす**

```bash
for gl in egl glfw osmesa; do MUJOCO_GL=$gl python scripts/libero_smoke.py --steps 100; done
```

**確かめる**

- [ ] `100 steps, xx.x ms/step（描画込み）` が表示される。設定ごとの値をメモに残す（9節の1つ目）
- [ ] `results/smoke/libero_random.mp4` を再生すると、ロボットがランダムに動いている（向きが上下逆でない）

### Step 6：固定観測セット ― `fixed_obs.py`（穴①②）と `scripts/dump_fixed_obs.py`（穴①）（30分）

推論時間の測定（26章）と、ランタイム間の一致の確認（22章）で、全ランタイムに**同じ観測**を入れるためのファイルを作ります。

**埋める**

| 穴 | ねらい |
| --- | --- |
| `fixed_obs.py` ① | `(H, W, 3)` の配列 N 個を、新しい軸で積んで `(N, H, W, 3)` にする |
| `fixed_obs.py` ② | `.npz` を開く関数。`with` で使うと、ファイルが確実に閉じられる |
| `dump_fixed_obs.py` ① | ステップ番号が `every` の倍数のときだけ保存する |

**動かす**

```bash
python scripts/dump_fixed_obs.py --suite libero_object --task 0 --n 25
python -c "
from pathlib import Path
from evla.sim.fixed_obs import load_fixed_obs
obs = load_fixed_obs(Path('data/fixed_obs/libero_object_t0.npz'))
print(len(obs), obs[0].task, obs[0].state.dtype, [o.sim_step for o in obs[:5]])
"
```

**確かめる**

- [ ] `saved 25 observations → data/fixed_obs/libero_object_t0.npz`
- [ ] 読み込むと `25 <指示文> float32 [0, 10, 20, 30, 40]` のように、10ステップおきの観測が戻る
- [ ] 23章が終わったら、`--policy configs/policy/torch_fp32.yaml` を付けて**実際の観測で作り直す**（行動の数値比較には実際の観測が必要。推論時間にはほとんど影響しない）

### Step 7：仕上げ（10分）

- [ ] 4ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/sim scripts/libero_smoke.py scripts/dump_fixed_obs.py` を通す
- [ ] 【要確認】の行を直した場合は、どう直したかを `docs/decisions.md` に1行残す
- [ ] コミットする

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
- [ ] 5節 Step 4 の一致の確認（`LiberoProcessorStep` との一致）の結果がメモにある
- [ ] `data/fixed_obs/libero_object_t0.npz` がある（N≥20）
- [ ] 範囲外の行動を `step` に渡すと `ValueError` になる

## 10. 理解度チェック・発展課題

- LIBEROの行動が「手先の差分」であることは、遅延に対する頑健性にどう影響するか。関節の絶対位置指令（トラック2のSO-101）と比べよ。
- なぜ学習データの画像は180°回転しているのか（ヒント：データ収集時の描画の向き）。
- 発展：`image_size` を256から128に下げると、SmolVLA内部の512pxへのリサイズで何が起きるか。推論時間は変わるか。
