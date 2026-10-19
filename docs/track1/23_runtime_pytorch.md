# track1/23. PyTorch CPUベースライン（LeRobotのSmolVLA）

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**7個**（`src/evla/policy/torch_policy.py`）
> - ゴール：固定観測で SmolVLA が `(50, 7)` の行動を返し、固定ノイズなら2回とも完全に一致する。簡易ループで `libero_object` タスク0に1回成功する
> - テスト：モデルが必要なので pytest はなし。各 Step の「動かす」のコマンドで確かめる

## 1. 目的と計画書の対応

- LeRobotの `SmolVLAPolicy` を `Policy` 抽象（22章）の実装として包み、**全比較の基準（FP32）**にする。
- スレッド数の効果を測れるようにする。
- 計画書の対応箇所：トラック1「掃引軸（ランタイム＝PyTorch CPU）」、日程D2「SmolVLA PyTorch CPUベースライン」

## 2. 前提知識・事前に読む資料

- LeRobotのSmolVLA：<https://huggingface.co/docs/lerobot/en/smolvla>
- ソース（読解用）：`src/lerobot/policies/smolvla/modeling_smolvla.py`（`predict_action_chunk`、`sample_actions`）、`processor_smolvla.py`、`src/lerobot/policies/factory.py`（`make_pre_post_processors`）
- PyTorchのCPUスレッド：<https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html>

### 2.1 調査で確認した事実

| 項目 | 値 |
| --- | --- |
| チェックポイント | `HuggingFaceVLA/smolvla_libero`（`lerobot/smolvla_libero` もある。vla.cppのGGUFとOpenVINO版は前者から変換されている → **前者を基準にする**） |
| 前処理パイプライン | rename → to_batch → 指示文の末尾に改行を付ける → トークナイズ（最大48、`HuggingFaceTB/SmolVLM2-500M-Instruct`） → デバイス移動 → 正規化（状態と行動はMEAN_STD、画像はIDENTITY） |
| デバイス設定 | 保存された前処理は `device: "cuda"` → **CPUで使うには上書きが必要** |
| モデル設定 | `chunk_size=50`、`n_action_steps=50`、`num_steps=10`、`resize_imgs_with_padding=(512, 512)`、`max_state_dim=32`、`max_action_dim=32` |
| 画像の値域 | 前処理には [0, 1] のfloatで渡す。モデル内部で [-1, 1] に変換する |
| 視点の欠落 | `config.image_features` にあるキーがバッチになければ、その視点を使わない（`empty_cameras=0` の場合）→ 1視点の実験ができる |
| torch.compile | `config.compile_model` で有効にできる（`sample_actions` と `forward` をコンパイル） |

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/policy/torch_policy.py` | `TorchSmolVLA(Policy)` |
| （5節 Step 2〜4 のコマンド） | 形、決定性（固定ノイズ）、1視点で動くことの確認。専用のスクリプトとテストは作らない |

## 4. インタフェース仕様

```python
# src/evla/policy/torch_policy.py
class TorchSmolVLA(Policy):
    def __init__(self, cfg: PolicyConfig) -> None:
        """cfg.model: HFのrepo_idまたはローカルパス。
        cfg.precision: "fp32" | "bf16"（CPUでのbf16は、AVX-512やAMXのないこのCPUでは遅い可能性が高い）
        cfg.n_threads: torch.set_num_threads に渡す。
        cfg.extra["compile"]: bool、torch.compile を使うか。"""

    def _to_batch(self, obs: Observation) -> dict[str, Any]:
        """Observation → LeRobotの前処理が期待する辞書。
        画像は (1, 3, H, W) float32 [0,1]。cfg.views にない視点は入れない。"""

    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        """前処理 → policy.predict_action_chunk(batch, noise=...) → 後処理（逆正規化）。
        breakdown_ms に preprocess / inference / postprocess を入れる。"""
```

`cfg.image_size` を512以外にする実験では、`policy.config.resize_imgs_with_padding` を書き換えます。ただし、学習時と異なる解像度でビジョンエンコーダが正しく動くかは保証されていません（位置埋め込みの補間の有無）。**【要確認】** 256で1回推論し、エラーにならないか、出力が極端に壊れないかを確認してから掃引に入れる。

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。初回はモデルのダウンロード（約1GB）があるので、Step 0 で先に済ませておきます。

### Step 0：準備（15分）

```bash
source .venv-bench/bin/activate
grep -n "【穴\|【要確認】" src/evla/policy/torch_policy.py
huggingface-cli download HuggingFaceVLA/smolvla_libero     # 先にダウンロードしておく
cat configs/policy/torch_fp32.yaml
```

- [ ] 穴7個と【要確認】の行が表示される
- [ ] 設定が `runtime: torch`、`n_threads: 8` になっている

### Step 1：モデルの読み込み ― 穴①〜③（30分）

**読む**：LeRobot の `SmolVLAPolicy` のソース（`from_pretrained` と `predict_action_chunk` の場所を探す）。

```bash
python -c "import lerobot.policies.smolvla.modeling_smolvla as m, inspect; print(inspect.getsourcefile(m))"
```

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | intra-op（1つの演算の中の並列）のスレッド数。**モデルを読む前に**設定する（後から変えると効かない場合がある） |
| ② | Hugging Face の流儀。repo_id でもローカルのパスでも読めるクラスメソッド |
| ③ | dropout などを推論の動作に切り替える。忘れると、同じ入力でも出力が毎回変わる |

**動かす**

```bash
python -c "
from evla.policy.base import PolicyConfig
from evla.policy.torch_policy import TorchSmolVLA
p = TorchSmolVLA(PolicyConfig(runtime='torch', model='HuggingFaceVLA/smolvla_libero', n_threads=8))
print(p.chunk_size, p.action_dim, p.describe())
"
```

- [ ] `50 7 {... 'num_threads': 8, 'dtype': 'torch.float32'}` のように表示される
- [ ] `policy.training` が False になっていることも確かめる（`print(p.policy.training)`）

### Step 2：推論 ― 穴④〜⑦（1時間）

**読む**：4節の `_to_batch` と `predict_chunk`。テンソルの形が `(H,W,3)` → `(3,H,W)` → `(1,3,H,W)` → … → `(1,50,7)` → `(50,7)` と変わる流れを紙に書く。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ④ | モデルはバッチで受け取る。先頭に大きさ1の軸を足す |
| ⑤ | 推論では勾配の記録が要らない。`no_grad` より軽い方のデコレータを使う |
| ⑥ | `select_action` は内部にキューを持ち、1行動ずつ返す（状態を持つ）。チャンクを丸ごと返す方を使う（キューは27章の executor で自分で持つ） |
| ⑦ | `(1, 50, 7)` の先頭の軸を外す |

**動かす**

```bash
python - <<'EOF'
from pathlib import Path
import numpy as np
from evla.common.config import load_config
from evla.policy.base import PolicyConfig
from evla.policy.equivalence import make_fixed_noise
from evla.policy.factory import make_policy
from evla.sim.fixed_obs import load_fixed_obs

obs = load_fixed_obs(Path("data/fixed_obs/libero_object_t0.npz"))[0]
noise = make_fixed_noise(50, 32, 0)
with make_policy(load_config(Path("configs/policy/torch_fp32.yaml"), PolicyConfig)) as p:
    r1, r2 = p.predict_chunk(obs, noise), p.predict_chunk(obs, noise)
    print(r1.actions.shape, r1.actions.min().round(3), r1.actions.max().round(3))
    print("identical:", np.array_equal(r1.actions, r2.actions))
    print({k: round(v, 1) for k, v in r2.breakdown_ms.items()}, round(r2.latency_ms, 1))
EOF
```

**確かめる**

- [ ] `(50, 7)` で、値がおおむね [-1, 1] の範囲にある
- [ ] `identical: True`（固定ノイズなら決定的。False なら③の `eval()` か、ノイズの渡し方を疑う）
- [ ] `breakdown_ms` で、`inference` が大半を占めている。2回目の値をメモに残す（1回目はウォームアップを含むので遅い）

### Step 3：1視点で動くか（15分）

```bash
python -c "
from pathlib import Path
from evla.policy.base import PolicyConfig
from evla.policy.torch_policy import TorchSmolVLA
from evla.sim.fixed_obs import load_fixed_obs
obs = load_fixed_obs(Path('data/fixed_obs/libero_object_t0.npz'))[0]
p = TorchSmolVLA(PolicyConfig(runtime='torch', model='HuggingFaceVLA/smolvla_libero', views=('image',), n_threads=8))
print(p.predict_chunk(obs).actions.shape)
"
```

- [ ] `(50, 7)` が表示される（26章で「視点数」を掃引するための確認）

### Step 4：簡易ループで成功させる（1時間）

21章のラッパで1エピソード回し、チャンクの先頭10行動ずつ実行します（本格的な executor は27章）。

```bash
MUJOCO_GL=egl python - <<'EOF'
from pathlib import Path
from evla.common.config import load_config
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy
from evla.sim.libero_env import LiberoConfig, LiberoTaskEnv

env = LiberoTaskEnv(LiberoConfig(suite="libero_object", task_id=0))
with make_policy(load_config(Path("configs/policy/torch_fp32.yaml"), PolicyConfig)) as p:
    obs, done, res = env.reset(seed=0), False, None
    while not done:
        for a in p.predict_chunk(obs).actions[:10]:
            res = env.step(a)
            obs, done = res.obs, res.done
            if done:
                break
    print("success:", res.success, "steps:", obs.sim_step)
env.close()
EOF
```

- [ ] `success: True` が1回以上出る（失敗したら `--seed` を変えて数回。21章 Step 2 の `lerobot-eval` が成功していれば、ここも成功するはず）

### Step 5：スレッド数の予備測定（1時間）

`n_threads` を 1, 2, 4, 8, 12, 16 と変えて、各20回の推論時間を測ります（本格的な測定は26章）。**`OMP_NUM_THREADS` も同じ値にして、torch の import 前に設定**します。

```bash
for n in 1 2 4 8 12 16; do
  OMP_NUM_THREADS=$n python -c "
from pathlib import Path; import numpy as np
from evla.policy.base import PolicyConfig
from evla.policy.torch_policy import TorchSmolVLA
from evla.sim.fixed_obs import load_fixed_obs
obs = load_fixed_obs(Path('data/fixed_obs/libero_object_t0.npz'))[0]
p = TorchSmolVLA(PolicyConfig(runtime='torch', model='HuggingFaceVLA/smolvla_libero', n_threads=$n))
p.warmup(obs)
t = [p.predict_chunk(obs).latency_ms for _ in range(20)]
print($n, 'p50=%.0f ms' % np.median(t), 'p90=%.0f ms' % np.percentile(t, 90))
"
done
```

- [ ] スレッド数ごとの p50 / p90 の表をメモに残す。どこで頭打ちになるか（物理コア数との関係）を1行で書く

### Step 6：仕上げ（10分）

- [ ] 先頭の `from evla._blank import ___` を消し、`ruff check src/evla/policy/torch_policy.py` を通す
- [ ] 【要確認】の行を直した場合は `docs/decisions.md` に残す
- [ ] 22章の Step 5（一致の表）に戻り、`torch_fp32` どうしで `max_abs=0` を確かめる
- [ ] 21章の Step 6 に戻り、`--policy configs/policy/torch_fp32.yaml` で固定観測を作り直す
- [ ] コミットする

## 6. ベストプラクティス

- **`inference_mode` を使う**：`no_grad` より軽い。勾配の記録用のメモリを確保しない。
- **バッチは1で固定**：エッジでの制御はバッチ1が本番。バッチ化すればスループットは上がるが、それはこの研究の問いではない。
- **スレッド数を明示し、結果に記録する**：既定値はPyTorchのビルドとCPUの検出結果で変わる。`torch.get_num_threads()` の値を `describe()` で返す。
- **torch.compile はモードAで効果を確かめてから使う**：CPUではコンパイルに数分かかり、効果は数%〜数十%と幅がある。採用するなら、コンパイル時間を別に報告する。
- **bf16は「測ったうえで不採用」と記録してもよい**：このCPU（AVX-512、AMXなし）では、bf16の行列演算が速くならない可能性が高い。測定した事実を残すことに価値がある。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `AssertionError: Torch not compiled with CUDA enabled` | 保存された前処理の `device: cuda` | `preprocessor_overrides` で上書きする |
| スレッド数を増やしても速くならない、または遅くなる | Eコアの参加で同期待ちが増える。over-subscription | 26章で掃引し、最適値を探す。`OMP_NUM_THREADS` と `set_num_threads` を一致させる |
| `set_num_interop_threads` でエラー | 並列処理が既に始まった後に呼んだ | スクリプトの冒頭で呼ぶ |
| 初回だけ数秒遅い | 重みの遅延ロード、oneDNNのカーネル選択 | ウォームアップで除外する |
| トークナイザのダウンロードで止まる | オフライン環境、またはHFのレート制限 | 事前に `huggingface-cli download` しておき、`HF_HUB_OFFLINE=1` で動かす |

## 8. ヒント

<details><summary>L1：前処理の入力形式</summary>

LeRobotの前処理は「環境から来た1サンプル」を想定している。キーは `observation.images.image`、`observation.images.image2`、`observation.state`、`task`。to_batchのステップがバッチ次元を足すので、自分で足すべきかどうかは、そのステップの実装を読んで確かめる。
</details>

<details><summary>L2：使うAPI</summary>

`SmolVLAPolicy.from_pretrained`、`lerobot.policies.factory.make_pre_post_processors`、`preprocessor(batch)`、`policy.predict_action_chunk(batch, noise=noise)`、`postprocessor(actions)`、`torch.from_numpy(img).permute(2, 0, 1).float().div(255)`
</details>

<details><summary>L3：predict_chunk の疑似コード</summary>

```
t0 = now()
batch = self._to_batch(obs)                  # numpy → torch、[0,1]、CHW
batch = self.pre(batch)                      # 改行、トークナイズ、正規化
t1 = now()
with torch.inference_mode():
    a = self.policy.predict_action_chunk(batch, noise=noise_t)   # (1, 50, 7)
t2 = now()
a = self.post(a)                             # 逆正規化
actions = a[0].numpy().astype(np.float32)
t3 = now()
return ChunkResult(actions, ms(t3 - t0), {"preprocess": ms(t1-t0), "inference": ms(t2-t1), "postprocess": ms(t3-t2)})
```
</details>

## 9. 完了条件（DoD）

- [ ] 固定ノイズで同じ観測を2回推論すると、完全に一致する
- [ ] `views=("image",)` でもエラーなく推論できる
- [ ] 簡易ループで `libero_object` タスク0を1回以上成功させた
- [ ] スレッド数別の予備測定の表がある
- [ ] `describe()` が torch の版、スレッド数、実行デバイスを返す

## 10. 理解度チェック・発展課題

- `predict_action_chunk` と `select_action` の違いは何か。このプロジェクトで前者を使う理由は。
- 前処理の「指示文の末尾に改行を付ける」ステップを外すと、何が起きると予想されるか（実際に試して、行動の差を22章の指標で測ってみよ）。
- 発展：`torch.profiler` で推論を1回プロファイルし、ビジョン・言語モデル・アクションエキスパートの時間の比率を求めよ。20章の表（ビジョンが6割以上）と比べてどうか。
