# track1/22. ランタイム共通のPolicy抽象と数値一致の検証

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**14個**（`src/evla/policy/` の `factory.py` 1、`preprocess.py` 5、`equivalence.py` 6、`scripts/check_equivalence.py` 2）
> - ゴール：`pytest tests/policy` がすべて通る。23章のあとで、PyTorch FP32 どうしの一致表（max_abs=0）が出る
> - 進め方：Step 1〜4 はモデルなしでできる。Step 5 は23章（PyTorch 版）が動いてから戻ってくる

## 1. 目的と計画書の対応

- PyTorch・OpenVINO・vla.cppを**同じ呼び出し方で使える抽象クラス**を作る。ベンチ（26章）や閉ループ（27章）は、この抽象だけに依存させる。
- ランタイム間で**同じ入力から同じ行動が出るか**を検証する仕組みを作る。
- 計画書の対応箇所：トラック1「掃引軸（ランタイム、精度、入力）」

## 2. 前提知識・事前に読む資料

- Pythonの抽象基底クラス：<https://docs.python.org/3/library/abc.html>、`typing.Protocol`
- flow matchingの推論：SmolVLA論文の3.2節。**初期ノイズを固定しないと、同じ入力でも毎回出力が変わる**。
- LeRobotの `SmolVLAPolicy.predict_action_chunk(batch, noise=None)`：ノイズを外から渡せる（`modeling_smolvla.py`）。
- vla.cppの `PredictRequest.noise`（`src/serving/vla.proto`）：同じくノイズを外から渡せる。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/policy/base.py` | `Policy` 抽象クラス、`PolicyConfig`、`ChunkResult` |
| `src/evla/policy/factory.py` | `make_policy(cfg)`：設定からアダプタを作る |
| `src/evla/policy/preprocess.py` | ランタイム間で共通の前処理（視点の選択、リサイズ、正規化の補助） |
| `src/evla/policy/equivalence.py` | 数値一致の指標と比較の実行 |
| `scripts/check_equivalence.py` | 固定観測×固定ノイズで全ランタイムを比較し、表を出す |
| `tests/policy/test_equivalence_metrics.py` | 指標関数のテスト |

## 4. インタフェース仕様

```python
# src/evla/policy/base.py
Runtime = Literal["torch", "ov", "vlacpp"]

@dataclass(frozen=True)
class PolicyConfig:
    runtime: Runtime
    model: str                      # HFのrepo_id、またはローカルパス（ランタイムごとに意味が違う）
    device: str = "CPU"             # "CPU" | "GPU"（ov）、vlacppでは情報用（実際の指定はサーバー側）
    precision: str = "fp32"         # "fp32" | "fp16" | "bf16" | "int8" | "q8_0" | "q4_0"
    views: tuple[str, ...] = ("image", "image2")   # 使う視点。1視点なら ("image",)
    image_size: int = 512           # モデル内部の入力解像度（SmolVLAの学習時は512）
    num_steps: int = 10             # flow matching のステップ数
    n_threads: int | None = None    # CPUスレッド数。None は各ランタイムの既定
    server_addr: str = "tcp://127.0.0.1:5555"      # vlacpp のみ
    extra: Mapping[str, Any] = field(default_factory=dict)   # ランタイム固有の設定

@dataclass(frozen=True)
class ChunkResult:
    actions: npt.NDArray[np.float32]     # (chunk_size, action_dim)。逆正規化済み（環境にそのまま渡せる値）
    latency_ms: float                    # predict_chunk の呼び出し全体（前処理・通信・後処理を含む）
    breakdown_ms: Mapping[str, float]    # 取れる範囲で内訳（"preprocess", "inference", "vision", "prefill", "denoise", "transport"）

class Policy(ABC):
    cfg: PolicyConfig

    @property
    @abstractmethod
    def chunk_size(self) -> int: ...
    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        """観測から行動チャンクを1つ推論する。
        noise: (chunk_size, max_action_dim) の初期ノイズ。Noneなら内部で乱数を使う。
        状態を持たない（キューは executor 側で管理する）。"""

    def warmup(self, obs: Observation, n: int = 3) -> None:
        """初回コンパイルやキャッシュ構築を済ませる。既定では predict_chunk を n 回呼ぶ。"""

    def close(self) -> None:
        """サーバー接続などの資源を解放する。"""

    def describe(self) -> dict[str, Any]:
        """結果ファイルに残すための情報（実際のデバイス名、ライブラリ版、スレッド数など）。"""
```

**設計上の要点**：

- `Policy` は**状態を持たない**。「チャンクから何行動を実行するか」「いつ次を推論するか」は、すべて executor（27章）が決める。LeRobotの `select_action` のような内部キューは使わない。キューが方策の中にあると、同期・非同期の比較や遅延注入ができなくなるため。
- `latency_ms` は**呼び出し側から見た時間**にする。ランタイムが返す内部の時間（vla.cppの `latency_ms_inference` など）は `breakdown_ms` に入れる。

```python
# src/evla/policy/equivalence.py
@dataclass(frozen=True)
class EquivalenceReport:
    max_abs: float               # 全要素の最大絶対誤差
    rmse: float
    cosine: float                # チャンク全体をベクトルとみなしたコサイン類似度
    per_dim_max_abs: list[float] # 次元ごと（グリッパの次元だけ大きい、などを見つける）
    gripper_sign_flips: int      # グリッパの開閉の符号が食い違ったステップ数

def compare_chunks(ref: npt.NDArray, test: npt.NDArray, gripper_dim: int = -1) -> EquivalenceReport: ...

def make_fixed_noise(chunk_size: int, max_action_dim: int, seed: int) -> npt.NDArray[np.float32]:
    """標準正規分布のノイズ。全ランタイムで同じものを使う。"""
```

```python
# src/evla/policy/factory.py
def make_policy(cfg: PolicyConfig) -> Policy:
    """cfg.runtime に応じて TorchSmolVLA / OVSmolVLA / VlaCppPolicy を返す。
    import はこの関数の中で遅延して行う（使わないランタイムのライブラリを読み込まない）。"""
```

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。

### Step 0：準備（5分）

```bash
source .venv-bench/bin/activate
grep -n "【穴" src/evla/policy/{factory,preprocess,equivalence}.py scripts/check_equivalence.py
pytest tests/policy -q          # まだ失敗する
```

- [ ] 穴14個が表示され、テストが `NotImplementedError` で失敗する

### Step 1：インタフェースを読む ― `base.py`（穴なし、20分）

`src/evla/policy/base.py` は穴のない完成版です。次の3つに答えられるまで読みます（答えはメモに1行ずつ）。

- [ ] `Policy` が**状態を持たない**のはなぜか（キューはどこで持つか）
- [ ] `predict_chunk` の `noise` 引数は、何のためにあるか（Step 5 の一致の確認で使う）
- [ ] `with make_policy(cfg) as policy:` と書くと、ブロックを出たときに何が呼ばれるか

### Step 2：一致の指標 ― `equivalence.py`（穴①〜⑥、40分）

ランタイムを替えたとき「同じ行動を出しているか」を数値で測る道具です。

**読む**：4節の `EquivalenceReport` と、`tests/policy/test_equivalence_metrics.py`。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 最悪の1要素。「どこか1か所でも大きくずれたか」を見る |
| ② | 全体の平均的なずれ。①と並べると、ずれが一部だけか全体かが分かる |
| ③ | 向きの一致（大きさは無視）。値の尺度が違っても比べられる |
| ④ | `(時間, 次元)` の配列で、どちらの軸に沿って最大をとれば「次元ごと」になるか |
| ⑤ | グリッパは「開くか閉じるか」が重要。値の差より、符号の食い違いを数える |
| ⑥ | NumPy の新しい乱数 API。シードを渡して、全ランタイムで同じノイズを作る |

**動かす**

```bash
pytest tests/policy/test_equivalence_metrics.py -v
python -c "
import numpy as np
from evla.policy.equivalence import compare_chunks
a = np.random.default_rng(0).normal(size=(50, 7))
print(compare_chunks(a, -a))           # 全部の符号を反転
print(compare_chunks(a, a + 1e-3))     # わずかにずらす
"
```

**確かめる**

- [ ] 4つのテストがすべて `PASSED`
- [ ] 符号反転：`cosine=-1.0`、`gripper_sign_flips` が 50 前後（0 の要素がなければ 50）
- [ ] わずかなずれ：`max_abs≈0.001`、`cosine` がほぼ 1。①②③の違いを、この2例で説明できる

### Step 3：前処理 ― `preprocess.py`（穴①〜⑤、40分）

ランタイム間で前処理が1ピクセルでも違うと、一致の表が意味をなさなくなります。だから共通の関数にしています。

**読む**：`resize_with_pad` の docstring の【要確認】（埋める側が「左と上」であること）。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 画素値の正規化。整数で割ると切り捨てになるので、浮動小数で割る |
| ② | 画像の並び。NumPy・OpenCV は `(H, W, C)`、PyTorch は `(C, H, W)`。`transpose` の軸の順を考える |
| ③ | 縦と横のうち、**大きい方の比**で縮める（そうしないと長辺がはみ出す） |
| ④ | `F.interpolate` の `size` は `(高さ, 幅)` の順 |
| ⑤ | `F.pad` の埋め幅は**最後の軸から**（左, 右, 上, 下）の順。左と上を埋め、右と下は0 |

**動かす**

```bash
pytest tests/policy/test_preprocess.py -v
python -c "
import numpy as np
from evla.policy.preprocess import resize_with_pad
y = resize_with_pad(np.ones((3, 480, 640), np.float32), 512)
print(y.shape, 'pad rows:', int((y[0].max(axis=1) == 0).sum()), 'pad cols:', int((y[0].max(axis=0) == 0).sum()))
"
```

**確かめる**

- [ ] 3つのテストがすべて `PASSED`（`test_resize_with_pad_keeps_aspect` が `SKIPPED` なら torch が入っていない → [env/12](../env/12_inference_envs.md) Step 1）
- [ ] `(3, 512, 512) pad rows: 128 pad cols: 0`。640×480 → 512×384 に縮み、上に 128 行の埋めが入る

### Step 4：ランタイムの切り替え ― `factory.py`（穴①、10分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 他の分岐と同じ形。import を関数の中に書く理由（使わないランタイムのライブラリを読み込まない）も確認する |

**動かす**

```bash
python -c "
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy
make_policy(PolicyConfig(runtime='foo', model='x'))
"
```

- [ ] `ValueError: 未知のランタイム: foo` で終わる（`torch` の分岐は23章で動かす）

### Step 5：一致の表を作る ― `scripts/check_equivalence.py`（穴①②、23章のあと、30分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `argparse` の `nargs`。`--test a.yaml b.yaml c.yaml` のように1個以上受け取る |
| ② | 観測ごとの `rmse` の平均（`reps` は観測ごとの `EquivalenceReport` のリスト） |

**動かす**：まず同じ設定どうしで比べ、**決定性**を確かめます。

```bash
python scripts/check_equivalence.py --fixed_obs data/fixed_obs/libero_object_t0.npz \
  --ref configs/policy/torch_fp32.yaml --test configs/policy/torch_fp32.yaml --n 3 \
  --out reports/equivalence_libero.md
```

**確かめる**

- [ ] 表の `max_abs` が `0.0`、`cosine_min` が `1.0`（0 でなければ、どこかに乱数か非決定的な並列化がある）
- [ ] 24・25章でアダプタができるたびに、`--test` に設定を足して表を更新する。次の目安と比べる

**比較の目安（vla.cppのOVバックエンド文書の値を参考にした）**：

| 比較 | max_abs の目安 | 解釈 |
| --- | --- | --- |
| 同じランタイムで2回 | 0（決定的なら） | 0でなければ、どこかに乱数か非決定的な並列化がある |
| FP32どうし（PyTorch と OV CPU） | 1e-4〜1e-3 | 演算順序の違い。正常 |
| FP32 と BF16 / FP16 | 1e-3〜1e-2 | 精度の違い。vla.cppでは、SmolVLAはこの1フラグだけで2e-3動く |
| FP32 と INT8 / Q4_0 | 1e-2〜1e-1 | 量子化誤差。成功率で影響を確かめる |
| 0.5以上、またはグリッパの符号反転が多い | — | **前処理のずれを疑う**（画像の向き、正規化、トークナイズの改行） |

### Step 6：仕上げ（10分）

- [ ] 4ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/policy scripts/check_equivalence.py` を通す
- [ ] `pytest tests/policy` がすべて通る
- [ ] コミットする

## 6. ベストプラクティス

- **抽象を小さく保つ**：`predict_chunk` 1つに絞る。ランタイム固有の機能（OVの非同期推論キューなど）を抽象に入れたくなったら、`extra` か別のメソッドで扱う。
- **遅延import**：`import openvino` や `import torch` はアダプタの中で行う。ベンチのプロセスに不要なライブラリが読み込まれると、メモリ測定（ピークRSS）が汚れる。
- **基準を1つに決める**：比較の基準は「PyTorch FP32＋固定ノイズ」。基準がぶれると、どの差が何に由来するのか分からなくなる。
- **ノイズの形に注意**：SmolVLAのノイズは `(chunk_size, max_action_dim)` = `(50, 32)` で、行動次元7ではない。出力は先頭7次元を切り出したもの。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| ランタイム間の誤差が0.5以上ある | 画像の値域（[0,1] か [0,255] か）、リサイズの補間方法、指示文の末尾の改行の有無 | 前処理を1段ずつ比較する（中間テンソルを保存して比べる） |
| 同じ入力で2回推論すると結果が違う | ノイズを固定していない。torchのスレッド並列の非決定性 | ノイズを渡す。`torch.use_deterministic_algorithms(True)` で確認する |
| vla.cppだけグリッパの次元がずれる | 正規化統計の扱い（サーバー側かクライアント側か）の違い | vla.cppの `vla_cpp_client.py` の `_predict_chunk` を読み、状態の正規化を誰が行うか確認する |
| 1視点にしたら出力が大きく変わる | 学習時と異なる入力。仕様どおりの挙動 | 一致の問題ではなく、成功率（モードB）で評価する |

## 8. ヒント

<details><summary>L1：前処理のずれを見つけるには</summary>

「入力 → リサイズ後の画像 → 正規化後のテンソル → トークン列 → 出力」の各段を、両方のランタイムで保存して比べる。最初にずれた段が原因。
</details>

<details><summary>L2：使う関数</summary>

`np.testing.assert_allclose`、`np.linalg.norm`、コサイン類似度＝`a.ravel() @ b.ravel() / (norm(a) * norm(b))`。グリッパの符号は `np.sign(ref[:, g]) != np.sign(test[:, g])` の数。
</details>

<details><summary>L3：check_equivalence の疑似コード</summary>

```
obs_set = load_fixed_obs(path)[:5]
noise = make_fixed_noise(50, 32, seed=0)
ref = make_policy(torch_fp32_cfg)
rows = []
for cfg in candidate_cfgs:
    pol = make_policy(cfg); pol.warmup(obs_set[0])
    for i, obs in enumerate(obs_set):
        r = ref.predict_chunk(obs, noise).actions      # 基準はキャッシュしてよい
        t = pol.predict_chunk(obs, noise).actions
        rows.append({**cfg_summary(cfg), "obs": i, **asdict(compare_chunks(r, t))})
    pol.close()
DataFrame(rows).groupby(cfgの列).agg(max) → Markdown表で出力
```
</details>

## 9. 完了条件（DoD）

- [ ] `pytest tests/policy` が通る
- [ ] `scripts/check_equivalence.py` が、少なくとも「PyTorch FP32 対 PyTorch FP32（2回目）」の行を出し、max_abs=0 になる（決定性の確認）
- [ ] 24・25章の完了時点で、全ランタイム・精度の一致表が `reports/equivalence_libero.md` にある

## 10. 理解度チェック・発展課題

- `Policy` に内部キューを持たせない理由を、非同期実行（27章）の観点から説明せよ。
- 固定ノイズを使って一致を確かめた後、本番の評価ではノイズを固定すべきか。成功率の推定の観点から考えよ。
- 発展：グリッパの次元は ±1 で飽和する出力なので、小さな誤差でも閾値をまたぐと動作が変わる。どの指標でそれを捉えるべきか。
