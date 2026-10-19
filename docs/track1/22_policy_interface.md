# track1/22. ランタイム共通のPolicy抽象と数値一致の検証

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`src/evla/policy/factory.py`（1）、`src/evla/policy/preprocess.py`（5）、`src/evla/policy/equivalence.py`（6）、`scripts/check_equivalence.py`（2）
> - テスト：`pytest tests/policy`
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

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

## 5. 実装手順

1. `base.py` を書く（インタフェースだけ）。
2. `equivalence.py` を実装し、テストを書く（同一配列でmax_abs=0・cosine=1、符号反転でcosine=-1、など）。
3. `factory.py` を書く。最初は `torch` だけ対応し、他は `NotImplementedError` にする。
4. 23章でPyTorchのアダプタができたら、`scripts/check_equivalence.py` を作る。
   - 固定観測（21章）×固定ノイズ（seed=0）で、PyTorch FP32 を基準（ref）とする。
   - 各ランタイム・精度の出力と比較し、`EquivalenceReport` を表にする。
5. 24・25章でアダプタが増えるたびに、この表に行を足す。

**比較の目安（vla.cppのOVバックエンド文書の値を参考にした）**：

| 比較 | max_abs の目安 | 解釈 |
| --- | --- | --- |
| 同じランタイムで2回 | 0（決定的なら） | 0でなければ、どこかに乱数か非決定的な並列化がある |
| FP32どうし（PyTorch と OV CPU） | 1e-4〜1e-3 | 演算順序の違い。正常 |
| FP32 と BF16 / FP16 | 1e-3〜1e-2 | 精度の違い。vla.cppでは、SmolVLAはこの1フラグだけで2e-3動く |
| FP32 と INT8 / Q4_0 | 1e-2〜1e-1 | 量子化誤差。成功率で影響を確かめる |
| 0.5以上、またはグリッパの符号反転が多い | — | **前処理のずれを疑う**（画像の向き、正規化、トークナイズの改行） |

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
- [ ] 24・25章の完了時点で、全ランタイム・精度の一致表が `reports/equivalence.md` にある

## 10. 理解度チェック・発展課題

- `Policy` に内部キューを持たせない理由を、非同期実行（27章）の観点から説明せよ。
- 固定ノイズを使って一致を確かめた後、本番の評価ではノイズを固定すべきか。成功率の推定の観点から考えよ。
- 発展：グリッパの次元は ±1 で飽和する出力なので、小さな誤差でも閾値をまたぐと動作が変わる。どの指標でそれを捉えるべきか。
