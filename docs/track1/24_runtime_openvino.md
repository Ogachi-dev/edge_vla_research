# track1/24. OpenVINOランタイム（CPU / iGPU）とINT8

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**10個**（`src/evla/policy/ov_policy.py` 7、`scripts/ov_download.py` 1、`scripts/ov_inspect.py` 2）
> - ゴール：OpenVINO 版の SmolVLA が CPU と GPU の両方で推論でき、22章の一致表に OV の行が入る
> - テスト：モデルが必要なので pytest はなし。各 Step の「動かす」のコマンドで確かめる

## 1. 目的と計画書の対応

- 公開済みのOpenVINO版SmolVLA（LIBERO学習済み）を `Policy` として包み、CPUとiGPU（Iris Xe）で動かす。
- 精度（FP32実行 / FP16実行 / INT8重み）を掃引できるようにする。
- 計画書の対応箇所：トラック1「対象モデル（OpenVINO版）」「掃引軸（OpenVINO CPU / GPU、精度）」、日程D3、リスク「OpenVINO Physical AIのAPI変更」

## 2. 前提知識・事前に読む資料

- OpenVINOの基本（Core、read_model、compile_model、デバイスとプロパティ）：<https://docs.openvino.ai/2026/openvino-workflow/running-inference.html>
- 性能ヒントとスレッド：<https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html>
- 精度の制御（`INFERENCE_PRECISION_HINT`）：<https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference/precision-control.html>
- NNCFの重み圧縮：<https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html>
- OpenVINO Physical AI（`physicalai`）：<https://github.com/openvinotoolkit/physicalai>

### 2.1 調査で確認した事実

**【計画書との差分】** 計画書の `modelapi/smolvla-libero-fp16-ov-catalog` は現在アクセスできません（HTTP 401）。代わりに、Intelが次の3つを公開しています。いずれも `HuggingFaceVLA/smolvla_libero` から変換したものです。

| リポジトリ | 重みの形式 |
| --- | --- |
| `OpenVINO/smolvla-libero-fp16-ov` | FP16 |
| `OpenVINO/smolvla-libero-bf16-ov` | BF16 |
| `OpenVINO/smolvla-libero-w8a16-ov` | 重みINT8、活性16bit（重みのみの量子化） |

各リポジトリの中身：`smolvla.xml/.bin`（モデル本体）、`tokenizer.xml/.bin`（OpenVINO Tokenizers）、`manifest.json`、`config.json`

`manifest.json` から分かる構成：

- runnerは `SinglePass`（**10ステップのflow matchingを含む全体が1つのIR**になっている）
- 前処理：`smolvla_resize`（512×512）→ `new_line` → `normalize`（状態をmean_std）→ `ov_tokenizer`
- 後処理：`denormalize`（行動をmean_std）

`physicalai`（v0.3.0）の `InferenceModel`：

```python
InferenceModel(export_dir, policy_name=None, backend="auto", device="auto",
               runner=None, preprocessors=None, postprocessors=None, callbacks=None,
               **adapter_kwargs)   # ← adapter_kwargs はそのまま compile_model の config になる
model.predict_action_chunk(observation)  # → (chunk_size, action_dim)
model.reset()
```

観測のキー：`"images.image"`、`"images.image2"`（`(1, 3, H, W)` float32 [0,1]）、`"state"`（`(1, 8)`）、`"task"`（`list[str]`）

### 2.2 「重みの精度」と「実行の精度」は別物

| 用語 | 意味 | この研究での扱い |
| --- | --- | --- |
| IRの重み形式 | `.bin` に保存されている型（FP16 / BF16 / INT8） | モデルの選択で決まる |
| 実行精度 | 実際に演算する型。`INFERENCE_PRECISION_HINT` で決まる | デバイスとヒントで決まる |
| CPUの既定 | x86でAMXがなければ **f32で実行**（FP16の重みも展開してf32で計算する） | i7-1360Pは f32 |
| GPUの既定 | **f16で実行** | Iris Xeは f16 |

つまり「FP16のIRをCPUで動かす」のは、FP32で計算しているのと同じです。掃引表では**重み形式と実行精度の両方を列に持つ**ようにします。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/policy/ov_policy.py` | `OVSmolVLA(Policy)` |
| `scripts/ov_download.py` | 3つのOVモデルを `data/models/ov/` に取得する |
| `scripts/ov_inspect.py` | IRの入出力名・形、演算の種類（乱数演算の有無など）を表示する |
| `scripts/ov_compress.py`（発展） | NNCFで自分でINT8 / INT4の重み圧縮をする |
| （5節 Step 3〜5 のコマンド） | 形、CPU/GPU両方で動くことの確認。専用のテストは作らない |

## 4. インタフェース仕様

```python
# src/evla/policy/ov_policy.py
class OVSmolVLA(Policy):
    def __init__(self, cfg: PolicyConfig) -> None:
        """cfg.model: OVモデルのディレクトリ（manifest.json があるもの）
        cfg.device: "CPU" | "GPU"
        cfg.precision: 実行精度のヒント。"fp32" → INFERENCE_PRECISION_HINT="f32"、"fp16" → "f16"
        cfg.n_threads: INFERENCE_NUM_THREADS（CPUのみ）
        cfg.extra: 追加のcompile config（例 {"NUM_STREAMS": 1, "CACHE_DIR": "..."}）"""

    def _ov_config(self) -> dict[str, Any]:
        """PERFORMANCE_HINT="LATENCY" を基本に、cfg から compile config を組み立てる。"""

    def predict_chunk(self, obs: Observation, noise: npt.NDArray[np.float32] | None = None) -> ChunkResult:
        """noise を渡せない場合（IRに入力がない場合）、noise が None でなければ警告を1回だけ出して無視する。"""

    def describe(self) -> dict[str, Any]:
        """openvinoの版、FULL_DEVICE_NAME、実際の EXECUTION_DEVICES、INFERENCE_PRECISION_HINT、
        INFERENCE_NUM_THREADS（compiled_model.get_property で読む）、モデルの重み形式。"""
```

**主なcompile config**

| プロパティ | 値の例 | 意味 |
| --- | --- | --- |
| `PERFORMANCE_HINT` | `"LATENCY"` | バッチ1・ストリーム1での最短レイテンシを狙う |
| `INFERENCE_NUM_THREADS` | `8` | CPUのスレッド数 |
| `NUM_STREAMS` | `1` | 並列推論の数。LATENCYなら通常1 |
| `INFERENCE_PRECISION_HINT` | `"f32"` / `"f16"` / `"bf16"` | 実行精度 |
| `SCHEDULING_CORE_TYPE` | `"ANY_CORE"` / `"PCORE_ONLY"` / `"ECORE_ONLY"` | ハイブリッドCPUでのコア選択。**WSL2ではP/Eの区別が見えないので効かない可能性が高い**（[env/10](../env/10_wsl2_and_igpu.md)）。効くかどうかを実測して記録する |
| `CACHE_DIR` | `"data/ov_cache"` | GPUのコンパイル結果のキャッシュ。2回目以降の起動が速くなる |

`CACHE_DIR` は OpenVINO 本体の機能です。vla.cpp のOVバックエンドで問題になっている `GGML_OPENVINO_CACHE_DIR`（25章）とは別物ですが、念のため**キャッシュありとなしで行動が一致するか**を22章の指標で確かめます。

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。

### Step 0：準備（5分）

```bash
source .venv-bench/bin/activate
grep -n "【穴\|【要確認】" src/evla/policy/ov_policy.py scripts/ov_download.py scripts/ov_inspect.py
python -c "import openvino as ov; print(ov.__version__, ov.Core().available_devices)"
```

- [ ] 穴10個が表示され、デバイスに `GPU` が含まれる（無ければ [env/10](../env/10_wsl2_and_igpu.md) Step 3）

### Step 1：モデルを取得する ― `scripts/ov_download.py`（穴①、20分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `huggingface_hub` で、リポジトリのファイルを**全部**取ってくる関数（1ファイルだけ取る関数もあるので区別する） |

**動かす**

```bash
python scripts/ov_download.py
```

- [ ] 3つのリポジトリについて、`.xml` と `.bin`、`manifest.json` などが並ぶ

### Step 2：IR の中身を見る ― `scripts/ov_inspect.py`（穴①②、30分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | IR を読むだけ（まだコンパイルしない）。`ov.Core()` のメソッドを使う。コンパイルとの違い（デバイスに載せるかどうか）を意識する |
| ② | 演算の型名の出現回数を数える。`most_common` が使えるクラス |

**動かす**

```bash
ls data/models/ov/smolvla-libero-fp16-ov/*.xml
python scripts/ov_inspect.py data/models/ov/smolvla-libero-fp16-ov/<上で見つけた>.xml | tee results/smoke/ov_inspect.txt
```

**確かめる**

- [ ] 入力に画像（2視点）・状態・トークンなどが並ぶ。**形が固定か（`?` や `-1` がないか）**を見る（固定なら、視点数や解像度は変えられない）
- [ ] `== 乱数演算:` の行を見て、**ノイズの扱いを決める**。乱数演算があれば固定ノイズは渡せないので、22章の比較は「同じ観測でN=20回推論した平均どうし」で行う。決めたことを `docs/decisions.md` に残す

### Step 3：CPU で動かす ― `ov_policy.py` の穴①〜⑥（1時間）

**読む**：4節の「主なcompile config」の表。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 辞書を `key=value` のキーワード引数として展開して渡す Python の構文 |
| ② | OpenVINO の性能ヒント。バッチ1で最短の応答を狙う方（もう1つは throughput） |
| ③ | CPU のスレッド数のプロパティ名（表を見る） |
| ④ | NumPy でバッチ次元を足す（23章の `unsqueeze(0)` の NumPy 版）。インデックスに書ける特別な値がある |
| ⑤ | 指示文を長さ1のリストにする |
| ⑥ | `actions.shape` をそのまま2つの変数に分ける |

**動かす**

```bash
python - <<'EOF'
from pathlib import Path
from evla.common.config import load_config
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy
from evla.sim.fixed_obs import load_fixed_obs

obs = load_fixed_obs(Path("data/fixed_obs/libero_object_t0.npz"))[0]
with make_policy(load_config(Path("configs/policy/ov_cpu_fp16.yaml"), PolicyConfig)) as p:
    p.warmup(obs)
    r = p.predict_chunk(obs)
    print(r.actions.shape, round(r.latency_ms, 1), "ms", p.chunk_size, p.action_dim)
EOF
```

**確かめる**

- [ ] `(50, 7)` と推論時間が表示される。23章の PyTorch（同じスレッド数）と比べてメモに残す
- [ ] 22章の Step 5 で `--test configs/policy/ov_cpu_fp16.yaml` を足し、`max_abs` が目安（FP32どうしで1e-4〜1e-3）に収まるか確かめる

### Step 4：describe で「実際に何で動いたか」を残す ― 穴⑦（20分）

設定に書いた値と、OpenVINO が実際に使った値は違うことがあります（例：GPU に `f32` を頼んでも `f16` で動く）。結果ファイルには**実際の値**を残します。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ⑦ | デバイスの正式名（`Intel(R) Iris(R) Xe Graphics` など）を表すプロパティ名 |

**動かす**：Step 3 のコードの最後に `import pprint; pprint.pprint(p.describe())` を足して実行する。

- [ ] `full_device_name` と、`EXECUTION_DEVICES`・`INFERENCE_PRECISION_HINT`・`INFERENCE_NUM_THREADS` が表示される
- [ ] これらが `None` なら、`_find_compiled_model` の【要確認】（physicalai の内部の属性名）を、`print(vars(p.model))` で調べて直す

### Step 5：GPU で動かす（1時間）

```bash
rm -rf data/ov_cache
for i in 1 2; do
  python -c "
from pathlib import Path; import time
from evla.common.config import load_config
from evla.policy.base import PolicyConfig
from evla.policy.ov_policy import OVSmolVLA
from evla.sim.fixed_obs import load_fixed_obs
obs = load_fixed_obs(Path('data/fixed_obs/libero_object_t0.npz'))[0]
p = OVSmolVLA(load_config(Path('configs/policy/ov_gpu_fp16.yaml'), PolicyConfig))
t = time.perf_counter(); p.predict_chunk(obs); first = time.perf_counter() - t
print('load_s=%.1f first_infer_s=%.1f' % (p.load_s, first), p.describe().get('EXECUTION_DEVICES'))
"
done
```

**確かめる**

- [ ] `EXECUTION_DEVICES` に `GPU` が出る（`CPU` なら GPU で動いていない）
- [ ] 1回目（キャッシュなし）と2回目（`data/ov_cache` あり）の `load_s` を記録する。2回目が大きく短くなる
- [ ] 22章の一致表に `ov_gpu_fp16` の行を足す。キャッシュありとなしで行動が変わらないことも確かめる

### Step 6：閉ループと、残りの組み合わせ（1〜2時間）

- [ ] 23章 Step 4 の簡易ループで、設定ファイルを `configs/policy/ov_cpu_fp16.yaml` に替えて1エピソード成功させる
- [ ] bf16 / w8a16 のモデル（`model:` を替えた設定ファイルを自分で作る）× CPU / GPU を、22章の一致表に足す
- [ ] （発展）`nncf.compress_weights` で、FP16 の IR から INT8_SYM / INT8_ASYM / INT4_SYM の重み圧縮版を作り、公開版（w8a16）と比べる

**入力条件（視点数・解像度）の掃引について**：OVのIRは入力の形が固定されている可能性が高いです（Step 2 で確認）。1視点や256pxの実験をOVで行うには、`physicalai-train` で設定を変えて書き出し直す必要があります（36章の手順を流用）。**【判断】** トラック1では、入力条件の掃引は PyTorch と vla.cpp で行い、OVは精度×デバイス×スレッドに絞る。時間があれば書き出し直しも試す。

### Step 7：仕上げ（10分）

- [ ] 3ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/policy/ov_policy.py scripts/ov_*.py` を通す
- [ ] コミットする

## 6. ベストプラクティス

- **実際の設定を読み戻して記録する**：渡したconfigではなく、`compiled_model.get_property(...)` で**実際に適用された値**を記録する。OpenVINOは、指定がデバイスに合わないと黙って別の値にすることがある。
- **GPUのコンパイル時間は別の指標にする**：初回の推論時間に混ぜない。ロボットの起動時間として、別に報告する価値がある。
- **重み形式ごとにメモリを測る**：w8a16はメモリを大きく減らすはず。メモリはエッジでの重要な制約なので、レイテンシと並べて報告する。
- **API変更に備える**：physicalaiはv0.x系で、変更が速い。アダプタの中でphysicalaiに依存する部分を1か所に閉じ込め、版を固定する（計画書のリスク表の「API変更」への対策）。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `device='GPU'` で `no supported devices found` | compute runtimeが見えていない | [env/10](../env/10_wsl2_and_igpu.md) の3段階の確認 |
| GPUの初回推論に数分かかる | カーネルのコンパイル | ウォームアップで除外する。`CACHE_DIR` |
| FP16のIRなのにCPUでのメモリ使用量がFP32並み | CPUはf32で実行するため、重みを展開している | 仕様どおり。表に実行精度を書く |
| GPUでの行動がCPUと1e-2程度ずれる | GPUはf16で実行する | `INFERENCE_PRECISION_HINT="f32"` で比べて切り分ける |
| `SCHEDULING_CORE_TYPE` を変えても結果が同じ | WSL2ではハイブリッド構成が見えない | 効かなかった事実を記録する |
| physicalaiの新しい版でキーや関数名が変わった | 開発途中のAPI | 版を固定する。変更点を `docs/decisions.md` に記録する |

## 8. ヒント

<details><summary>L1：IRの入力を調べる</summary>

OpenVINOのモデルオブジェクトは、入力と出力の一覧（名前、形、型）を持っている。形に `?` や `-1` があれば動的な次元。演算の一覧も取れるので、乱数を生成する演算が含まれているか分かる。
</details>

<details><summary>L2：使うAPI</summary>

`ov.Core().read_model(path)`、`model.inputs` / `model.outputs`（`.get_any_name()`、`.get_partial_shape()`、`.get_element_type()`）、`model.get_ordered_ops()` → `op.get_type_name()`、`compiled.get_property("EXECUTION_DEVICES")`、`core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")`、`nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM)`、`ov.save_model(model, path)`
</details>

<details><summary>L3：OVSmolVLA.predict_chunk の疑似コード</summary>

```
t0 = now()
o = {f"images.{k}": hwc_uint8_to_nchw_float01(obs.images[k]) for k in self.cfg.views}
o["state"] = obs.state[None, :]
o["task"] = [obs.task]
t1 = now()
actions = self.model.predict_action_chunk(o)        # (50, 7)
t2 = now()
return ChunkResult(actions.astype(np.float32), ms(t2 - t0), {"preprocess": ms(t1 - t0), "inference": ms(t2 - t1)})
```
</details>

## 9. 完了条件（DoD）

- [ ] `ov_inspect.py` の出力（入出力の形、乱数演算の有無）がメモにあり、ノイズの扱いが決まっている
- [ ] CPUとGPUの両方で推論でき、`describe()` が実際のデバイス名と実行精度を返す
- [ ] GPUの初回コンパイル時間（キャッシュありとなし）を記録した
- [ ] 22章の一致表にOVの行がある
- [ ] 簡易ループで1エピソード以上成功した（少なくともCPUで）

## 10. 理解度チェック・発展課題

- 「w8a16」と「W8A8」の違いは何か。それぞれ、どの資源（メモリ帯域、演算器）を節約するか。
- Iris Xe（96EU）がCPUより速くなる条件と、遅くなる条件を予想せよ（ヒント：メモリ帯域を共有している）。
- 発展：NNCFの `nncf.quantize`（W8A8の量子化）には校正データが必要。LIBEROの固定観測から校正データを作る手順を設計せよ。IRの入力が前処理後のテンソルであることに注意。
