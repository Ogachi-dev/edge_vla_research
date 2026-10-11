# track1/24. OpenVINOランタイム（CPU / iGPU）とINT8

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
| `tests/policy/test_ov_policy.py` | 形、CPU/GPU両方で動くこと（GPUがなければskip） |

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

## 5. 実装手順

1. **取得と確認（1時間）**
   - `huggingface_hub.snapshot_download` で3モデルを `data/models/ov/<name>/` に保存する。
   - `ov_inspect.py`：`ov.Core().read_model("smolvla.xml")` の `inputs` と `outputs`（名前・形・型）を表示する。`get_ordered_ops()` の型名を数え、`RandomUniform` などの乱数演算があるかを確認する。
   - **ノイズの扱いを決める**：IRにノイズの入力があれば固定ノイズで比較できる。なければ（IR内で乱数を生成していれば）、22章の比較は「同じ観測でN=20回推論した平均どうし」で行う。
2. **CPUで動かす（1時間）**：`InferenceModel(dir, backend="openvino", device="CPU", PERFORMANCE_HINT="LATENCY")` で固定観測を推論する。
3. **アダプタの実装（2時間）**：`Observation` → physicalai の観測辞書への変換、`_ov_config`、`describe`。
4. **GPUで動かす（1時間）**：`device="GPU"` にする。初回のコンパイル時間を別に測って記録する（`CACHE_DIR` ありとなしの両方）。
5. **一致の確認（1時間）**：22章の `check_equivalence.py` に、OVの各設定（fp16 / bf16 / w8a16 × CPU / GPU）の行を追加する。
6. **閉ループの確認（1時間）**：23章と同じ簡易ループで1エピソード成功させる。
7. **（発展）自前の重み圧縮**：`nncf.compress_weights` で、FP16のIRから INT8_SYM / INT8_ASYM / INT4_SYM の重み圧縮版を作り、公開版（w8a16）と比較する。

**入力条件（視点数・解像度）の掃引について**：OVのIRは入力の形が固定されている可能性が高いです。1視点や256pxの実験をOVで行うには、`physicalai-train` で設定を変えて書き出し直す必要があります（36章の手順を流用）。**【判断】** トラック1では、入力条件の掃引は PyTorch と vla.cpp で行い、OVは精度×デバイス×スレッドに絞る。時間があれば書き出し直しも試す。

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
