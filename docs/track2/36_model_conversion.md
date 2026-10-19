# track2/36. 学習済みモデルの変換（OpenVINO IR / GGUF）と数値一致の検証

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`scripts/export_ov.py`（3）、`scripts/export_gguf.sh`（1）
> - テスト：なし（「9. 完了条件」で確認）
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

## 1. 目的と計画書の対応

- 35章のファインチューニング済みSmolVLAを、トラック1で選んだランタイム（OpenVINO または vla.cpp）で動く形式に変換する。
- 変換の前後で**行動が一致するか**を、22章の仕組みで検証する。
- 計画書の対応箇所：「トラック2（OpenVINO / GGUFへ変換、トラック1の最適設定を適用）」、日程D8、リスク「OpenVINO Physical AIのAPI変更」

## 2. 前提知識・事前に読む資料

- 24章（OpenVINOの重み形式と実行精度）、25章（GGUFの量子化）、22章（数値一致）
- Physical AI のエクスポート：<https://github.com/open-edge-platform/physical-ai-studio>（`library/docs/how-to/export/export_inference.md`）
- vla.cpp の `docs/MODELS.md`（`scripts/convert_smolvla_to_gguf.py`、`scripts/quantize_gguf.py`）

### 2.1 調査で確認した事実

**OpenVINO（physicalai-train 0.3.0）**

- `physicalai.policies.SmolVLA` は `pretrained_name_or_path` を受け取り、HFのrepoまたはローカルのディレクトリから、LeRobot形式の `config.json`、`model.safetensors`、`policy_preprocessor.json` を読み込める（`_from_hf`）。
- `policy.export(out_dir, backend="openvino", post_export_hooks=[...])` で書き出す。組み込みのフック `physicalai.export.compress_weights_openvino_int8_sym` を渡すと、NNCFでINT8対称の重み圧縮をする（`pip install physicalai-train[nncf]` が必要）。
- 書き出したディレクトリは `physicalai.inference.InferenceModel` で読み込める。

**GGUF（vla.cpp）**

```bash
python scripts/convert_smolvla_to_gguf.py --ckpt /path/to/pretrained_model --out smolvla-so101-bf16.gguf
python scripts/quantize_gguf.py --in smolvla-so101-bf16.gguf --out smolvla-so101-q8_0.gguf --type Q8_0
# --type は Q8_0 / Q4_0 / Q4_1 / Q5_0 / Q5_1。--vision でビジョン部も量子化する
```

vla.cpp の README には `smolvla-so101.gguf` の例があり、SO-101（6次元）での利用が想定されています。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `scripts/export_ov.py`（`.venv-export` で実行） | LeRobotのチェックポイント → OpenVINO IR（FP16、INT8重み） |
| `scripts/export_gguf.sh` | LeRobotのチェックポイント → GGUF（BF16、Q8_0、Q4_0） |
| `scripts/check_equivalence_so101.py` | 取り分けたエピソードのフレームで、PyTorchと変換後の行動を比較する |
| `data/models/so101_v1/{torch,ov_fp16,ov_int8,gguf_bf16,gguf_q8_0,...}/` | 変換結果 |
| `reports/equivalence_so101.md` | 一致の表 |

## 4. インタフェース仕様

```python
# scripts/export_ov.py
def export_openvino(ckpt_dir: Path, out_dir: Path, int8_weights: bool) -> Path:
    """physicalai の SmolVLA で LeRobot のチェックポイントを読み、OpenVINO IR を書き出す。
    書き出し後に InferenceModel で読み込めることを確認し、入出力の形を表示する。"""
```

```python
# scripts/check_equivalence_so101.py
def frames_as_observations(dataset_root: Path, episodes: Sequence[int], n: int, seed: int) -> list[Observation]:
    """LeRobotDataset からフレームを取り出し、Observation（HWC uint8、state (6,) [deg]、task）に変換する。"""

def main(cfgs: Sequence[PolicyConfig], ref: PolicyConfig, obs: list[Observation], noise_seed: int) -> pd.DataFrame:
    """22章の compare_chunks で、各設定を基準（PyTorch FP32）と比べる。"""
```

## 5. 実装手順

1. **書き出しの試行（1〜2時間）**：`.venv-export` で、まず**公開のLIBERO版**（`HuggingFaceVLA/smolvla_libero`）を書き出し、Intelの公開IR（`OpenVINO/smolvla-libero-fp16-ov`）と同じ結果になるかを確かめる。手順が正しいことを、既知の答えで先に確認する。
2. **SO-101版の書き出し（1時間）**：35章のチェックポイントで同じことをする。INT8重み版も作る。
3. **GGUFの変換（1時間）**：vla.cppの変換用venvで、BF16 → Q8_0 / Q4_0（＋ `--vision`）を作る。`vla-server` で読み込めて、`action_dim` が6であることを応答で確かめる。
4. **一致の確認（1〜2時間）**：取り分けたエピソードから20フレームを取り出し、固定ノイズで、各変換結果をPyTorch FP32と比べる。
5. **トラック1の最適設定を適用する**：28章で選んだ設定（ランタイム、精度、スレッド数、視点、解像度）に対応する変換結果を、37章の推論サーバーで使う。

## 6. ベストプラクティス

- **既知の答えで手順を検証してから、本番に使う**：LIBERO版で公開IRと一致することを確認してから、自分のモデルを変換する。どちらも初めてだと、ずれたときに原因が手順なのかモデルなのか分からない。
- **変換の環境を分ける**：physicalai-train は特定のLeRobotの版を要求する。vla.cppの変換器にも独自の依存がある。それぞれ専用のvenvで行う（[01](../01_repository_design.md)）。
- **変換の記録を残す**：入力のチェックポイント（ハッシュ）、ツールの版、コマンド、出力のハッシュを、出力ディレクトリの `convert_log.json` に書く。
- **一致の表を報告書に入れる**：Gazeboでの成功率の差を、「変換で壊れた」のではなく「精度の差による」と主張するための根拠になる。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| physicalai でチェックポイントの読み込みに失敗する | LeRobotの版の違いによるキー名の差 | physicalai-train が要求する版（0.6.0）と、学習に使った版の差を確認する。`fix_state_dict_keys` の実装を読む |
| 書き出したIRで、視点の数や解像度が変えられない | 入力の形が固定されている | 入力条件の掃引はPyTorchとvla.cppで行う（24章の【判断】） |
| GGUFの行動がPyTorchと大きく違う | 状態の次元（6）と、最大の次元（32）の扱い、正規化の統計 | 応答の `action_dim` と、クライアントでの切り出しを確認する |
| INT8版でグリッパの開閉が反転する | 量子化誤差で、飽和する出力が閾値をまたいだ | 22章の `gripper_sign_flips` で数える。成功率への影響は38章で評価する |

## 8. ヒント

<details><summary>L1：書き出し結果が正しいかの確かめ方</summary>

同じ観測と同じノイズを、変換前と変換後の両方に入れて、出てきた行動チャンクを比べる。ノイズを入力できない形式なら、同じ観測で多数回推論した平均どうしを比べる。
</details>

<details><summary>L2：使うAPI</summary>

`from physicalai.policies import SmolVLA`、`SmolVLA(pretrained_name_or_path=str(ckpt))`、`policy.export(str(out), backend="openvino", post_export_hooks=[compress_weights_openvino_int8_sym])`、`from physicalai.inference import InferenceModel`、`hashlib.sha256`
</details>

<details><summary>L3：export_openvino の疑似コード</summary>

```
policy = SmolVLA(pretrained_name_or_path=str(ckpt_dir)); policy.eval()
hooks = [compress_weights_openvino_int8_sym] if int8_weights else []
policy.export(str(out_dir), backend="openvino", post_export_hooks=hooks)
m = InferenceModel(str(out_dir), backend="openvino", device="CPU")
print(m.chunk_size); 入出力の形を表示
write_json(out_dir / "convert_log.json", {"ckpt_sha256": ..., "physicalai_train": version, ...})
```
</details>

## 9. 完了条件（DoD）

- [ ] LIBERO版の書き出しで、自分の手順の出力と公開IRの行動が一致した（または差とその理由が説明できる）
- [ ] SO-101版のOpenVINO IR（FP16、INT8重み）とGGUF（BF16、Q8_0、Q4_0）がある
- [ ] `reports/equivalence_so101.md` に一致の表がある
- [ ] トラック1の最適設定に対応する変換結果を、37章の推論サーバーで読み込めた

## 10. 理解度チェック・発展課題

- 重みだけのINT8化（w8a16）は、CPUとiGPUのどちらで効果が大きいと予想されるか。理由は。
- 変換の一致が良い（max_abs < 1e-3）のに、成功率が下がることはありうるか。どんな場合か。
- 発展：ビジョン部だけFP16のまま、言語モデルだけを量子化する、のように部分ごとに精度を変えると何が得られるか。vla.cppの `--vision` の有無で確かめよ。
