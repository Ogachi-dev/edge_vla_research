# track2/36. 学習済みモデルの変換（OpenVINO IR / GGUF）と数値一致の検証

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**4個**（`scripts/export_ov.py` 3、`scripts/export_gguf.sh` 1）
> - ゴール：SO-101 版の OpenVINO IR（FP16、INT8 重み）と GGUF（BF16、Q8_0、Q4_0）ができ、PyTorch との一致の表が `reports/equivalence_so101.md` にある
> - 進め方：**答えの分かっている LIBERO 版で手順を確かめてから**（Step 1）、SO-101 版に同じ手順を当てる（Step 2〜）

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
| `scripts/check_equivalence_so101.py`（自分で作る） | 取り分けたエピソードのフレームで、PyTorchと変換後の行動を比較する（22章の `scripts/check_equivalence.py` を写し、観測を LeRobotDataset から作る） |
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

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。変換は「動いたように見えて、実は行動がずれている」ことが一番怖いので、**既知の答えと比べる**ことを繰り返します。

### Step 0：準備（5分）

```bash
grep -n "【穴" scripts/export_ov.py scripts/export_gguf.sh
ls outputs/train/so101_v1/checkpoints/last/pretrained_model/     # 35章の学習結果（Step 2 から使う）
```

- [ ] 穴4個が表示される
- [ ] `.venv-export`（[env/12](../env/12_inference_envs.md) Step 2）と、vla.cpp の変換用 venv（25章 Step 4）がある

### Step 1：LIBERO 版で手順を確かめる ― `scripts/export_ov.py`（穴①〜③、1〜2時間）

**読む**：4節の書き出しの流れ（physicalai の `SmolVLA` → `export` → `InferenceModel` で読み込めるか確認）。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | ハッシュに少しずつデータを足すメソッド（01章の `hashlib` と同じ仲間）。大きなファイルを一度にメモリに載せない |
| ② | physicalai の `SmolVLA` に、HF の repo かローカルのディレクトリを渡す引数名（ソースか `help(SmolVLA)` で調べる） |
| ③ | 書き出し先の形式の名前（`InferenceModel` の `backend` と同じ） |

**動かす**

```bash
source .venv-export/bin/activate
python scripts/export_ov.py --ckpt HuggingFaceVLA/smolvla_libero --out data/models/ov_check/libero_fp16
deactivate
source .venv-bench/bin/activate
sed 's#data/models/ov/smolvla-libero-fp16-ov#data/models/ov_check/libero_fp16#' configs/policy/ov_cpu_fp16.yaml > /tmp/ov_mine.yaml
python scripts/check_equivalence.py --fixed_obs data/fixed_obs/libero_object_t0.npz \
  --ref configs/policy/ov_cpu_fp16.yaml --test /tmp/ov_mine.yaml --n 5 --out reports/equivalence_export_check.md
```

**確かめる**

- [ ] `exported:` に IR のファイル一式が並び、`convert_log.json` に元のチェックポイントのハッシュと physicalai の版が残る
- [ ] 自分で書き出した IR と、Intel の公開 IR の行動の `max_abs` が小さい（24章 Step 2 で「乱数演算あり」と分かっていれば、N回推論した平均どうしで比べる）。差があれば理由をメモに残す

### Step 2：SO-101 版を書き出す（1時間）

```bash
source .venv-export/bin/activate
CKPT=outputs/train/so101_v1/checkpoints/last/pretrained_model
python scripts/export_ov.py --ckpt $CKPT --out data/models/so101_v1/ov_fp16
python scripts/export_ov.py --ckpt $CKPT --out data/models/so101_v1/ov_int8 --int8
deactivate
du -sh data/models/so101_v1/ov_*
```

- [ ] `ov_fp16` と `ov_int8` ができ、INT8 の方が小さい

### Step 3：GGUF に変換する ― `scripts/export_gguf.sh`（穴①、1時間）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 変換スクリプトに変換元のチェックポイントを渡す引数（`python third_party/vla.cpp/scripts/convert_smolvla_to_gguf.py --help` で見る） |

**動かす**

```bash
source third_party/vla.cpp/.venv-converter/bin/activate
bash scripts/export_gguf.sh outputs/train/so101_v1/checkpoints/last/pretrained_model so101_v1
deactivate
```

- [ ] `data/models/so101_v1/` に `gguf_bf16.gguf`・`gguf_q8_0.gguf`・`gguf_q4_0.gguf` と `sha256.txt` ができる
- [ ] 25章 Step 3 の Python で `gguf` を `gguf_bf16.gguf` に替えてサーバーを起動し、応答の `action_dim` が **6**（SO-101 の関節数）であることを確かめる

### Step 4：一致の表を作る（1〜2時間）

`scripts/check_equivalence.py` を `scripts/check_equivalence_so101.py` に写し、観測を **取り分けたエピソード**（学習に使っていないもの）の LeRobotDataset から20フレーム作るように書き換えます。PyTorch FP32（`runtime: torch`、`model: <チェックポイント>`）を基準にして、変換したすべての形式と比べます。

- [ ] `reports/equivalence_so101.md` に、OV（FP16 / INT8）と GGUF（BF16 / Q8_0 / Q4_0）の行がある
- [ ] 値が22章の目安の表の範囲に入っている。外れていれば、前処理（画像の向き・カメラ名・状態の単位）を疑う

### Step 5：トラック1の最適設定を当てる（30分）

- [ ] 28章の `reports/track1_selection.md` で選んだ設定（ランタイム・精度・スレッド数・視点・解像度）に対応する変換結果を決め、37章の推論サーバーの設定ファイルに書く
- [ ] `export_ov.py` の先頭の `from evla._blank import ___` を消し、コミットする（`data/models/` は git の管理外）

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
