# track2/35. SmolVLAのファインチューニング（RTX PRO 6000）

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章に穴埋めはありません。学習コマンド（4.1節の雛形の `TODO(you)`）を自分で決め、学習を回して記録します
> - 自分で書くもの：`configs/train/smolvla_so101_v1.sh`（4.1節）、`scripts/offline_action_eval.py`（4.2節。任意だが、チェックポイントを選ぶ根拠になる）
> - ゴール：学習の記録（データの版・所要時間・損失曲線）があり、選んだチェックポイントをエッジの `TorchSmolVLA` で読み込める

## 1. 目的と計画書の対応

- 34章のデータセットで `lerobot/smolvla_base` をファインチューニングし、Gazebo SO-101用の方策を作る。
- 学習の再現性（設定、版、データの版）と、学習中の監視の方法を決める。
- 計画書の対応箇所：「トラック2（SmolVLAファインチューニング）」、日程D5「RTX PRO 6000で学習開始」、成果物「ファインチューニング済みSmolVLA」

## 2. 前提知識・事前に読む資料

- LeRobotのSmolVLAドキュメント：<https://huggingface.co/docs/lerobot/en/smolvla>
- `lerobot-train --help`（全オプション）
- SmolVLA論文の4章（学習の設定）
- 学習データの量の目安：論文用データは50エピソード（5位置×10）で、25エピソードでは不足だったと報告されている

### 2.1 ドキュメントにある基本コマンド

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

「A100 1枚で20kステップ約4時間」とされています。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `configs/train/smolvla_so101_v1.sh` | 学習コマンド（引数をすべて明示したシェル） |
| `docs/notes/train_v1.md` | 学習の記録（データの版、コミット、GPU、所要時間、損失曲線、気づいたこと） |
| `scripts/offline_action_eval.py` | 取り分けたエピソードで、行動の誤差（MSE）を計算する |
| `data/models/smolvla_so101_v1/`（エッジ側にコピー） | `pretrained_model/`（`config.json`、`model.safetensors`、`policy_preprocessor.json` など） |

## 4. インタフェース仕様

### 4.1 学習設定の雛形（値は自分で決める）

```bash
#!/usr/bin/env bash
set -euo pipefail
DATASET_ROOT=/path/to/data/lerobot/so101_pickplace_v1   # TODO(you)
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_pickplace_v1 \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=outputs/train/smolvla_so101_v1 \
  --job_name=smolvla_so101_v1 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --seed=1000 \
  --wandb.enable=true
  # TODO(you): 画像の拡張（--dataset.image_transforms.enable=true）を使うか
  # TODO(you): カメラのキー名が smolvla_base の想定と違う場合の rename_map（【要確認】lerobot-train --help）
```

**【要確認】** `--dataset.root`、`--save_freq`、`--seed` などの引数名は、使うLeRobotの版の `lerobot-train --help` で確認する。

### 4.2 データの分割

| 分割 | 割合 | 用途 |
| --- | --- | --- |
| 学習 | 約90% | `lerobot-train` |
| 取り分け | 約10%（例：エピソード番号が10の倍数） | オフラインの行動誤差（MSE）。Gazeboでの評価とは別 |

`lerobot-train` には、検証用の損失を出す機能が標準では無い（**【要確認】**）。取り分けたエピソードでの行動の誤差を、自作スクリプトでチェックポイントごとに計算します。

```python
# scripts/offline_action_eval.py
def eval_checkpoint(ckpt: Path, dataset_root: Path, episode_ids: Sequence[int],
                    n_samples: int = 200, seed: int = 0) -> dict[str, float]:
    """取り分けたエピソードからフレームをサンプリングし、予測チャンクと正解チャンク（同じ時刻からの50ステップ）の
    関節ごとのMSE[deg²]、グリッパの開閉の一致率を返す。"""
```

## 5. ハンズオン

各 Step は「**実行する → 出力を確かめる → `docs/notes/train_v1.md` に記録する**」の1サイクルです。学習は学習機（RTX PRO 6000）、確認はエッジ（この端末）で行います。

### Step 1：学習機の準備（1時間）

- エッジ側と**同じ版のLeRobot**を入れる（チェックポイントの互換性のため）。torchは学習機のGPU（Blackwell）に合うCUDA版にする。

```bash
nvidia-smi
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), lerobot.__version__)"
```

**確かめる**

- [ ] GPU 名と、`True`、LeRobot の版が表示される。版がエッジの `requirements/bench.lock.txt` と同じ

### Step 2：データの転送（30分）

```bash
rsync -avP data/lerobot/so101_pickplace_v1/ <学習機>:<置き場所>/so101_pickplace_v1/
find data/lerobot/so101_pickplace_v1 -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum   # データの版
```

HF の Hub に非公開で push する方法もあります。どちらの場合も、**データの版（ファイルのハッシュ、または Hub のコミット）を記録**します。

- [ ] 学習機側でも同じコマンドでハッシュを計算し、一致した

### Step 3：短い試運転（30分）

4.1節の雛形を `configs/train/smolvla_so101_v1.sh` に写し、`TODO(you)` を決めます（`lerobot-train --help` で引数名を確かめる）。まず `--steps=200` で回します。

```bash
bash configs/train/smolvla_so101_v1.sh --steps=200 --save_freq=200 --output_dir=outputs/train/smoke
ls outputs/train/smoke/checkpoints/
```

**確かめる**

- [ ] 損失が下がり始めている（wandb かログで）
- [ ] チェックポイントが保存される（`pretrained_model/` に `config.json`・`model.safetensors`・前処理の設定がある）
- [ ] 1ステップの時間から、本番（20k ステップ）の所要時間を見積もってメモに残した

### Step 4：本番の学習（数時間、D5の昼に開始）

```bash
bash configs/train/smolvla_so101_v1.sh
```

- [ ] wandb で損失を監視している。開始時刻・コミット・データの版・GPU を `docs/notes/train_v1.md` に書いた

### Step 5：チェックポイントの評価（1時間）

`scripts/offline_action_eval.py`（4.2節）で、5k・10k・15k・20k の各チェックポイントの行動の誤差を、取り分けたエピソードで計算します。

- [ ] チェックポイントごとの関節別 MSE と、グリッパの開閉の一致率の表がある
- [ ] 使うチェックポイントと、選んだ理由を記録した

### Step 6：エッジで読み込む（30分）

選んだチェックポイントの `pretrained_model/` を、エッジの `data/models/smolvla_so101_v1/` にコピーし、23章の `TorchSmolVLA` で読み込みます（`cfg.model` にローカルパスを渡す）。

```bash
source .venv-bench/bin/activate
python -c "
from evla.policy.base import PolicyConfig
from evla.policy.torch_policy import TorchSmolVLA
p = TorchSmolVLA(PolicyConfig(runtime='torch', model='data/models/smolvla_so101_v1/pretrained_model', views=('top', 'wrist')))
print(p.chunk_size, p.action_dim)
"
```

- [ ] `50 6` が表示される（SO-101 は6関節）

### Step 7：Gazebo で1エピソード回す（D8で本格的に評価する）

- [ ] 37章の推論サーバー（`--image_key_map '{"top": "top", "wrist": "wrist"}'`）経由で、Gazebo で1エピソード回し、腕がキューブに向かって動く

## 6. ベストプラクティス

- **学習コマンドはファイルにする**：シェルの履歴に頼らない。引数はすべて明示する（既定値が版によって変わるため）。
- **データの量と多様性を優先する**：ハイパーパラメータの調整より、データの範囲の偏りを直すほうが効く（33章の散布図を見る）。
- **過学習に注意する**：200エピソード×300フレーム＝6万フレームに対し、バッチ64×20kステップ＝128万サンプルで、約20周になる。取り分けたエピソードでの誤差が、途中から上がり始めないか見る。
- **学習の設定と結果を1か所に残す**：`train_config.json`（LeRobotが保存する）、使ったデータの版、gitのコミット、損失曲線のスクリーンショットを `docs/notes/train_v1.md` にまとめる。
- **エッジ側では学習機のことを気にしない**：エッジに持ってくるのは `pretrained_model/` だけ。前処理（正規化の統計）もその中に入っている。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `KeyError: observation.images.xxx` | データセットのカメラのキーと、方策の設定が合わない | rename_map で対応づける。データセット側を `top` / `wrist` で作ってあれば、そのまま方策の入力になる |
| エッジでチェックポイントを読み込めない | LeRobotの版の違い | 学習機とエッジで版を揃える |
| 損失は下がるのにGazeboで動かない | 前処理のずれ（画像の向き、単位）。行動の定義の違い | 34章の規約を確認する。取り分けたエピソードで、予測と正解の軌道を重ねて描く |
| 学習が遅い | データの読み込みがボトルネック（動画のデコード） | `--num_workers` を増やす。`dataset.video_backend` を確認する |
| GPUのメモリ不足 | バッチが大きすぎる | RTX PRO 6000（96GB）では起きにくいが、起きたらバッチを下げる |

## 8. ヒント

<details><summary>L1：オフラインの行動誤差は何を測っているか</summary>

「データセットと同じ状況で、教師と同じ行動を出せるか」を測っている。閉ループの成功率とは別物だが、学習が壊れていないことの確認と、チェックポイントの比較には使える。
</details>

<details><summary>L2：使うAPI</summary>

`LeRobotDataset(repo_id, root=..., episodes=[...], delta_timestamps={"action": [i / 30 for i in range(50)]})`（正解チャンクの取得）、23章の `TorchSmolVLA`、`torch.manual_seed`
</details>

<details><summary>L3：offline_action_eval の疑似コード</summary>

```
ds = LeRobotDataset(..., episodes=held_out, delta_timestamps={"action": [k/30 for k in range(50)]})
idx = rng.choice(len(ds), n_samples, replace=False)
pol = TorchSmolVLA(PolicyConfig(runtime="torch", model=str(ckpt)))
for i in idx:
    s = ds[i]; obs = to_observation(s)                 # 画像CHW float → HWC uint8、状態、task
    pred = pol.predict_chunk(obs, noise=fixed_noise).actions     # (50, 6)
    gt = s["action"].numpy()                                      # (50, 6)
    err.append(((pred - gt) ** 2).mean(axis=0))
return {"mse_per_joint": mean(err, axis=0), ...}
```
</details>

## 9. 完了条件（DoD）

- [ ] 学習コマンドのファイルと、学習の記録（データの版、所要時間、損失曲線）がある
- [ ] チェックポイントごとのオフラインの行動誤差の表があり、使うチェックポイントを選んだ理由が書かれている
- [ ] エッジの `.venv-bench` で `TorchSmolVLA` として読み込み、固定観測で推論できる

## 10. 理解度チェック・発展課題

- `smolvla_base` から始めることの利点は何か。ゼロから学習する場合と比べて、データの量の要求はどう変わるか。
- ロックステップで作ったデータで学習した方策を、実時間（遅延あり）で動かすと、何が起きると予想されるか。
- 発展：学習データに、意図的に「推論遅延のぶんだけ古い観測」を混ぜる（遅延を考慮した学習）と、38章の遅延ありの評価で成功率は上がるか。どう実装するか考えよ。
