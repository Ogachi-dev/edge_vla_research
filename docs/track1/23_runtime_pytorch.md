# track1/23. PyTorch CPUベースライン（LeRobotのSmolVLA）

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
| `scripts/torch_smoke.py` | 固定観測で1回推論し、行動の形と時間を表示する |
| `tests/policy/test_torch_policy.py` | 形、決定性（固定ノイズ）、1視点で動くこと |

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

## 5. 実装手順

1. **読み込み（1時間）**
   - `SmolVLAPolicy.from_pretrained(cfg.model)` でモデルを読む。`.eval()`、`.to("cpu")`。
   - `make_pre_post_processors(policy.config, pretrained_path=cfg.model, preprocessor_overrides={"device_processor": {"device": "cpu"}})` で前処理と後処理を作る。
2. **スレッド設定（30分）**
   - `torch.set_num_threads(n)` を、**モデルを読む前に**呼ぶ。`torch.set_num_interop_threads(1)` も設定する（interopは起動直後に1回しか設定できない）。
   - 環境変数 `OMP_NUM_THREADS` もスクリプトの冒頭（torchのimport前）で設定し、両者を一致させる。
3. **`_to_batch` と `predict_chunk`（2時間）**
   - `torch.inference_mode()` の中で実行する。
   - ノイズは `(1, 50, 32)` のtensorにして渡す。
   - 出力は `(1, 50, 7)` → `(50, 7)` のNumPyにする。
4. **スモークテスト（30分）**：固定観測で推論し、`actions.shape == (50, 7)` で値が [-1, 1] 付近であることを確かめる。
5. **閉ループでの確認（1時間）**：21章のラッパで1エピソード回し、チャンクの先頭10行動ずつ実行して成功することを確認する（簡易ループ。本格的な実装は27章）。
6. **スレッド数の予備測定（1時間）**：`n_threads` を 1, 2, 4, 8, 12, 16 と変え、各20回の推論時間を測る（本格的な測定は26章）。

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
