# track1/25. vla.cppランタイム（GGUF：CPU / SYCL / OpenVINOバックエンド）

## 1. 目的と計画書の対応

- vla.cppの `vla-server` をサブプロセスとして管理し、ZeroMQ＋protobufで通信する `Policy` アダプタを作る。
- GGUFの量子化（Q8_0 / Q4_0）とバックエンド（CPU / SYCL / OpenVINO）を掃引できるようにする。
- 計画書の対応箇所：トラック1「掃引軸（vla.cpp CPU / SYCL、Q8_0、Q4_0）」「上限確認（π0 / GR00T）」、日程D3、D8

## 2. 前提知識・事前に読む資料

- vla.cppのREADME、`docs/USAGE.md`、`docs/MODELS.md`、`docs/backend/sycl.md`、`docs/backend/ov.md`、`docs/benchmark/README.md`
- vla.cppのPythonクライアント（読解用）：`eval/client/vla_cpp_client.py` の `VlaCppClient.__init__` と `_predict_chunk`
- ZeroMQのREQ/REP：<https://zguide.zeromq.org/docs/chapter3/>（"Lazy Pirate" パターン＝タイムアウト時の回復）
- protobuf（Python）：<https://protobuf.dev/getting-started/pythontutorial/>
- GGUFとggmlの量子化：Q8_0、Q4_0は「32要素ごとに1つのスケールを持つブロック量子化」

### 2.1 調査で確認した事実

**通信仕様（`src/serving/vla.proto`）**

```protobuf
message Image { enum Encoding { JPEG = 0; RGB_U8 = 1; F32_RGB_01 = 2; }
                Encoding encoding = 1; uint32 height = 2; uint32 width = 3; bytes data = 4; }
message PredictRequest {
  repeated Image images = 1;  repeated int32 lang_tokens = 2;  repeated float state = 3;
  repeated float noise = 4;   uint64 request_id = 5;  /* 6〜8は省略 */ }
message PredictResponse {
  uint64 request_id = 1;  repeated float action_chunk = 2;  uint32 chunk_size = 3;  uint32 action_dim = 4;
  float latency_ms_total = 5;  float latency_ms_inference = 6;  float latency_ms_prefill = 8;
  float latency_ms_denoise = 9;  float latency_ms_vision = 10;  string error = 7; }
```

**SmolVLAのときのクライアント側の責任**（`vla_cpp_client.py` の `_predict_chunk` より）

| 項目 | クライアントがすること |
| --- | --- |
| 画像 | CHW float32 [0,1] を、**512×512にパディング付きリサイズ**（pad=0）→ HWCにして `F32_RGB_01` で送る |
| 指示文 | 末尾に `"\n"` を付け、`HuggingFaceTB/SmolVLM2-500M-Instruct` のトークナイザで、パディングなし・最大長で切り詰めてトークン化し、`lang_tokens` で送る |
| 状態 | **正規化せず**に、`max_state_dim=32` までゼロ埋めして送る（正規化はサーバー側＝GGUF内の統計で行う） |
| 応答 | `action_chunk` を `(chunk_size, action_dim)` に整形し、先頭の実次元（LIBEROなら7）を使う |
| ソケット | `zmq.REQ`、`LINGER=0`、`RCVTIMEO`、`REQ_RELAXED=1`、`REQ_CORRELATE=1` |

**その他**：`vla-server` はリクエストを1件ずつ同期的に処理する。既定の待ち受けは `tcp://*:5555`。起動時に `vla-server: bound to tcp://*:5555. ready.` と出力する。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/policy/vlacpp_server.py` | `VlaServerProcess`：vla-serverの起動・準備完了待ち・バナー検証・終了 |
| `src/evla/policy/vlacpp_policy.py` | `VlaCppPolicy(Policy)`：ZeroMQクライアント |
| `scripts/vlacpp_quantize.sh` | GGUFの量子化版（Q8_0、Q4_0、`--vision` あり/なし）を作る |
| `scripts/vlabench_reference.sh` | `vla-bench` を公開ベンチと同じ手順（ウォームアップ3回＋計測20回×3プロセス）で回す |
| `tests/policy/test_vlacpp_policy.py` | サーバーを実際に起動して1回推論する（重いので `@pytest.mark.slow`） |

## 4. インタフェース仕様

```python
# src/evla/policy/vlacpp_server.py
@dataclass(frozen=True)
class VlaServerConfig:
    binary: Path                       # third_party/vla.cpp/build{,-sycl,-ov}/vla-server
    gguf: Path
    bind: str = "tcp://127.0.0.1:5555"
    env: Mapping[str, str] = field(default_factory=dict)   # VLA_N_THREADS, GGML_OPENVINO_DEVICE, VLA_DEVICE ...
    args: tuple[str, ...] = ()         # ("--weight-dtype", "f16", "--num-steps", "10")
    expect_backend: str | None = None  # "CPU" | "SYCL" | "OPENVINO"。バナーと一致しなければ例外
    ready_timeout_s: float = 300.0     # OV-GPUはコンパイルで遅いので長めにする
    log_path: Path | None = None

class VlaServerProcess:
    """with文で使う。__enter__ で起動して ready まで待ち、__exit__ で SIGTERM → 待機 → SIGKILL。"""
    def __init__(self, cfg: VlaServerConfig) -> None: ...
    def __enter__(self) -> "VlaServerProcess": ...
    def __exit__(self, *exc) -> None: ...
    @property
    def pid(self) -> int: ...            # メモリ測定（26章）で使う
    @property
    def banner(self) -> list[str]: ...   # 起動ログの先頭。結果に保存する
```

```python
# src/evla/policy/vlacpp_policy.py
class VlaCppPolicy(Policy):
    def __init__(self, cfg: PolicyConfig) -> None:
        """cfg.server_addr に接続する（サーバーの起動は呼び出し側の責任）。
        cfg.extra["tokenizer"]: 既定 "HuggingFaceTB/SmolVLM2-500M-Instruct"
        cfg.extra["recv_timeout_ms"]: 既定 600_000（初回のコンパイルを考慮）"""

    def _encode(self, obs: Observation, noise: npt.NDArray[np.float32] | None) -> bytes:
        """Observation → PredictRequest のバイト列。前処理は2.1の表のとおり。"""

    def predict_chunk(self, obs: Observation, noise=None) -> ChunkResult:
        """送受信。応答の error が空でなければ RuntimeError。
        breakdown_ms に、サーバー側の latency_ms_*（total / inference / vision / prefill / denoise）と、
        transport = (クライアントで測った往復時間) - latency_ms_total を入れる。"""
```

## 5. 実装手順

1. **サーバー管理（2時間）**
   - `subprocess.Popen(..., stdout=PIPE, stderr=STDOUT, env=...)` で起動し、別スレッドで標準出力を読んで `ready` の行を待つ。
   - バナーから `backend = ...` の行（SYCL）や `OpenVINO: using device ...` の行（OV）を拾い、`expect_backend` と照合する。**一致しなければ例外にする**（黙ってCPUにフォールバックした結果を測らないため）。
   - 終了時は SIGTERM → `wait(timeout=10)` → SIGKILL。`atexit` にも登録して、ゾンビプロセスを残さない。
2. **クライアント（2時間）**
   - `vla_cpp_client.py` を**読んで理解したうえで**、`_encode` を自分で実装する（コピーしない）。
   - リサイズは `resize_with_pad`（アスペクト比を保って縮小し、余白を0で埋める）。LeRobot の SmolVLA と同じ方式であることを確認する。
3. **一致の確認（1時間）**：22章の比較表にvla.cpp（CPU、既定のweight-dtype）を追加する。固定ノイズは `noise` フィールドで渡す（`chunk_size × max_action_dim` = 50×32 を平坦化）。
4. **量子化版の作成（1時間）**
   - vla.cppの変換用venv（`python3 -m venv .venv-converter` → `pip install -e ".[convert]"`。vla.cppのディレクトリで実行）を作る。
   - 公開GGUF（`vrfai/smolvla-libero-gguf` の `smolvla-libero.gguf`）から、`scripts/quantize_gguf.py --in ... --out ...-q8_0.gguf --type Q8_0` で量子化版を作る。Q4_0も同様。`--vision` を付けた版（ビジョン部も量子化する）も作る。
   - **【要確認】** 公開GGUFの重みの型（bf16かf16か）を `gguf` パッケージ（`gguf-dump`）で確認し、メモに残す。
5. **バックエンドの切り替え（1時間）**：CPU / SYCL / OV-GPU の各ビルドで、同じクライアントが動くことを確認する。
6. **参照ベンチ（30分）**：`vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown` を、公開ベンチと同じ手順で回す。**自作ベンチ（26章）の値との差＝前処理と通信のオーバーヘッド**になる。

### 5.1 掃引に使うvla.cppの軸

| 軸 | 値 | 指定方法 |
| --- | --- | --- |
| バックエンド | CPU / SYCL / OV-CPU / OV-GPU | ビルドの選択＋ `GGML_OPENVINO_DEVICE` |
| 重みの保持型 | f32 / bf16 / f16 | `--weight-dtype` |
| 量子化 | なし / Q8_0 / Q4_0（＋ `--vision`） | GGUFファイルの選択 |
| スレッド | 1〜16 | `VLA_N_THREADS`（環境変数） |
| 入力 | 視点1 / 2、解像度 | クライアントが送る画像の枚数とサイズ |
| ステップ数 | 10 / 5 | `--num-steps` |

**【要確認】** サーバーが512以外の画像サイズを受け付けるか（内部でリサイズするのか、エラーになるのか）は、ドキュメントに書かれていません。256で送って挙動を確かめる。

## 6. ベストプラクティス

- **サーバーの寿命を `with` で管理する**：例外で抜けてもサーバーが必ず止まるようにする。止め忘れたサーバーがCPUを使い続けると、次の測定が汚れる。
- **起動ログを結果に保存する**：どのバックエンドとデバイスで動いたかの唯一の証拠になる。
- **REQソケットのタイムアウト後の回復**：REQ/REPは「送信→受信」の順序を厳格に守るので、受信がタイムアウトすると、そのソケットでは次の送信ができなくなる。`REQ_RELAXED` と `REQ_CORRELATE` を設定するか、ソケットを作り直す（Lazy Pirateパターン）。
- **ポートの衝突を避ける**：並列でサーバーを立てるときは、`bind` のポートを変える。測定中は1台だけにする。
- **トークン化をキャッシュする**：同じ指示文なら、トークン列は毎回同じになる。初回だけトークン化してキャッシュすれば、クライアント側のオーバーヘッドが減る。キャッシュを使ったかどうかは記録する。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 応答が来ずに固まる | サーバーが初回コンパイル中（OV-GPU）、またはクラッシュした | `RCVTIMEO` を長めに設定し、サーバーのプロセスが生きているか確認する |
| タイムアウト後、すべての送信が失敗する | REQソケットの状態機械が壊れた | ソケットを作り直す |
| 行動がPyTorchと大きく違う | 指示文の改行がない。状態を正規化してから送った（二重の正規化） | 2.1の表を再確認する |
| SYCLなのに遅い／バナーがCPU | GPUが見えていない | `sycl-ls`、render グループ、`VLA_DEVICE` |
| OVバックエンドの2回目の起動から行動がおかしい | `GGML_OPENVINO_CACHE_DIR` を設定している | 設定しない（vla.cppの既知の問題） |
| OVでSmolVLAの段階別の時間が変 | `VLA_TIMING=phase` の経路がOVでは誤り（既知の問題） | OVでは段階別の時間を使わない。total だけを使う |
| Q4_0で行動がほぼランダム | 量子化誤差が大きすぎる | 仕様として成功率で評価する。`--vision` を外した版と比べる |

## 8. ヒント

<details><summary>L1：サーバーの準備完了をどう待つか</summary>

標準出力を1行ずつ読み、特定の文字列（`ready`）が出たら完了とする。読み込みはブロックするので、別スレッドで読んでキューに入れ、メインスレッドはタイムアウト付きでキューを待つ。
</details>

<details><summary>L2：使うAPI</summary>

`subprocess.Popen(cmd, env={**os.environ, **cfg.env}, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`、`threading.Thread(daemon=True)`、`queue.Queue.get(timeout=...)`、`proc.send_signal(signal.SIGTERM)`、`zmq.Context.instance().socket(zmq.REQ)`、`sock.setsockopt(zmq.RCVTIMEO, ms)`、`vla_pb2.PredictRequest()`、`req.images.add()`、`req.SerializeToString()`、`resp.ParseFromString(body)`、`transformers.AutoTokenizer.from_pretrained(...)`
</details>

<details><summary>L3：_encode の疑似コード</summary>

```
req = PredictRequest(request_id=self._next_id())
for k in cfg.views:
    chw = obs.images[k].transpose(2,0,1).astype(f32) / 255
    chw = resize_with_pad(chw, cfg.image_size, cfg.image_size, pad=0)
    hwc = ascontiguousarray(chw.transpose(1,2,0), f32)
    im = req.images.add(); im.encoding = F32_RGB_01; im.height, im.width = hwc.shape[:2]; im.data = hwc.tobytes()
req.lang_tokens.extend(self._tokens(obs.task))       # 改行付き、キャッシュ
s = zeros(32, f32); s[:len(obs.state)] = obs.state; req.state.extend(s.tolist())
if noise is not None: req.noise.extend(noise.ravel().tolist())
return req.SerializeToString()
```
</details>

## 9. 完了条件（DoD）

- [ ] `with VlaServerProcess(...)` で起動・推論・終了ができ、終了後に `pgrep vla-server` が空
- [ ] バナーが期待と違うと例外になる（わざと `expect_backend="SYCL"` をCPUビルドに指定して確かめる）
- [ ] 22章の一致表に vla.cpp（CPU）の行がある
- [ ] Q8_0 / Q4_0（`--vision` あり/なし）のGGUFが `data/models/gguf/` にある
- [ ] `vla-bench` の参照値と、自作クライアントでの値の両方がある
- [ ] SYCLまたはOV-GPUのどちらか一方以上が動いた（動かなかった方は、試したことを記録した）

## 10. 理解度チェック・発展課題

- REQ/REPで、タイムアウト後にソケットが使えなくなるのはなぜか。DEALER/ROUTERならどうなるか。
- `breakdown_ms["transport"]` が大きい場合、何を疑うか（画像のシリアライズ形式、float32かuint8か）。
- 発展（D8、任意）：`vrfai/pi05-libero-gguf` や `vrfai/gr00tn1d6-libero-gguf` で同じ測定をし、「どこで破綻するか」を示せ。GR00Tは `VLA_GR00T_EMBODIMENT` と `--stats-json` が必要なことに注意する。
