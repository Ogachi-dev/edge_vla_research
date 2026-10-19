# track1/25. vla.cppランタイム（GGUF：CPU / SYCL / OpenVINOバックエンド）

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**20個**（`src/evla/policy/vlacpp_server.py` 7、`src/evla/policy/vlacpp_policy.py` 11、`scripts/vlacpp_quantize.sh` 1、`scripts/vlabench_reference.sh` 1）
> - ゴール：`with VlaServerProcess(...)` で vla-server を起動・推論・終了でき、22章の一致表に vla.cpp（CPU）の行が入る
> - テスト：サーバーが必要なので pytest はなし。Step 1〜2 は**偽のサーバー**と**送信内容の復元**で、vla.cpp なしで確かめられる

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
| （5節 Step 1〜3 のコマンド） | 偽のサーバーでの起動・終了、送信内容の復元、実際のサーバーでの推論の確認。専用のテストは作らない |

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

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章は「プロセスの管理」と「通信の中身」に分かれます。どちらも、vla.cpp を動かす前に**偽物**で確かめられるようにしてあります。

### Step 0：準備（10分）

```bash
source .venv-bench/bin/activate
grep -n "【穴\|【要確認】" src/evla/policy/vlacpp_{server,policy}.py scripts/vlacpp_quantize.sh scripts/vlabench_reference.sh
ls src/evla/policy/_proto/vla_pb2.py              # 無ければ env/12 の Step 6
ls third_party/vla.cpp/build/vla-server           # env/12 の Step 3 でビルドしたもの
third_party/vla.cpp/build/vla-server --help       # 引数名を確かめる（_cmd の【要確認】）
```

- [ ] 穴20個が表示され、`vla_pb2.py` と `vla-server` がある
- [ ] `--help` の引数名（`-m`、`--bind`）が `_cmd()` と合っている。違えば直す

### Step 1：サーバーの起動と終了 ― `vlacpp_server.py`（穴①〜⑦、1.5時間）

**読む**：4節の `VlaServerProcess`。「起動 → 標準出力を別スレッドで読む → `ready` の行で合図 → 本体は合図を待つ」の流れを図にする。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 正規表現が**文字列のどこかに**一致するか。先頭だけを見るメソッドとの違いに注意 |
| ② | 起動完了の行に含まれる語（docstring の例を見る） |
| ③④⑤ | スレッド間の合図（`threading.Event`）。読む側が `set`、待つ側が `wait`。④はエラーの出力も同じパイプで読むための指定 |
| ⑥⑦ | 終了の作法。まず終了を頼むシグナル、応じなければ強制終了 |

**動かす**：`ready` と言って眠るだけの**偽のサーバー**で確かめます。

```bash
printf '#!/bin/sh\necho "backend = CPU"\necho "vla-server bound to tcp://*:5555. ready."\nexec sleep 600\n' > /tmp/fake-vla-server
chmod +x /tmp/fake-vla-server
python - <<'EOF'
from pathlib import Path
from evla.policy.vlacpp_server import VlaServerConfig, VlaServerProcess, detect_backend

cfg = VlaServerConfig(binary=Path("/tmp/fake-vla-server"), gguf=Path("x.gguf"), ready_timeout_s=5)
with VlaServerProcess(cfg) as s:
    pid = s.pid
    print("pid:", pid, "banner:", s.banner, "backend:", detect_backend(s.banner))
print("alive after exit:", Path(f"/proc/{pid}").exists())

try:
    with VlaServerProcess(VlaServerConfig(binary=Path("/tmp/fake-vla-server"), gguf=Path("x"), expect_backend="SYCL")):
        pass
except RuntimeError as e:
    print("RuntimeError OK:", e)
print(detect_backend(["OpenVINO: using device GPU"]), detect_backend(["ggml_sycl_init: found 1 device"]))
EOF
```

**確かめる**

- [ ] `banner` に2行が入り、`backend: CPU`
- [ ] `alive after exit: False`（with を出たらプロセスが消えている＝ゾンビなし）
- [ ] `RuntimeError OK: バックエンドが違う: 期待 SYCL、実際 CPU`（**黙ってCPUにフォールバックした結果を測らない**仕組み）
- [ ] 最後の行が `OPENVINO SYCL`

### Step 2：送るメッセージを作る ― `vlacpp_policy.py` の穴①〜⑧（1.5時間）

**読む**：2.1の「クライアント側の責任」の表と、vla.cpp の `vla_cpp_client.py`（**読んで理解したうえで、コピーせずに**埋める）。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | ZeroMQ のソケットの種類。「送って、返事を待つ」側（サーバー側と対になる） |
| ②③ | ソケットのオプション。②は終了時に固まらないため、③は返事が来ないときに永久に待たないため |
| ④ | 22章の②の逆変換（CHW → HWC） |
| ⑤ | NumPy 配列を、そのままのバイト列にする（protobuf の `bytes` に入れる） |
| ⑥ | 指示文の末尾の改行。**これを忘れると行動が大きくずれる**（7節の落とし穴）。学習時の書式に合わせる |
| ⑦ | 状態を32次元にゼロ埋めする。スライスで「先頭の実次元ぶん」を指定する |
| ⑧ | `(50, 32)` を1次元に平坦化する |

**動かす**：サーバーに送らずに、作ったバイト列を**復元して中身を見ます**。

```bash
python - <<'EOF'
from pathlib import Path
from evla.policy.base import PolicyConfig
from evla.policy.equivalence import make_fixed_noise
from evla.policy.vlacpp_policy import VlaCppPolicy
from evla.sim.fixed_obs import load_fixed_obs

obs = load_fixed_obs(Path("data/fixed_obs/libero_object_t0.npz"))[0]
p = VlaCppPolicy(PolicyConfig(runtime="vlacpp", model="x"))     # connect はサーバーがなくても成功する
raw = p._encode(obs, make_fixed_noise(50, 32, 0), request_id=7)
req = p.pb.PredictRequest(); req.ParseFromString(raw)
print("images:", [(im.height, im.width, len(im.data)) for im in req.images])
print("state:", len(req.state), "noise:", len(req.noise), "id:", req.request_id)
print("tokens:", len(req.lang_tokens), "last:", repr(p.tokenizer.decode(req.lang_tokens[-1:])))
p.close()
EOF
```

**確かめる**

- [ ] `images: [(512, 512, 3145728), (512, 512, 3145728)]`（512×512×3×4バイト）
- [ ] `state: 32 noise: 1600 id: 7`
- [ ] `last: '\n'`（指示文の末尾が改行のトークン）

### Step 3：受け取って、実際のサーバーで推論する ― 穴⑨〜⑪（1時間）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ⑨ | Step 2 で使った「バイト列から復元する」メソッド |
| ⑩ | 応答の平坦な配列を、応答に書いてある行数・列数で整形する（決め打ちしない） |
| ⑪ | 時間の内訳。クライアントで測った往復時間から、サーバーの中で使った時間を引いた残りが「通信など」 |

**動かす**

```bash
huggingface-cli download vrfai/smolvla-libero-gguf smolvla-libero.gguf --local-dir data/models/gguf
python - <<'EOF'
from pathlib import Path
from evla.common.config import load_config
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy
from evla.policy.vlacpp_server import VlaServerConfig, VlaServerProcess
from evla.sim.fixed_obs import load_fixed_obs

obs = load_fixed_obs(Path("data/fixed_obs/libero_object_t0.npz"))[0]
srv = VlaServerConfig(binary=Path("third_party/vla.cpp/build/vla-server"),
                      gguf=Path("data/models/gguf/smolvla-libero.gguf"),
                      env={"VLA_N_THREADS": "8"}, expect_backend="CPU",
                      log_path=Path("results/smoke/vla-server.log"))
with VlaServerProcess(srv), make_policy(load_config(Path("configs/policy/vlacpp_cpu.yaml"), PolicyConfig)) as p:
    p.warmup(obs)
    r = p.predict_chunk(obs)
    print(r.actions.shape, {k: round(v, 1) for k, v in r.breakdown_ms.items()})
EOF
pgrep -a vla-server || echo "no vla-server"
```

**確かめる**

- [ ] `(50, 7)` と内訳が表示され、`transport` が数ms程度の正の値
- [ ] 最後に `no vla-server`
- [ ] 22章の Step 5 で `--test configs/policy/vlacpp_cpu.yaml` を足す（**先に別の端末で vla-server を起動しておく**）。固定ノイズは `noise` フィールドで渡るので、PyTorch と直接比べられる

### Step 4：量子化版を作る ― `scripts/vlacpp_quantize.sh`（穴①、1時間）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 同じループの2行目（`--vision` 付き）と見比べる |

**動かす**（vla.cpp の変換用 venv で）

```bash
(cd third_party/vla.cpp && python3 -m venv .venv-converter && .venv-converter/bin/pip install -e ".[convert]")
source third_party/vla.cpp/.venv-converter/bin/activate
bash scripts/vlacpp_quantize.sh
gguf-dump data/models/gguf/smolvla-libero.gguf | head -40     # 元の重みの型（【要確認】bf16 か f16 か）
deactivate
```

**確かめる**

- [ ] `data/models/gguf/` に `-q8_0`、`-q4_0`、それぞれの `-vision` 版の計4ファイルができる。サイズが小さくなっていく
- [ ] 公開 GGUF の重みの型をメモに残す

### Step 5：バックエンドの切り替えと参照ベンチ ― `scripts/vlabench_reference.sh`（穴①、1時間）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 連続実行で CPU が熱くなると、クロックが下がって後の回ほど遅くなる。プロセスの間に冷やす秒数（数十秒） |

**動かす**

```bash
BUILD=build bash scripts/vlabench_reference.sh           # CPU
BUILD=build-sycl bash scripts/vlabench_reference.sh      # SYCL（env/12 Step 4 が動いていれば）
```

- [ ] `results/vlabench/<日時>_<ビルド>/run{1,2,3}.md` ができる。3回の値のばらつきを見る
- [ ] Step 3 の Python から、`binary` を `build-sycl` / `build-ov` に、`expect_backend` を `SYCL` / `OPENVINO` に替えて、同じクライアントで動くことを確かめる
- [ ] `vla-bench` の値と自作クライアントの値の差（＝前処理と通信のオーバーヘッド）をメモに残す

### Step 6：仕上げ（10分）

- [ ] Python の2ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/policy/vlacpp_*.py` を通す
- [ ] `BACKEND_PATTERNS` の【要確認】を、実際のバナー（`results/smoke/vla-server.log`）に合わせて直した
- [ ] コミットする

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
