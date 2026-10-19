# env/12. 推論環境：uvのvenvとvla.cppのビルド

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**1個**（`scripts/gen_vla_proto.sh`）。ほとんどは環境構築の作業です
> - ゴール：`.venv-bench` で `import lerobot, openvino, physicalai` が通り、vla.cpp の CPU 版で `vla-bench` の結果表が出る
> - 進め方：5節の Step を順に実行し、各 Step の「確かめる」を全部チェックしてから次へ進む。SYCL（Step 4）と OV（Step 5）は、時間を決めて打ち切ってよい

## 1. 目的と計画書の対応

- トラック1と推論サーバーで使う Python 環境（`.venv-bench`、`.venv-export`）を uv で作る。
- vla.cpp を CPU / SYCL / OpenVINO の3つのバックエンドでビルドする。
- 計画書の対応箇所：「環境構成（推論ランタイム、iGPUドライバ）」、日程D1〜D3（トラック1）

## 2. 前提知識・事前に読む資料

- uv：<https://docs.astral.sh/uv/pip/environments/>
- LeRobotのインストール：<https://huggingface.co/docs/lerobot/en/installation>
- vla.cpp の README と `docs/USAGE.md`、`docs/backend/sycl.md`、`docs/backend/ov.md`：<https://github.com/VinRobotics/vla.cpp>
- OpenVINOのアーカイブ版インストール（vla.cppのOVバックエンド用）：<https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html>
- oneAPI（SYCL用）：<https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `.venv-bench/` | トラック1と推論サーバー用のvenv |
| `.venv-export/` | physicalai-train（OpenVINO IRへの書き出し）用のvenv |
| `requirements/bench.lock.txt` | `uv pip freeze` の結果（再現用） |
| `third_party/vla.cpp/`（submodule、コミット固定） | vla.cpp のソース |
| `third_party/vla.cpp/build{,-sycl,-ov}/` | バックエンドごとのビルドディレクトリ |
| `scripts/vla_env.sh` | vla.cpp を使うときの環境変数（setvarsなど）をまとめたもの |
| `src/evla/policy/_proto/vla_pb2.py` | `vla.proto` から生成した Python スタブ（`scripts/gen_vla_proto.sh` で生成。git では管理しない） |

## 4. インタフェース仕様

### 4.1 調査で確認したバージョン（2026-10時点）

| パッケージ | 版 | 備考 |
| --- | --- | --- |
| lerobot | 0.6.1（PyPI）／ main は 0.6.2 | `requires-python>=3.12`、`numpy>=2.0,<2.3`、`torch>=2.7,<2.12` |
| hf-libero | `>=0.1.4,<0.2.0` | `lerobot[libero]` の依存。Linuxのみ |
| openvino | 2026.4.1 | |
| nncf | 3.4.0 | INT8の重み圧縮・量子化 |
| physicalai | 0.3.0 | `openvino>=2026.4`、推論ランタイム（`InferenceModel`） |
| physicalai-train | 0.3.0 | `lerobot[dataset]==0.6.0` を固定で要求 → 別venvにする |

**【要確認】** 実装開始時に `uv pip index versions <pkg>`（または PyPI のページ）で最新版を確認し、上の表を更新してから固定する。

### 4.2 vla.cpp の環境変数とフラグ（`docs/USAGE.md` より）

| 種類 | 名前 | 意味 |
| --- | --- | --- |
| 環境変数 | `VLA_N_THREADS` | CPUバックエンドのスレッド数（既定はコア数、上限16）。**フラグではなく環境変数で指定する** |
| 環境変数 | `VLA_DEVICE` | CUDA / SYCL のGPU番号（既定0） |
| 環境変数 | `GGML_OPENVINO_DEVICE` | OpenVINOバックエンドのデバイス（`CPU` / `GPU` / `NPU`）。`VLA_DEVICE` は効かない |
| 環境変数 | `VLA_CACHE` | `-hf` でダウンロードしたモデルの置き場所（既定 `~/.cache/vla`） |
| フラグ | `--weight-dtype f32\|bf16\|f16` | GEMMの重みを保持する型 |
| フラグ | `--num-steps N` | flow matching の積分ステップ数（SmolVLAの既定は10） |
| フラグ | `--flash-attn` | 速くなる場合があるが、数値が変わる |
| `vla-bench` | `--images N --size S --markdown` | 合成入力で `predict()` だけを計時する |
| `vla-server` | `--bind tcp://*:5555` | ZeroMQのREQ/REP。リクエストは1件ずつ同期的に処理する |

### 4.3 `scripts/vla_env.sh`（雛形）

```bash
#!/usr/bin/env bash
# 使い方: source scripts/vla_env.sh {cpu|sycl|ov}
backend="${1:-cpu}"
export VLA_CACHE="$HOME/.cache/vla"
case "$backend" in
  cpu)  export VLA_BIN="$PWD/third_party/vla.cpp/build" ;;
  sycl) source /opt/intel/oneapi/setvars.sh --force >/dev/null
        export VLA_BIN="$PWD/third_party/vla.cpp/build-sycl" ;;
  ov)   source /opt/intel/openvino/setupvars.sh
        export VLA_BIN="$PWD/third_party/vla.cpp/build-ov" ;;
esac
# TODO(you): VLA_N_THREADS は、ここではなく実験スクリプト側で毎回明示的に設定する
```

## 5. ハンズオン

各 Step は「**実行する → 出力を確かめる → 記録する**」の1サイクルです。出力は `docs/notes/` のその日のメモに貼っておくと、後で報告書の「環境」の節にそのまま使えます。

### Step 1：`.venv-bench` を完成させる（1〜2時間）

[01](../01_repository_design.md) の Step 0 で最小の `.venv-bench` を作っていれば、`uv venv` は飛ばして `source` から始めます。

```bash
export UV_LINK_MODE=copy          # WSLでのハードリンク警告を避ける（vla.cppのWSL文書でも推奨）
uv venv -p 3.12 .venv-bench       # 01 で作っていれば不要
source .venv-bench/bin/activate
```

**torch は CPU 版を先に入れる**。PyPI の torch（Linux）は CUDA ライブラリを同梱していて数GBあり、この端末では無駄になる。

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install "lerobot[smolvla,libero]==0.6.1"   # TODO(you): 4.1で確定した版
uv pip install openvino nncf physicalai pyzmq protobuf grpcio-tools psutil pandas pyarrow matplotlib
uv pip install -e ".[bench,dev]"                    # 自作パッケージ evla
mkdir -p requirements && uv pip freeze > requirements/bench.lock.txt
```

**動かす**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import lerobot, openvino as ov, physicalai; print(lerobot.__version__, ov.Core().available_devices)"
```

**確かめる**

- [ ] 1行目が `2.x.x+cpu False` のように、`+cpu` で終わり、`False` になっている
- [ ] 2行目に lerobot の版と、`['CPU', 'GPU']` が出る（`GPU` が無ければ [env/10](10_wsl2_and_igpu.md) の Step 3 に戻る）
- [ ] `requirements/bench.lock.txt` ができている

torch が CUDA 版に置き換わっていたら（`+cpu` が無い）、`uv pip install` に `--torch-backend cpu` を付けて入れ直す（**【要確認】** uv のバージョンによる）。

### Step 2：`.venv-export`（30分。D8までに準備すればよい）

```bash
uv venv -p 3.12 .venv-export
source .venv-export/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install "physicalai-train[nncf]==0.3.0"
python -c "import physicalai, nncf; print('ok')"
deactivate
```

- [ ] `ok` が表示される

### Step 3：vla.cpp CPUビルド（1時間）

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config \
  libzmq3-dev cppzmq-dev libprotobuf-dev protobuf-compiler
git submodule add https://github.com/VinRobotics/vla.cpp third_party/vla.cpp
cd third_party/vla.cpp && git checkout <固定するコミット>   # TODO(you)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./build/vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown
cd -
```

**確かめる**

- [ ] `vla-bench` が Markdown の表（ステージごとの時間）を出す。その表をメモに貼る
- [ ] `git -C third_party/vla.cpp rev-parse HEAD` のコミットをメモに残す

動作確認だけなら、ビルド済みバイナリ（`linux-x86_64-cpu`、AVX2が必要）を使ってもよい。ただし本番の計測では、**自分でビルドしたもの**（`GGML_NATIVE=ON` でこのCPU向けに最適化される）を使う。

### Step 4：vla.cpp SYCLビルド（2〜3時間。うまくいかなければ打ち切る）

1. compute runtime が入っていることを確認する（[env/10](10_wsl2_and_igpu.md)）。
2. oneAPIを入れる。Deep Learning Essentials で足りる（DPC++、oneMKL、oneDNN）。
3. `source /opt/intel/oneapi/setvars.sh` → `sycl-ls` を実行する。
4. ビルドする（**ビルドディレクトリは新しく作る**。コンパイラを途中で替えたディレクトリは壊れる）。

```bash
cd third_party/vla.cpp
cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl -j"$(nproc)"
./build-sycl/vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown
cd -
```

**確かめる**

- [ ] `sycl-ls` に `[level_zero:gpu]` または `[opencl:gpu]` が出る
- [ ] 起動時のバナーに `backend = SYCL (device 0: Intel(R) Iris(R) Xe ...)` と出る（`CPU` と出たら、GPUが見えていない）
- [ ] 動かなかった場合も、エラーと試したことをメモに残して先へ進む（DoD はそれで満たせる）

### Step 5：vla.cpp OpenVINOバックエンド（1〜2時間）

1. OpenVINO をアーカイブ版で `/opt/intel/openvino` に入れる（vla.cppの `scripts/install_ov.sh` も使える）。
2. `source /opt/intel/openvino/setupvars.sh` を読んでからビルドする。

```bash
cd third_party/vla.cpp
cmake -B build-ov -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENVINO=ON
cmake --build build-ov -j"$(nproc)"
GGML_OPENVINO_DEVICE=GPU ./build-ov/vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown
cd -
```

**確かめる**

- [ ] 起動ログに `OpenVINO: using device GPU`（ggml側の行が正）が出る
- [ ] CPU 版（Step 3）の表と並べて、メモに貼る

### Step 6：Python のプロトコルモジュールを生成する ― `scripts/gen_vla_proto.sh`（穴①、15分）

vla.cpp のサーバー（25章）と話すための `vla_pb2.py` を、`vla.proto` から生成します。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `protoc` は出力する言語ごとにオプションがある。Python 用のものを選ぶ（`python -m grpc_tools.protoc --help` で一覧が見られる） |

**動かす**

```bash
source .venv-bench/bin/activate
bash scripts/gen_vla_proto.sh
python -c "from evla.policy._proto import vla_pb2; print([m for m in dir(vla_pb2) if m[0].isupper()][:10])"
```

**確かめる**

- [ ] `generated: src/evla/policy/_proto/vla_pb2.py` と表示される
- [ ] 2つ目のコマンドで、`vla.proto` に定義されたメッセージ名が並ぶ（`vla.proto` を開いて見比べる）

## 6. ベストプラクティス

- **ロックファイルを残す**：`uv pip freeze` の結果をコミットする。2週間のうちに上流が更新されて、結果が変わる事故を防ぐ。
- **外部リポジトリはコミットを固定する**：vla.cpp は更新が速い（2026-09だけで複数の数値修正がある）。submoduleのコミットを `env.json` にも記録する。
- **バックエンドごとにビルドディレクトリを分ける**：`build`、`build-sycl`、`build-ov`。同じディレクトリで切り替えると、キャッシュが混ざって壊れる。
- **「本当にそのデバイスで動いたか」を毎回ログで確認する**：SYCLもOpenVINOも、GPUが使えないと**黙ってCPUにフォールバック**する。起動バナーを結果ディレクトリに保存する。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| torch が CUDA 版になり数GBダウンロードされる | PyPIのLinux向けtorchはCUDA同梱 | CPU用のindexを指定する（Step 1） |
| `lerobot` の import で NumPy のエラー | ROSのシェルでvenvを使った | シェルを分ける |
| SYCL の `sycl-ls` が segfault する | oneAPI 2026.1 の Level Zero アダプタと、Ubuntu 24.04 の `libze1` 1.16 の組み合わせ（vla.cpp の既知の問題） | `UR_ADAPTERS_FORCE_LOAD` でOpenCLアダプタを強制する。または oneAPI 2025.3 を使う |
| SYCL で `could not create a memory object` | VMMプールとoneDNNの組み合わせ | `GGML_SYCL_ENABLE_VMM=0`（vla.cppの既定）のままにする |
| SYCL / OV の初回推論が極端に遅い | グラフのコンパイル（OVのGPUでは数十秒〜数分） | ウォームアップで除外する。クライアントのタイムアウトを長くする |
| OV バックエンドの2回目の起動から行動がおかしい | **`GGML_OPENVINO_CACHE_DIR` を設定すると誤った行動を返す**（vla.cpp の既知の問題） | 設定しない |
| Iris Xe で bf16 の重みが遅い／動かない | Iris Xe はbf16演算を持たない（**【要確認】**） | `--weight-dtype f16` または `f32` を試し、両方の結果を記録する |

## 8. ヒント

<details><summary>L1：torch が CUDA 版に置き換わる理由</summary>

後から入れたパッケージの依存解決で、PyPIのtorchが「より新しい版」として選ばれることがある。torchの版を固定するか、index を明示する。
</details>

<details><summary>L2：確認に使うコマンド</summary>

`uv pip show torch`、`uv pip tree --package lerobot`、`python -c "import torch;print(torch.__config__.show())"`（使われているBLASとOpenMPを確認）、`ldd build/vla-server | grep -i -E "zmq|proto"`
</details>

<details><summary>L3：vla.cpp をバックエンドごとに順に確認する流れ</summary>

```
for b in cpu sycl ov:
    source scripts/vla_env.sh $b
    $VLA_BIN/vla-bench -hf vrfai/smolvla-libero-gguf --images 2 --size 512 --markdown | tee results/vlabench_$b.md
    バナー（backend=...）が期待どおりか目視する
```
</details>

## 9. 完了条件（DoD）

- [ ] `.venv-bench` で `import lerobot, openvino, physicalai` が通り、`torch.cuda.is_available()` が False
- [ ] `requirements/bench.lock.txt` がコミットされている
- [ ] vla.cpp CPU版の `vla-bench`（SmolVLA、2視点、512px）の結果表がある
- [ ] SYCL版とOV版それぞれについて、「動いた（バナーでGPUを確認）」または「動かなかった（エラーと試したこと）」がメモにある
- [ ] `src/evla/policy/_proto/vla_pb2.py` が生成され、`from evla.policy._proto import vla_pb2` が通る

## 10. 理解度チェック・発展課題

- `--weight-dtype bf16` と `f32` で、メモリ使用量と速度はどう変わると予想されるか。iGPUとCPUで答えは同じか。
- OpenVINOのCPUプラグインがggmlのCPUバックエンドより遅いことがある理由を、vla.cppの `docs/backend/ov.md` から読み取って説明せよ（重みの型に注目する）。
- 発展：`vla-bench` の計測（合成入力、エンジン内のみ）と、自作ベンチ（前処理・通信込み）の差は何で生じるか。
