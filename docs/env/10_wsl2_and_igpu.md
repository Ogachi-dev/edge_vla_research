# env/10. WSL2の設定とiGPU（Iris Xe）、CPUトポロジ

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章に穴埋めはありません。コマンドを順に実行し、各 Step の「確かめる」をすべてチェックしてから次へ進みます
> - 自分で書くもの：`scripts/probe_vcpu_speed.py`（Step 4。4.3節のインタフェースのとおり）。4.2節の `check_igpu.py` は任意（Step 3 の1行のコマンドで代用できる）
> - ゴール：Ubuntu 22.04 の WSL から iGPU が見え、「WSL 内で P/E コアを固定できるか」の結論が出ている

## 1. 目的と計画書の対応

- WSL2上で、**メモリ・CPU・iGPUを計測に使える状態**にする。
- WSL2特有の「CPUトポロジが見えない」問題を実測で確かめ、P/Eコアの扱い方を決める。
- 計画書の対応箇所：「前提条件とスコープ（検証端末）」「環境構成（iGPUドライバ）」「測定条件の統一」、日程D1（iGPU認識の確認）

## 2. 前提知識・事前に読む資料

- WSLの設定ファイル `.wslconfig`：<https://learn.microsoft.com/windows/wsl/wsl-config>
- WSLのGPUコンピュート：<https://learn.microsoft.com/windows/wsl/tutorials/gpu-compute>
- Intel compute runtime（NEO）：<https://github.com/intel/compute-runtime>（README の WSL の項）
- OpenVINOのGPU設定：<https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html>
- i7-1360Pの構成：Pコア4基（HTあり、8スレッド）＋Eコア8基（HTなし、8スレッド）＝16論理CPU。AVX2・AVX-VNNIあり、AVX-512・AMXなし。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `C:\Users\<you>\.wslconfig`（Windows側） | WSL VMのメモリ・CPU・スワップの設定 |
| `scripts/probe_vcpu_speed.py`（自分で書く） | vCPUごとのシングルスレッド性能を測る（P/Eコア推定実験） |
| `scripts/check_igpu.py`（任意） | OpenVINOのデバイス一覧とGPU名を表示する（5節 Step 3 の1行のコマンドで代用できる） |
| `results/runs/<id>_env_vcpu-probe/` | 上の実験結果 |
| `docs/notes/env_setup.md`（自作） | 実際に打ったコマンドと結果のメモ |

## 4. インタフェース仕様

### 4.1 `.wslconfig`（雛形）

```ini
[wsl2]
memory=24GB          # 32GB中。Windows側（iGPU の共有メモリもここから取られる）に 8GB 残す
processors=16        # 論理CPUをすべて渡す（実験によっては8に絞る）
swap=8GB
# TODO(you): 必要に応じて追加する（例：networkingMode=mirrored）
```

`.wslconfig` は**全ディストリビューション共通**です（WSL2のディストリビューションはすべて1つのVMの中で動く）。変更後は PowerShell で `wsl --shutdown` を実行して再起動します。

### 4.2 `scripts/check_igpu.py`

```python
def list_ov_devices() -> dict[str, str]:
    """{device_name: FULL_DEVICE_NAME} を返す。例 {'CPU': '13th Gen Intel(R) Core(TM) i7-1360P', 'GPU': 'Intel(R) Iris(R) Xe Graphics ...'}"""

def gpu_properties(device: str = "GPU") -> dict[str, Any]:
    """OPTIMIZATION_CAPABILITIES（FP16やINT8に対応しているか）、GPU_DEVICE_TOTAL_MEM_SIZE などを返す。"""
```

### 4.3 `scripts/probe_vcpu_speed.py`

```python
@dataclass(frozen=True)
class ProbeResult:
    vcpu: int
    round: int
    elapsed_ms: float

def run_workload(n_iter: int) -> None:
    """シングルスレッドで、決まった量の計算をする（BLASに依存しない純Pythonか、1スレッドに固定したNumPy）。"""

def probe(vcpus: Sequence[int], rounds: int, n_iter: int) -> list[ProbeResult]:
    """各vCPUに順番に固定して run_workload を計時する。round ごとに vCPU の順序をシャッフルする。"""
```

## 5. ハンズオン

各 Step は「**実行する → 出力を確かめる → メモに残す**」の1サイクルです。実際に打ったコマンドと出力は `docs/notes/env_setup.md` に貼っていきます（報告書の「環境」の節の材料になる）。

### Step 1：Windows側の準備（30分）

1. Intelの最新グラフィックスドライバ（Iris Xe）をWindowsに入れる。WSLのGPUは、Windows側のドライバを `/dev/dxg` 経由で使う。
2. AC電源につなぎ、電源モードを「最適なパフォーマンス」にする（設定 → システム → 電源）。電源プランも最高パフォーマンスにする（`powercfg /list` で現在のプランを確認）。
3. `.wslconfig` を作成し、`wsl --shutdown` で再起動する。

**確かめる**

- [ ] `powercfg /list` で、使っている電源プランに `*` が付いている。電源モードと合わせてメモに残す
- [ ] WSL を起動し直して `free -g` の total が約23GB、`nproc` が `.wslconfig` の `processors` と同じ

### Step 2：使うディストリビューションの確認（15分）

```powershell
wsl -l -v      # 使うディストリビューションの VERSION が 2 であること
```

**【判断】** 作業はすべて、ROS 2 Humble を入れてある既存の **Ubuntu 22.04** で行う（**【計画書との差分】** 計画書は 24.04 を追加する案。理由は [decisions](../decisions.md)）。24.04 を選んだ理由だった3点は、次のように扱う。

| 24.04 を選んだ理由 | 22.04 での扱い |
| --- | --- |
| LeRobot 0.6系は Python 3.12 以上が必須 | `.venv-bench` は uv が入れた Python 3.12 で作る（システムの 3.10 は使わない）。ROS 側（3.10）からは lerobot を import しない（[01](../01_repository_design.md) 4.1） |
| vla.cpp のビルド済みバイナリは 24.04 以降の glibc を要求する | 22.04 の glibc は 2.35 なので、ビルド済みバイナリは使わず、ソースからビルドする（vla.cpp は CMake 3.22・C++17 で、22.04 の CMake 3.22.1・gcc 11 で足りる） |
| oneAPI 2026.x は 22.04（jammy）のサポートを打ち切った | vla.cpp の SYCL 版は oneAPI 2025.x で試す（**【要確認】**）。OpenVINO の GPU と vla.cpp の OpenVINO バックエンドは oneAPI が要らない |

**確かめる**

- [ ] `wsl -l -v` で、使うディストリビューションの VERSION が `2`
- [ ] Ubuntu 側で `lsb_release -a` が `22.04`、`python3 --version` が `3.10.x`、`uv python list` に 3.12 がある

### Step 3：WSL内でのGPU確認（1時間）

1. デバイスファイルを確認する：`ls -l /dev/dxg`（存在すればよい。Intel iGPUの場合、WSLには `/dev/dri` が無いことが多い）
2. Intel compute runtime を入れる。**【要確認】** パッケージの入手元は時期によって変わります。2026年時点の候補は次の2つです。
   - Intel の GPU 用 apt リポジトリ（<https://dgpu-docs.intel.com/>）の 22.04（jammy）向けから `intel-opencl-icd libze1 clinfo` などを入れる。Intel の PPA（`ppa:kobuk-team/intel-graphics`）は 24.04 以降向けなので使えない
   - <https://github.com/intel/compute-runtime/releases> の `.deb` 一式を入れる。最新版は新しい Ubuntu でビルドされているので、22.04 の glibc（2.35）で動く版かを確かめる
   - 22.04 の標準の `intel-opencl-icd`（22.14）は古く、WSL で GPU が見えないことがある
   - どちらの場合も、**WSLをサポートするビルドか**をリリースノートで確認する。
3. `clinfo -l` を実行する。`Intel(R) Graphics` などのGPUプラットフォームが列挙されればOK。
4. `.venv-bench` を作った後（[env/12](12_inference_envs.md)）、OpenVINO から `GPU` が見えることを確認する。

```bash
python -c "import openvino as ov; c = ov.Core(); print([(d, c.get_property(d, 'FULL_DEVICE_NAME')) for d in c.available_devices])"
```

**確かめる**

- [ ] `ls -l /dev/dxg` でデバイスファイルがある
- [ ] `clinfo -l` に `Intel(R) Graphics` などの GPU プラットフォームが出る
- [ ] （env/12 の後）OpenVINO のデバイス一覧に `CPU` と `GPU` の両方があり、GPU の名前に `Iris(R) Xe` が入る

### Step 4：vCPUのP/Eコア推定実験（1〜2時間）

1. `lscpu -e` を実行し、WSLから見えるトポロジを記録する（この端末では「8コア×2スレッド」の均質な構成に見えた）。
2. `scripts/probe_vcpu_speed.py` を4.3節のインタフェースのとおりに書き（`os.sched_setaffinity` で1つの vCPU に固定 → `run_workload` を計時、を vCPU の順番をシャッフルしながら繰り返す）、`rounds=5` で全16 vCPUを測る。
3. vCPUごとの所要時間を箱ひげ図にする。次のどちらに当たるかで、方針を決める。

| 観測結果 | 解釈 | 方針 |
| --- | --- | --- |
| 速い群と遅い群に分かれ、roundをまたいでも安定している | vCPUが物理コアに概ね固定されている | 速い群のvCPUを「Pコア相当」として `taskset` に使う |
| ばらつきが大きい、またはroundごとに速いvCPUが入れ替わる | ホスト側のスケジューラがvCPUを動的に移動させている | WSL内でのP/E固定は諦める（下記の代替案） |

**代替案（【計画書との差分】）**：計画書は「P/Eコアを固定（taskset）」としていますが、WSL2では効かない可能性が高いです。その場合は次のどちらかにします。

- (a) `.wslconfig` の `processors` を変えたり（8 / 16）、スレッド数を掃引したりして、**スレッド数の効果だけ**を報告する。P/Eの効果は「WSL2では制御不能」と明記する。
- (b) P/Eの実験だけ、**Windowsネイティブ**のPython（OpenVINOはWindowsで動く）で行う。`start /affinity <mask>` で起動し、マスクで論理プロセッサを指定する。Windowsでの論理プロセッサの番号付けは `Get-CimInstance Win32_Processor` やCoreinfo（Sysinternals）で確認する。

**確かめる**

- [ ] `lscpu -e` の出力がメモにある
- [ ] vCPU ごとの所要時間の箱ひげ図（5 round 分）がある
- [ ] 上の表のどちらに当たったかと、採る方針（taskset／代替案 (a)／(b)）を `docs/decisions.md` に書いた（26章の Step 6 で使う）

## 6. ベストプラクティス

- **「確認したこと」と「仮定したこと」を分けて記録する**：WSL2の内部挙動はドキュメントが少ないので、実測した事実をメモに残す。報告書の「限界」の節にそのまま使える。
- **GPUの確認は3段階で行う**：`/dev/dxg` がある → `clinfo` で見える → OpenVINOで見える。どの段で失敗したかで原因を切り分けられる。
- **iGPUのメモリは共有メモリ**：Iris Xeの専用VRAMはなく、システムRAMを使う。`.wslconfig` の `memory` を増やしても、iGPUに見えるメモリは**Windows側**の共有GPUメモリの上限（通常は物理RAMの半分）で決まる。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `free -g` が15GBのまま | `.wslconfig` の位置かファイル名が違う。または再起動していない | ファイルは `%UserProfile%\.wslconfig`（拡張子を付けない）。`wsl --shutdown` で再起動 |
| `clinfo` で `Number of platforms 0` | compute runtimeが入っていない、WSL非対応のビルド、Windows側のドライバが古い | 3段階の確認をやり直す。Windows側のドライバを更新する |
| OpenVINOで `GPU` が見えない | `intel-opencl-icd` が見つからない、または `libOpenCL` の版が合わない | `ls /etc/OpenCL/vendors/` に `intel.icd` があるか確認する |
| 測るたびに速さが変わる | vCPUがホスト側で移動している。ターボの持続時間切れ | Step 4の実験で確かめる。温度・クロックをWindows側で記録する（[40](../common/40_metrics_measurement.md)） |
| `taskset -c 0-7` で速くならない | WSL内のvCPU番号は物理Pコアを意味しない | 上記の代替案 |
| OV-GPU の推論や Gazebo の描画中に Windows 全体が重くなる、落ちる | RAM 32GB のうち WSL に 24GB を渡しているので、Windows 側と iGPU の共有メモリが足りない | `memory=20GB` に下げる。Gazebo とトラック1のベンチを同時に動かさない |

## 8. ヒント

<details><summary>L1：シングルスレッド性能をどう測るか</summary>

並列化されない固定量の計算を、1つのvCPUに固定した状態で何回か実行し、時間を測る。BLASは内部でスレッドを立てるので使わないか、1スレッドに制限する。
</details>

<details><summary>L2：使うAPI</summary>

`os.sched_setaffinity(0, {cpu})`、`time.perf_counter_ns()`、`random.Random(seed).shuffle(order)`。NumPyを使う場合は、import前に `OPENBLAS_NUM_THREADS=1` と `OMP_NUM_THREADS=1` を設定する。OpenVINO：`openvino.Core().available_devices`、`core.get_property(dev, "FULL_DEVICE_NAME")`。
</details>

<details><summary>L3：probe の疑似コード</summary>

```
results = []
for r in range(rounds):
    order = list(vcpus); rng.shuffle(order)
    for cpu in order:
        sched_setaffinity(0, {cpu})
        run_workload(n_iter // 10)          # ウォームアップ
        t0 = perf_counter_ns(); run_workload(n_iter); t1 = perf_counter_ns()
        results.append(ProbeResult(cpu, r, (t1 - t0) / 1e6))
sched_setaffinity(0, 元のaffinity)          # 戻す
```
</details>

## 9. 完了条件（DoD）

- [ ] Ubuntu 22.04のWSLで `free -g` の total が約23GB
- [ ] `clinfo -l` でIntel GPUが列挙される
- [ ] OpenVINO のデバイス一覧が `CPU` と `GPU` の両方を表示する（5節 Step 3。[env/12](12_inference_envs.md) の後）
- [ ] vCPUの推定実験の結果（図と表）があり、「WSL内でP/E固定ができるか」の結論がメモに書かれている
- [ ] Windowsの電源設定を記録した

## 10. 理解度チェック・発展課題

- WSL2のGPUは、Linuxカーネルのドライバ（i915/xe）ではなく何を経由して動いているか。
- iGPUで推論しているとき、CPU側の性能に影響しうる理由を2つ挙げよ（ヒント：電力枠の共有、メモリ帯域）。
- 発展：Gazeboの描画（Iris Xeを使う）とOpenVINO GPU推論を同時に動かすと、何が起きると予想されるか。計画書の「リスク」の表と照らし合わせよ。
