# env/10. WSL2の設定とiGPU（Iris Xe）、CPUトポロジ

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
| `scripts/check_igpu.py` | OpenVINOのデバイス一覧とGPU名を表示する |
| `scripts/probe_vcpu_speed.py` | vCPUごとのシングルスレッド性能を測る（P/Eコア推定実験） |
| `results/runs/<id>_env_vcpu-probe/` | 上の実験結果 |
| `docs/notes/env_setup.md`（自作） | 実際に打ったコマンドと結果のメモ |

## 4. インタフェース仕様

### 4.1 `.wslconfig`（雛形）

```ini
[wsl2]
memory=48GB          # 64GB中。Windows側とGazebo描画の余裕を残す
processors=16        # 論理CPUをすべて渡す（実験によっては8に絞る）
swap=16GB
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

## 5. 実装手順

### Step 1：Windows側の準備（30分）

1. Intelの最新グラフィックスドライバ（Iris Xe）をWindowsに入れる。WSLのGPUは、Windows側のドライバを `/dev/dxg` 経由で使う。
2. AC電源につなぎ、電源モードを「最適なパフォーマンス」にする（設定 → システム → 電源）。電源プランも最高パフォーマンスにする（`powercfg /list` で現在のプランを確認）。
3. `.wslconfig` を作成し、`wsl --shutdown` で再起動する。

### Step 2：Ubuntu 24.04 ディストリビューションの追加（30分）

```powershell
wsl --install -d Ubuntu-24.04
wsl -l -v      # VERSION が 2 であること
```

**【判断】** 作業はすべて **Ubuntu 24.04 側**で行う。既存の22.04は触らずに残す。理由：

- LeRobot 0.6系は Python 3.12 以上が必須。
- vla.cppのビルド済みバイナリは Ubuntu 24.04 以降の glibc を要求する。
- oneAPI 2026.x は22.04（jammy）のサポートを打ち切った。

### Step 3：WSL内でのGPU確認（1時間）

1. デバイスファイルを確認する：`ls -l /dev/dxg`（存在すればよい。Intel iGPUの場合、WSLには `/dev/dri` が無いことが多い）
2. Intel compute runtime を入れる。**【要確認】** パッケージの入手元は時期によって変わります。2026年時点の候補は次の2つです。
   - Ubuntu 24.04向けのIntel PPA（`ppa:kobuk-team/intel-graphics`）から `libze-intel-gpu1 libze1 intel-opencl-icd clinfo` を入れる
   - <https://github.com/intel/compute-runtime/releases> の `.deb` 一式を入れる
   - どちらの場合も、**WSLをサポートするビルドか**をリリースノートで確認する。
3. `clinfo -l` を実行する。`Intel(R) Graphics` などのGPUプラットフォームが列挙されればOK。
4. `.venv-bench` を作った後（[env/12](12_inference_envs.md)）、`python scripts/check_igpu.py` で `GPU` が見えることを確認する。

### Step 4：vCPUのP/Eコア推定実験（1〜2時間）

1. `lscpu -e` を実行し、WSLから見えるトポロジを記録する（この端末では「8コア×2スレッド」の均質な構成に見えた）。
2. `probe_vcpu_speed.py` を実装し、`rounds=5` で全16 vCPUを測る。
3. vCPUごとの所要時間を箱ひげ図にする。次のどちらに当たるかで、方針を決める。

| 観測結果 | 解釈 | 方針 |
| --- | --- | --- |
| 速い群と遅い群に分かれ、roundをまたいでも安定している | vCPUが物理コアに概ね固定されている | 速い群のvCPUを「Pコア相当」として `taskset` に使う |
| ばらつきが大きい、またはroundごとに速いvCPUが入れ替わる | ホスト側のスケジューラがvCPUを動的に移動させている | WSL内でのP/E固定は諦める（下記の代替案） |

**代替案（【計画書との差分】）**：計画書は「P/Eコアを固定（taskset）」としていますが、WSL2では効かない可能性が高いです。その場合は次のどちらかにします。

- (a) `.wslconfig` の `processors` を変えたり（8 / 16）、スレッド数を掃引したりして、**スレッド数の効果だけ**を報告する。P/Eの効果は「WSL2では制御不能」と明記する。
- (b) P/Eの実験だけ、**Windowsネイティブ**のPython（OpenVINOはWindowsで動く）で行う。`start /affinity <mask>` で起動し、マスクで論理プロセッサを指定する。Windowsでの論理プロセッサの番号付けは `Get-CimInstance Win32_Processor` やCoreinfo（Sysinternals）で確認する。

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

- [ ] Ubuntu 24.04のWSLで `free -g` の total が約47GB
- [ ] `clinfo -l` でIntel GPUが列挙される
- [ ] `python scripts/check_igpu.py` が `CPU` と `GPU` の両方を表示する（[env/12](12_inference_envs.md) の後）
- [ ] vCPUの推定実験の結果（図と表）があり、「WSL内でP/E固定ができるか」の結論がメモに書かれている
- [ ] Windowsの電源設定を記録した

## 10. 理解度チェック・発展課題

- WSL2のGPUは、Linuxカーネルのドライバ（i915/xe）ではなく何を経由して動いているか。
- iGPUで推論しているとき、CPU側の性能に影響しうる理由を2つ挙げよ（ヒント：電力枠の共有、メモリ帯域）。
- 発展：Gazeboの描画（Iris Xeを使う）とOpenVINO GPU推論を同時に動かすと、何が起きると予想されるか。計画書の「リスク」の表と照らし合わせよ。
