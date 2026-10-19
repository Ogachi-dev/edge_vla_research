# track1/26. モードA：単体ベンチ（計時・統計・メモリ）

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`src/evla/bench/timing.py`（6）、`src/evla/bench/memory.py`（4）、`src/evla/bench/mode_a.py`（5）、`scripts/bench_mode_a.py`（1）
> - テスト：`pytest tests/bench`
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

## 1. 目的と計画書の対応

- 推論だけを繰り返して、**レイテンシの分布・起動時間・メモリ**を正しく測る基盤を作る。
- スレッド数（とP/Eコア）の効果を測る。
- 計画書の対応箇所：トラック1「測定モードA」「掃引軸（CPU：スレッド数、P/E割り当て）」、評価指標（推論レイテンシp50/p99、メモリ）、測定条件の統一、日程D2〜D4

## 2. 前提知識・事前に読む資料

- [02_best_practices.md](../02_best_practices.md) の「計測の作法」
- パーセンタイルの推定：`numpy.percentile` の `method` 引数（既定は線形補間）
- Linuxのメモリ指標：RSS、VmHWM（ピークRSS）、`getrusage` の `ru_maxrss`（Linuxでは**KB単位**）
- psutil：<https://psutil.readthedocs.io/>
- vla.cppのベンチ手順（`docs/benchmark/README.md`）：ウォームアップ3回＋計測20回を1プロセスとし、3プロセスのうち平均が最小のものを報告する

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `src/evla/bench/timing.py` | `Stopwatch`、`summarize_latencies()` |
| `src/evla/bench/memory.py` | `PeakRssMonitor`：別スレッドで任意のPIDのRSSを監視する |
| `src/evla/bench/mode_a.py` | `run_mode_a(cfg) -> RunResult`：1設定×1プロセス分の測定 |
| `src/evla/bench/schema.py` | 結果の列定義（Parquetのスキーマ） |
| `scripts/bench_mode_a.py` | 1設定を測るCLI（28章のスイープから子プロセスとして呼ばれる） |
| `configs/bench/*.yaml` | 設定例（torch_cpu_t8.yaml など） |
| `tests/bench/test_timing.py` | 統計関数のテスト、`sleep` を使った計時の健全性テスト |

## 4. インタフェース仕様

```python
# src/evla/bench/mode_a.py
@dataclass(frozen=True)
class ModeAConfig:
    policy: PolicyConfig
    fixed_obs: Path = Path("data/fixed_obs/libero_object_t0.npz")
    n_warmup: int = 3
    n_measure: int = 50                 # 予備の掃引は50、上位の設定は200以上（6節）
    obs_cycle: bool = True              # 固定観測を順番に使い回す（毎回同じ1枚にしない）
    fixed_noise_seed: int | None = 0
    server: VlaServerConfig | None = None   # runtime == "vlacpp" のときに起動する
    cooldown_s: float = 0.0             # 測定前の待機（温度を揃える）
    tag: str = ""

@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    load_s: float                  # モデルの読み込み（GPUならコンパイルを含む）
    first_infer_ms: float          # ウォームアップの1回目
    summary: dict[str, float]      # summarize_latencies の結果
    peak_rss_mb: float
    peak_rss_server_mb: float | None

def run_mode_a(cfg: ModeAConfig, run_dir: Path) -> RunResult: ...
```

```python
# src/evla/bench/timing.py
def summarize_latencies(ms: npt.ArrayLike) -> dict[str, float]:
    """{"n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max", "cv"} を返す。
    n < 100 のときは p99 を NaN にする（信頼できないため。理由は6節）。"""

class Stopwatch:
    """with Stopwatch() as sw: ... ; sw.ms で経過ミリ秒（perf_counter_ns ベース）。"""
```

```python
# src/evla/bench/memory.py
class PeakRssMonitor:
    """with PeakRssMonitor(pids=[os.getpid(), server_pid], interval_s=0.05) as m: ...
    m.peak_mb[pid] で各プロセスのピークRSS。終了時に /proc/<pid>/status の VmHWM とも照合する。"""
```

**samples.parquet の列（1推論＝1行）**

| 列 | 型 | 内容 |
| --- | --- | --- |
| `run_id` | str | |
| `iter` | int | 0から（ウォームアップは負の番号にして残す） |
| `is_warmup` | bool | |
| `obs_index` | int | どの固定観測を使ったか |
| `latency_ms` | float | `ChunkResult.latency_ms` |
| `bd_<name>_ms` | float | 内訳（ランタイムによって列が変わる。なければNaN） |
| `t_start_ns` | int | 開始時刻（`time.time_ns()`。Windows側の温度ログとの突き合わせ用） |
| `vcpu` | int | 開始時に動いていたCPU（`os.sched_getcpu()`、参考値） |

`config.yaml` と `env.json`（[01](../01_repository_design.md)）に加え、`summary.json` に `RunResult` を保存します。

## 5. 実装手順

1. **計時と統計（1時間）**：`Stopwatch` と `summarize_latencies` を実装し、テストする（`[1,2,3,4]` のp50＝2.5、`sleep(0.02)` が約20ms、など）。
2. **メモリ監視（1時間）**：`PeakRssMonitor` を実装する。テストでは、大きな配列を確保して解放するプロセスのピークが、確保量と同程度になることを確かめる。
3. **`run_mode_a`（2時間）**
   - 順序：環境変数の設定 → 監視開始 → （vlacppなら）サーバー起動 → `make_policy` → 読み込み時間の記録 → ウォームアップ → 計測ループ → 保存。
   - スレッド関連の環境変数（`OMP_NUM_THREADS`、`VLA_N_THREADS` など）は、**ライブラリのimport前**に設定する必要がある。そのため `run_mode_a` は**新しいプロセスで実行する**前提にする（28章のスイープが `subprocess` で呼ぶ）。
4. **CLI（30分）**：`scripts/bench_mode_a.py --config configs/bench/torch_cpu_t8.yaml [--set policy.n_threads=4]`
5. **ベースラインの測定（1時間）**：PyTorch CPUで、スレッド数を 1/2/4/8/12/16 にして各50回測る。
6. **P/Eコアの実験（1時間、[env/10](../env/10_wsl2_and_igpu.md) の結論に従う）**
   - vCPUに速い群があった場合：`taskset -c <速い群>` と `taskset -c <遅い群>` で同じ設定を測る。
   - なかった場合：`.wslconfig` の `processors=8` と `16` を比べるか、Windowsネイティブでの測定に回す。

## 6. ベストプラクティス

- **1設定＝1プロセス**：スレッドプールの設定やメモリの断片化が、次の設定に持ち越されないようにする。読み込み時間と初回推論時間も、毎回きれいに測れる。
- **p99を名乗るにはサンプルが要る**：n=20では、p99は事実上「最大値」になる。p99を報告するなら n≥100（できれば200）にする。予備の掃引では p50・p90・max を報告し、上位の設定だけ n=200 で測り直す。1回2秒なら200回で約7分かかる。
- **固定観測を回す**：同じ画像1枚だけだと、キャッシュに乗って実際より速く見えることがある。20〜30枚を順番に使う。
- **ウォームアップも保存する**：捨てずに `is_warmup=True` で残す。「初回推論が遅い」こと自体が、ロボットの起動時の性能として報告に値する。
- **温度を揃える**：設定の間に `cooldown_s`（例：60秒）を入れる。可能なら、Windows側で温度が一定値以下になるまで待つ（[40](../common/40_metrics_measurement.md)）。
- **交互に測る**：スレッド数の比較は、1→2→4→…→16 と1回ずつ測る1周を、3周繰り返す（28章のスイープでシャッフルする）。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `ru_maxrss` の値が1000倍ずれる | Linuxでは KB、macOSでは bytes | 単位を明記し、`/proc/self/status` の VmHWM と照合する |
| vla.cppのメモリが小さすぎる | 測っているのがクライアントのプロセスだけ | サーバーのPIDも監視する |
| OV-GPUのメモリがRSSに現れない | iGPUの割り当てがプロセスのRSSに計上されない場合がある（**【要確認】**） | `psutil.virtual_memory().used` の増分も記録し、両方を報告する |
| スレッド数を変えても結果が同じ | 環境変数をimportの後に設定した | `describe()` で実際のスレッド数を確認する |
| 測定の後半ほど遅くなる | 熱によるスロットリング | `t_start_ns` と温度ログを突き合わせる。冷却を入れる |
| 計測値に周期的なスパイクがある | 他のプロセス（Windows Update、インデクサ、Gazeboの残骸） | 測定前に `ps aux` を記録し、不要なものを止める |

## 8. ヒント

<details><summary>L1：ピークRSSを外から測るには</summary>

一定間隔で `/proc/<pid>/status` を読み、最大値を更新し続ける。読み取りはメインの測定を邪魔しないよう、別スレッドで、間隔を空けて行う。終了時にカーネルが記録しているピーク値（VmHWM）とも比べる。
</details>

<details><summary>L2：使うAPI</summary>

`psutil.Process(pid).memory_info().rss`、`/proc/<pid>/status` の `VmHWM:` 行、`threading.Event`（停止の合図）、`os.sched_getcpu()`、`pandas.DataFrame(...).to_parquet(path)`、`np.percentile(x, [50, 90, 95, 99])`
</details>

<details><summary>L3：run_mode_a の疑似コード</summary>

```
set_thread_env(cfg.policy.n_threads)                  # importより前（プロセスの冒頭）
obs_set = load_fixed_obs(cfg.fixed_obs); noise = make_fixed_noise(...) if seed is not None else None
with PeakRssMonitor([os.getpid()]) as mon, maybe_server(cfg.server) as srv:
    if srv: mon.add(srv.pid)
    with Stopwatch() as sw: policy = make_policy(cfg.policy)
    load_s = sw.ms / 1000
    rows = []
    for i in range(-cfg.n_warmup, cfg.n_measure):
        obs = obs_set[i % len(obs_set)] if cfg.obs_cycle else obs_set[0]
        t_ns = time.time_ns(); r = policy.predict_chunk(obs, noise)
        rows.append(row(i, r, t_ns))
    policy.close()
save(rows, summarize([r for r in rows if not warmup]), mon.peak_mb)
```
</details>

## 9. 完了条件（DoD）

- [ ] `pytest tests/bench` が通る
- [ ] PyTorch CPUのスレッド数別（6水準×50回）の結果が `results/runs/` にあり、p50とスレッド数のグラフがある
- [ ] vla.cpp（CPU）で、サーバーのピークRSSが記録されている
- [ ] P/Eコアの扱いについての結論（実験した、またはWSL2では不可なので代替した）が書かれている
- [ ] 同じ設定を3プロセスで測ったときのp50のばらつき（変動係数）を確認した

## 10. 理解度チェック・発展課題

- n=50のサンプルから求めたp99は、何番目に大きい値に相当するか。それはなぜ不安定なのか。
- スレッド数を16にしたときに8より遅くなるとしたら、考えられる原因を3つ挙げよ。
- 発展：推論中のCPU使用率をvCPUごとに記録すると、何が分かるか。実装して、スレッド数8の場合に実際に何個のvCPUが動いているか確かめよ。
