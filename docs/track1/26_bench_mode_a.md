# track1/26. モードA：単体ベンチ（計時・統計・メモリ）

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**16個**（`src/evla/bench/` の `timing.py` 6、`memory.py` 4、`mode_a.py` 5、`scripts/bench_mode_a.py` 1）
> - ゴール：`pytest tests/bench` が通り、`scripts/bench_mode_a.py` で PyTorch CPU のスレッド数別の結果が `results/runs/` にたまる
> - 進め方：Step 1〜2 はモデルなしでできる。Step 3 以降は23章（と25章）が動いている前提

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
| `scripts/bench_mode_a.py` | 1設定を測るCLI（28章のスイープから子プロセスとして呼ばれる） |
| `configs/bench/*.yaml` | 設定例（`torch_cpu.yaml`、`vlacpp_cpu.yaml`、掃引の基準の `base.yaml`） |
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

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章で作るものは、以降のすべての数値の土台になります。**測り方そのものを疑う**つもりで確かめてください。

### Step 0：準備（5分）

```bash
source .venv-bench/bin/activate
grep -n "【穴" src/evla/bench/{timing,memory,mode_a}.py scripts/bench_mode_a.py
pytest tests/bench -q          # まだ失敗する
```

- [ ] 穴16個が表示され、テストが `NotImplementedError` で失敗する

### Step 1：計時と統計 ― `timing.py`（穴①〜⑥、45分）

**読む**：4節の `summarize_latencies` と、6節の「p99 を出す条件」。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 複数のパーセンタイルを1回で求める NumPy の関数。リストで `[50, 90, 95, 99]` を渡せる |
| ② | 母集団の標準偏差（n で割る）と、標本の標準偏差（n−1 で割る）の違い |
| ③ | サンプルが少ないと、上位1%に1点も入らず p99 ≒ 最大値になる。何点以上なら出してよいか（定数がファイルの上にある） |
| ④ | 変動係数。平均の大きさが違う設定どうしで、ばらつきを比べられる |
| ⑤ | 計時に使う時計。壁時計（`time.time`）は NTP の補正で戻ることがある。単調増加でナノ秒の整数を返す方 |
| ⑥ | 単位の変換。1 ms は何 ns か |

**動かす**

```bash
pytest tests/bench/test_timing.py -v -k "summary or p99 or stopwatch"
python -c "
from evla.bench.timing import summarize_latencies as s
import pprint; pprint.pprint(s([1, 2, 3, 4]))
"
```

**確かめる**

- [ ] 3つのテストが `PASSED`
- [ ] `p50: 2.5`、`p99: nan`（n=4 < 100）、`std` が約 1.29（n−1 で割った値。n で割ると 1.118）

### Step 2：メモリの監視 ― `memory.py`（穴①〜④、45分）

**読む**：`cat /proc/self/status | grep -E "VmRSS|VmHWM"` を実行して、行の形を見る。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `'VmRSS:   12345 kB'` から数字だけを取り出す（`split()` は連続した空白をまとめて区切る） |
| ② | 「今」の常駐メモリのキー（`VmHWM` はカーネルが記録しているピークで、終了時の照合に使う） |
| ③ | これまでのピークと今の値（kB → MB）の大きい方 |
| ④ | `threading.Event` の `wait(timeout)` は、待つ（sleep の代わり）と同時に、止められたかどうかを返す。`sleep` だと止めるのに最大 interval だけ遅れる |

**動かす**

```bash
pytest tests/bench/test_timing.py -v -k peak_rss
```

**確かめる**

- [ ] `test_peak_rss_sees_allocation` が `PASSED`（50MB の確保が、ピークに 30MB 以上の増加として見える）
- [ ] 5秒考える：別スレッドで 0.05 秒ごとに見る方式は、何を取りこぼしうるか（だから `VmHWM` とも照合する）

### Step 3：1設定の測定 ― `mode_a.py`（穴①〜⑤、1.5時間）

**読む**：4節の `ModeAConfig` と `samples.parquet` の列。`run_mode_a` の順序（設定の保存 → 監視開始 → サーバー起動 → 方策の読み込み → ウォームアップ → 計測 → 保存）を確認する。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `ExitStack`：with 文を動的に（条件付きで）積み重ねる道具。抜けるときは**積んだ逆順**に後片付けされる（方策 → 監視 → サーバー） |
| ② | 固定観測を順番に使い回す。添字が観測の数を超えないようにする |
| ③ | ウォームアップを捨てずに、負の番号で残す（`-3, -2, -1, 0, 1, …`）。後から「何回目で安定したか」を見られる |
| ④ | pandas の DataFrame を Parquet で保存するメソッド |
| ⑤ | `is_warmup` が False の行。pandas の真偽の列を反転する演算子 |

### Step 4：CLI から動かす ― `scripts/bench_mode_a.py`（穴①、20分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 環境変数を設定する辞書。**torch を import する前**に設定しないと効かない（だから `main` の中で、重い import より先に呼んでいる） |

**動かす**

```bash
python scripts/bench_mode_a.py --config configs/bench/torch_cpu.yaml --set n_measure=20
ls results/runs/ | tail -1
RUN=results/runs/$(ls results/runs | tail -1)
ls $RUN; python -c "
import pandas as pd; df = pd.read_parquet('$RUN/samples.parquet')
print(df[['iter', 'is_warmup', 'obs_index', 'latency_ms', 'vcpu']].head(6))
print(df.filter(like='bd_').columns.tolist())
"
```

**確かめる**

- [ ] `<run_id>: p50=...ms p90=...ms load=...s rss=...MB` が1行表示される
- [ ] run のディレクトリに `config.yaml`、`env.json`、`samples.parquet`、`summary.json`、`artifacts/` がある
- [ ] `iter` が `-3, -2, -1, 0, 1, 2`、`is_warmup` が最初の3行だけ True、`obs_index` が 0, 1, 2, … と進む
- [ ] `bd_preprocess_ms` などの内訳の列がある
- [ ] `env.json` の `thread_env` に `OMP_NUM_THREADS: "8"` が入っている（④が効いている証拠）

### Step 5：ベースラインを測る（1時間）

PyTorch CPU で、スレッド数を変えて各50回測ります。

```bash
for n in 1 2 4 8 12 16; do
  python scripts/bench_mode_a.py --config configs/bench/torch_cpu.yaml --set policy.n_threads=$n --tag threads
done
python -c "
import json, glob
for p in sorted(glob.glob('results/runs/*_t1_modeA_*/summary.json')):
    d = json.load(open(p)); print(d['describe'].get('num_threads'), round(d['summary']['p50'], 1), round(d['summary']['cv'], 3))
"
```

- [ ] スレッド数ごとの p50 と cv が並ぶ。p50 とスレッド数のグラフを作る（28章の `plots.py` ができるまでは手で作ってよい）
- [ ] 同じ設定（例：8スレッド）を**3プロセス**で測り、p50 の変動係数を確認する（6節）
- [ ] vla.cpp でも1回測る：`python scripts/bench_mode_a.py --config configs/bench/vlacpp_cpu.yaml`。`summary.json` の `peak_rss_server_mb` に値が入っている

### Step 6：P/Eコアの実験（1時間、[env/10](../env/10_wsl2_and_igpu.md) の結論に従う）

- vCPUに速い群があった場合：`taskset -c <速い群> python scripts/bench_mode_a.py ...` と `taskset -c <遅い群> ...` で同じ設定を測る。`env.json` の `cpu_affinity` に指定が残っていることを確かめる。
- なかった場合：`.wslconfig` の `processors=8` と `16` を比べるか、Windowsネイティブでの測定に回す。

- [ ] どちらにしたかと、その結論をメモに残す

### Step 7：仕上げ（10分）

- [ ] 4ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/bench scripts/bench_mode_a.py` を通す
- [ ] `pytest tests/bench` がすべて通る
- [ ] コミットする

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
