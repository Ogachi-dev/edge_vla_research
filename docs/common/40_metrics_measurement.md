# common/40. 評価指標の取得方法と、測定条件の統一

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`src/evla/analysis/power.py`（4）、`src/evla/analysis/rtf.py`（2）、`scripts/attach_power.py`（1）、`scripts/clock_check.sh`（1）
> - テスト：`pytest tests/analysis/test_power.py tests/analysis/test_extrapolate_rtf.py`
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

## 1. 目的と計画書の対応

- 計画書の評価指標8つについて、**どこで・どう取るか**を決め、取得の仕組みを作る。
- WSL2から見えない指標（電力、温度、クロック）を、Windows側で取って突き合わせる。
- 測定前の手順書（チェックリスト）を作る。
- 計画書の対応箇所：「評価指標と測定方法」「測定条件の統一」、リスク「サーマルスロットリング」

## 2. 前提知識・事前に読む資料

- HWiNFO64（Windows）のセンサーログ（CSV出力）：<https://www.hwinfo.com/>
- Windowsのパフォーマンスカウンタ（`typeperf`）：<https://learn.microsoft.com/windows-server/administration/windows-commands/typeperf>
- RAPL（Running Average Power Limit）の概念：パッケージ電力の測定はCPU内部のカウンタで行う
- **調査で確認した事実**：このWSL2環境には `/sys/class/powercap` が存在せず、RAPLのカウンタが見えない。→ 電力・温度・クロックは Windows 側で測るしかない。

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `docs/measurement_sop.md`（自作） | 測定前のチェックリスト（6節をもとに自分用に作る） |
| `src/evla/analysis/power.py` | HWiNFOのCSVを読み、各runの時間区間の電力・温度・クロックを集計する |
| `src/evla/analysis/rtf.py` | sim時刻と壁時計からRTFを計算する |
| `scripts/attach_power.py` | `results/runs/*` に電力の集計を追記する |
| `scripts/clock_check.sh` | WSLとWindowsの時計のずれを確認する |

## 4. インタフェース仕様

### 4.1 指標の取得方法の一覧

| 指標（計画書） | 定義・単位 | どこで取るか | 実装する章 |
| --- | --- | --- | --- |
| 推論レイテンシ | p50 / p99 [ms] | モードAの `samples.parquet`、ROSノードの `/policy/metrics` | [26](../track1/26_bench_mode_a.md)、[37](../track2/37_policy_node_zmq.md) |
| 実効制御周期 | 新しい行動（軌道）を発行した周期 [Hz] | ポリシーノードのログ。LIBEROでは `steps / (推論回数 × 1推論あたりの時間)` 相当を executor の記録から計算する | [27](../track1/27_closed_loop_b_c.md)、[37](../track2/37_policy_node_zmq.md) |
| タスク成功率 | 成功数 / 試行数 [%]、Wilson区間 | LIBERO：`info["is_success"]`。Gazebo：把持判定の `success` イベント | [27](../track1/27_closed_loop_b_c.md)、[38](../track2/38_gazebo_eval.md) |
| 把持精度 | TCP–キューブのずれ（軸ごと[mm]、ヨー[°]） | `attached` イベントの幾何量 | [32](../track2/32_grasp_judge_node.md) |
| メモリ | ピークRSS [GB] | `PeakRssMonitor`（クライアントとサーバーの両方） | [26](../track1/26_bench_mode_a.md) |
| 電力 | パッケージ電力の平均 [W]、1推論あたりのエネルギー [J] | **Windows側**のHWiNFOのログ → 時刻で突き合わせ | この章 |
| シミュレーション速度 | RTF | sim時刻の増分 ÷ 壁時計の増分 | この章 |
| 温度・クロック | パッケージ温度[°C]、実効クロック[MHz]、スロットリングの有無 | **Windows側**のHWiNFOのログ | この章 |

**【判断】** 電力の指標に「**1推論あたりのエネルギー [J] ＝ 平均電力（アイドル分を引く）× レイテンシ**」を加える。エッジ（バッテリ駆動のロボット）での比較では、ワット数よりエネルギーのほうが意味を持つことが多い。

### 4.2 電力ログの突き合わせ

```python
# src/evla/analysis/power.py
@dataclass(frozen=True)
class PowerSummary:
    mean_package_w: float
    p95_package_w: float
    mean_temp_c: float
    max_temp_c: float
    mean_clock_mhz: float
    throttled: bool              # 温度やクロックの基準で判定（6節）
    idle_package_w: float        # 直前のアイドル区間の平均
    energy_per_infer_j: float | None

def load_hwinfo_csv(path: Path) -> pd.DataFrame:
    """HWiNFOのCSVを読む。文字コード（UTF-8 BOM / cp1252）、日付と時刻の列の結合、
    列名の揺れ（"CPU Package Power [W]" など）を吸収し、tz付きの時刻をindexにする。"""

def summarize_window(df: pd.DataFrame, t0_ns: int, t1_ns: int, idle_window_s: float = 30.0,
                     latency_ms: float | None = None) -> PowerSummary:
    """[t0, t1] の区間を集計する。t0 の直前 idle_window_s 秒をアイドルの基準とする。"""
```

各runの区間（`t_start_ns` の最小と最大）は、26章の `samples.parquet` から取れます。

### 4.3 RTF

```python
# src/evla/analysis/rtf.py
def rtf_series(t_sim_s: npt.ArrayLike, t_wall_ns: npt.ArrayLike, window_s: float = 5.0) -> npt.NDArray:
    """移動窓でのRTF。ROSノードで (sim時刻, time.time_ns()) の組を記録しておけば計算できる。"""
```

Gazeboの統計（`/stats` トピックの `real_time_factor`）も参照値として記録します（`gz topic -e -t /stats` を別プロセスでファイルに流す）。

## 5. 実装手順

1. **Windows側のログ（1時間）**：HWiNFO64を入れ、「センサーのみ」で起動し、ログの間隔を1秒（できれば500ms）にしてCSVに保存する設定をする。CPU Package Power、IA Cores Power、GT（iGPU）Power、CPU Package温度、コアの実効クロック、（あれば）Power Limit の状態の列が入ることを確認する。
2. **時計のずれの確認（30分）**：WSLの `date +%s.%N` と、Windows（PowerShellの `Get-Date -Format o`）を並べて表示し、ずれを測る。スリープ復帰後にWSLの時計がずれることがあるので、**測定の前に毎回確認する**。ずれが大きければ `wsl --shutdown` で再起動する。
3. **`power.py`（2時間）**：CSVの読み込み、区間の集計、アイドルの差し引き。テストには、手で作った小さなCSVを使う。
4. **`attach_power.py`（1時間）**：各runの `summary.json` に `PowerSummary` を追記する。
5. **RTF（1時間）**：34章・38章のノードで、`(sim時刻, 壁時計)` を記録するようにし、`rtf.py` で計算する。
6. **SOPの作成（30分）**：6節を自分用のチェックリストにする。

## 6. ベストプラクティス：測定の手順書（SOP）の骨子

**測定セッションの前**

- [ ] AC電源につなぐ。電源モード「最適なパフォーマンス」、電源プランを確認する（`powercfg /getactivescheme`）
- [ ] 不要なアプリを閉じる（ブラウザ、IDEのインデックス作成、同期ソフト）。Windows Updateを一時停止する
- [ ] 他のWSLのプロセス（前回のGazebo、vla-server）が残っていないか確認する（`ps aux`）
- [ ] HWiNFOのログを開始する（ファイル名にセッションの日時を入れる）
- [ ] WSLとWindowsの時計のずれを確認する
- [ ] 室温を記録する（分かる範囲で）
- [ ] 2分間アイドルにして、温度を落ち着かせる（アイドルの電力の基準にもなる）

**測定中**

- [ ] 設定の間に冷却時間を入れる（例：60秒）
- [ ] 端末に触らない（キーボード操作もCPUを使う）

**測定後**

- [ ] HWiNFOのログを止め、CSVを `results/power/` にコピーする
- [ ] `attach_power.py` で突き合わせる
- [ ] スロットリングの判定が付いたrunを確認し、必要なら測り直す

**スロットリングの判定（【判断】、HWiNFOの値で決める）**：パッケージ温度が95°C以上の時間が区間の10%を超える、または実効クロックがアイドル直後の値の70%未満に落ちた区間がある → `throttled=True`。i7-1360Pは電力上限（PL1/PL2）で制限がかかるので、「電力上限による制限」と「温度による制限」を分けて記録できるとよい（HWiNFOのPower Limit Exceeded系の列）。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 電力のログとrunの区間が合わない | WSLとWindowsの時計のずれ、タイムゾーン（UTC と JST） | 時計を確認する。時刻はすべてUTCのnsで扱い、HWiNFOのローカル時刻を変換する |
| 電力の差が設定間で見えない | 1秒間隔のログでは短い推論を捉えられない | 区間全体の平均を使う。推論を連続で回して区間を長くする |
| iGPU推論でパッケージ電力が下がる | CPUの負荷が減り、電力の配分がGPUに移った | GT Powerも合わせて報告する |
| 長時間の測定で後半が遅い | PL2（短時間の上限）からPL1（持続の上限）に移った | 最初の数十秒と後半を分けて報告する。これ自体がエッジでの重要な知見 |

## 8. ヒント

<details><summary>L1：2つのログを時刻で突き合わせるには</summary>

両方をUTCの時刻（ナノ秒の整数か、tz付きのdatetime）に揃え、runの区間 [t0, t1] でHWiNFOの行を切り出して集計する。1秒間隔のログなので、区間の端の扱い（含めるかどうか）を決めておく。
</details>

<details><summary>L2：使う関数</summary>

`pd.read_csv(path, encoding="utf-8-sig")`（失敗したら `"cp1252"`）、`pd.to_datetime(df["Date"] + " " + df["Time"], format=...)`、`.dt.tz_localize("Asia/Tokyo").dt.tz_convert("UTC")`、`df.loc[t0:t1]`、列名の部分一致：`[c for c in df.columns if "Package Power" in c]`
</details>

<details><summary>L3：summarize_window の疑似コード</summary>

```
w = df.loc[ts(t0_ns):ts(t1_ns)]; idle = df.loc[ts(t0_ns) - idle_window_s : ts(t0_ns)]
p = w[pkg_col]; p_idle = idle[pkg_col].mean()
energy = (p.mean() - p_idle) * latency_ms / 1000 if latency_ms else None
throttled = (w[temp_col] >= 95).mean() > 0.1 or w[clk_col].min() < 0.7 * idle[clk_col].median()
return PowerSummary(p.mean(), p.quantile(.95), w[temp_col].mean(), w[temp_col].max(), w[clk_col].mean(), throttled, p_idle, energy)
```
</details>

## 9. 完了条件（DoD）

- [ ] HWiNFOのCSVに必要な列が入っていることを確認した
- [ ] `power.py` のテストが通る
- [ ] 少なくとも1つのモードAのrunに、電力・温度・クロック・1推論あたりのエネルギーが付いている
- [ ] Gazeboの評価のrunにRTFが付いている
- [ ] `docs/measurement_sop.md` があり、測定セッションで実際に使った

## 10. 理解度チェック・発展課題

- 「平均電力が低い設定」と「1推論あたりのエネルギーが低い設定」が違うことはありうるか。例を挙げよ。
- アイドルの電力を差し引く理由と、差し引くことの問題点は何か。
- 発展：Jetson Orinでは `tegrastats` で電力が取れる。i7-1360Pの測定値をJetsonと比べるとき、何を揃える必要があるか（→ [41](41_analysis_reporting.md)）。
