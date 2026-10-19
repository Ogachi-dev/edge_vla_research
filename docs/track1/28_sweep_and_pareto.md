# track1/28. 掃引の設計・実行と、パレート図

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**15個**（`src/evla/sweep/` の `spec.py` 4、`runner.py` 3、`src/evla/analysis/` の `load.py` 1、`pareto.py` 2、`plots.py` 3、`scripts/sweep.py` 1、`scripts/make_report_tables.py` 1）
> - ゴール：`pytest tests/sweep tests/analysis/test_pareto.py` が通り、`python scripts/sweep.py configs/sweep/stage1.yaml --dry_run` が設定数と所要時間を表示する
> - 進め方：Step 1〜5 はモデルなしでできる（図は作り物のデータで試す）。Step 6 で、実際の掃引を夜に回す

## 1. 目的と計画書の対応

- 掃引軸の組み合わせを**予算内に収まるよう段階的に**設計し、自動で回す仕組みを作る。
- 結果を集計し、ベンチマーク表とパレート図（レイテンシ×成功率）を作る。
- 計画書の対応箇所：トラック1「掃引軸」「最適な推論設定（パレート）」、日程D4（掃引）、D7（パレート図）、成果物「ベンチマーク表」「パレート図」

## 2. 前提知識・事前に読む資料

- パレート最適（多目的最適化の非劣解）の定義
- pandas の `groupby`、`merge`、`pivot_table`
- matplotlib の基本（`fig, ax = plt.subplots()`、`errorbar`、`savefig`）

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `configs/sweep/stage{1..4}.yaml` | 段階ごとの掃引の定義（`stage1.yaml` は用意済み。2〜4は5節 Step 6 で自分で作る） |
| `src/evla/sweep/spec.py` | 掃引の定義を設定のリストに展開する |
| `src/evla/sweep/runner.py` | 子プロセスで1設定ずつ実行する。再開・冷却・シャッフルに対応する |
| `scripts/sweep.py` | CLI |
| `src/evla/analysis/load.py` | `results/runs/*` を1つのDataFrameに読み込む |
| `src/evla/analysis/pareto.py` | `pareto_front()` |
| `src/evla/analysis/plots.py` | 図（レイテンシの分布、スレッド数、パレート図、遅延許容曲線） |
| `scripts/make_report_tables.py` | `reports/` に表（Markdown / CSV）と図（PNG / SVG）を出力する |

## 4. インタフェース仕様

### 4.1 掃引の定義（YAML）

```yaml
# configs/sweep/stage1.yaml（例。値は自分で決める）
name: stage1_runtime_precision
mode: A
base: configs/bench/base.yaml          # ModeAConfig の既定値
rounds: 3                              # 全設定を3周する（周ごとに順序をシャッフル）
cooldown_s: 60
grid:                                  # 直積で展開する
  - policy.runtime: [torch]
    policy.precision: [fp32, bf16]
  - policy.runtime: [ov]
    policy.device: [CPU, GPU]
    policy.model: [data/models/ov/smolvla-libero-fp16-ov, data/models/ov/smolvla-libero-w8a16-ov]
  - policy.runtime: [vlacpp]
    server.binary: [third_party/vla.cpp/build/vla-server]
    server.gguf: [data/models/gguf/smolvla-libero.gguf, data/models/gguf/smolvla-libero-q8_0.gguf,
                  data/models/gguf/smolvla-libero-q4_0.gguf]
exclude:                               # 意味のない組み合わせを除く
  - {policy.runtime: ov, policy.device: GPU, policy.n_threads: 16}
```

```python
# src/evla/sweep/spec.py
@dataclass(frozen=True)
class SweepItem:
    overrides: dict[str, Any]      # "a.b": value
    key: str                       # 設定のハッシュ（再開と結合に使う）

def expand(spec_path: Path) -> list[SweepItem]:
    """grid（リストの各要素ごとに直積をとり、全体を連結）を展開し、exclude に合うものを除く。"""

def policy_key(cfg: Mapping[str, Any]) -> str:
    """policy.* の部分だけから計算したハッシュ。モードAの結果とモードBの結果を結合するキーになる。"""
```

```python
# src/evla/sweep/runner.py
def run_sweep(spec_path: Path, resume: bool = True, dry_run: bool = False) -> None:
    """各周で順序をシャッフルし、各設定を `python scripts/bench_mode_a.py --config ... --set ...` として子プロセスで実行する。
    resume=True なら、同じ key・同じ周の summary.json がある設定を飛ばす。
    子プロセスが失敗したら記録して次へ進む（一晩の実行が1つの失敗で止まらないように）。
    dry_run=True なら、展開した設定の数と予想される所要時間だけを表示する。"""
```

```python
# src/evla/analysis/pareto.py
def pareto_front(df: pd.DataFrame, minimize: str, maximize: str) -> pd.DataFrame:
    """minimize 列は小さいほど良く、maximize 列は大きいほど良いとして、非劣な行だけを返す（minimize の昇順）。"""
```

### 4.2 段階的な掃引計画（予算から逆算する）

全組み合わせ（ランタイム5×精度5×解像度3×視点2×スレッド6＝900）は回せません。段階を分けて、前の段の上位だけを次の段に進めます。

| 段 | モード | 変える軸 | 固定 | 試行 | 規模の目安 |
| --- | --- | --- | --- | --- | --- |
| 1 | A | ランタイム×精度（×デバイス） | 2視点・512px・スレッドは既定 | n=50×3周 | 約20設定 → 約3時間 |
| 2 | A | スレッド数 {1,2,4,8,12,16} | 段1のCPU系の上位3つ | n=50×3周 | 18設定 → 約2.5時間 |
| 3 | A | 視点 {1,2}×解像度 {512, 384, 256} | 段2の上位2つ（PyTorchとvla.cpp） | n=50×1周 | 12設定 → 約1時間 |
| 4 | B | 段1〜3の候補から、速い順と精度の違う代表を5〜8個 | lockstep、n_exec=10 | 50エピソード | 約1時間×設定数 |
| 確認 | A | 最終候補 | | n=200 | p99を出す |

所要時間は `--dry_run` で見積もり、**D4の夜に段1〜3、D5の夜に段4**を回すように計画します。

**【判断】** パレート図の横軸はモードAの **p50**（上位設定ではp90も併記）、縦軸はモードBの成功率にします。モードCの成功率は「遅延の影響」なので、別の図（遅延許容曲線）に分けます。

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。

### Step 0：準備（5分）

```bash
source .venv-bench/bin/activate
grep -n "【穴" src/evla/sweep/{spec,runner}.py src/evla/analysis/{load,pareto,plots}.py scripts/sweep.py scripts/make_report_tables.py
pytest tests/sweep tests/analysis/test_pareto.py -q        # まだ失敗する
```

- [ ] 穴15個が表示され、テストが失敗する

### Step 1：掃引の定義を展開する ― `spec.py`（穴①〜④、45分）

**読む**：4.1節の YAML と、`tests/sweep/test_spec.py`。`grid` は「ブロックごとに直積 → 全ブロックを連結」であることを確認する（ランタイムごとに意味のある軸が違うので、1つの巨大な直積にしない）。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 直積をとる標準ライブラリの関数（`itertools`）。値のリストを引数として**展開して**渡す |
| ② | ①を使う関数を、ブロックごとに呼ぶ |
| ③ | `any` / `all` のどちらを使えば「どのルールにも当てはまらない」になるか |
| ④ | キーが `"policy."` で始まるか。これで、`n_measure` などを変えてもモードAとモードBの結果を同じキーで結合できる |

**動かす**

```bash
pytest tests/sweep/test_spec.py -v
python -c "
from pathlib import Path
from evla.sweep.spec import expand
s = expand(Path('configs/sweep/stage1.yaml'))
print(len(s.items), s.rounds); [print(' ', it.key, it.overrides) for it in s.items]
"
```

**確かめる**

- [ ] 2つのテストが `PASSED`
- [ ] `stage1.yaml` は 2（torch）＋ 4（ov：デバイス2×モデル2）＝ **6設定 × 3周**。key がすべて違う

### Step 2：子プロセスで順に実行する ― `runner.py`（穴①〜③）と `scripts/sweep.py`（穴①）（1時間）

**読む**：4.1節の `run_sweep` の docstring。なぜ1設定ごとに**別プロセス**にするのか（26章：スレッドの環境変数は import 前にしか効かない／メモリのピークを設定ごとに測りたい）を確認する。

**埋める**

| 穴 | ねらい |
| --- | --- |
| `runner.py` ① | 周ごとに順序をシャッフルする（時間による温度やバックグラウンドの影響を、設定の間で均す）。シードを周の番号にすると、途中から再開しても同じ順になる |
| `runner.py` ② | 再開モードで、完了の印（`.done` ファイル）があれば飛ばす |
| `runner.py` ③ | `subprocess` で子プロセスを実行し、終わるまで待ち、標準出力と標準エラーを**文字列で**受け取る（`check=True` にしないのは、失敗しても次へ進むため） |
| `sweep.py` ① | `argparse` のフラグ（値を取らない） |

**動かす**

```bash
python scripts/sweep.py configs/sweep/stage1.yaml --dry_run --sec_per_item 120
```

**確かめる**

- [ ] `stage1_runtime_precision: 6 設定 × 3 周 = 18 実行` と、見積もり時間・各設定の key が表示される
- [ ] 小さく試運転する（夜の本番の前に必ず。6節）：`n_measure` を 5 にした小さな掃引ファイル（例：`configs/sweep/smoke.yaml`、torch の1行だけ、`rounds: 2`）を作って回し、`results/sweeps/<name>/` に `*_r0.done`・`*_r1.done` と `sweep_log.jsonl` ができること、**もう一度実行すると全部飛ばされる**ことを確かめる

### Step 3：結果を1つの表に読み込む ― `load.py`（穴①、20分）

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 再帰。入れ子の辞書を、`親のキー.` を接頭辞にして自分自身に渡す |

**動かす**

```bash
python -c "
from evla.analysis.load import flatten, load_runs
print(flatten({'policy': {'runtime': 'ov', 'extra': {'CACHE_DIR': 'x'}}, 'n_measure': 50}))
df = load_runs(pattern='*_t1_modeA_*'); print(df.shape); print(df.filter(like='result.summary').columns.tolist()[:5])
"
```

- [ ] `{'policy.runtime': 'ov', 'policy.extra.CACHE_DIR': 'x', 'n_measure': 50}`
- [ ] 26章で測った run が行として読める（`result.summary.p50` などの列がある）

### Step 4：パレート境界 ― `pareto.py`（穴①②、30分）

**読む**：`tests/analysis/test_pareto.py` の5点の例を、**紙に散布図として描き**、どれが境界に乗るかを先に答えておく。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `sort_values` で列ごとに昇順・降順を指定する（リストで渡せる）。同じ遅延なら成功率の高い方を先に見るため |
| ② | 「これまでの最大」を**真に**超えたときだけ残す（同じ値は劣っているとみなす） |

**動かす**

```bash
pytest tests/analysis/test_pareto.py -v
```

- [ ] 2つのテストが `PASSED`。紙に描いた答え（`a, b, d`）と一致する

### Step 5：図と表 ― `plots.py`（穴①〜③）と `scripts/make_report_tables.py`（穴①）（1.5時間）

**埋める**

| 穴 | ねらい |
| --- | --- |
| `plots.py` ① | `errorbar` で**上下非対称**のエラーバーを渡す引数（Wilson の区間は非対称）。すぐ上の行で `(2, n)` の配列を作っている |
| `plots.py` ② | 2本の線（下限と上限）の間を塗る関数。信頼区間を帯で見せる |
| `plots.py` ③ | ECDF の縦軸。n 個の点に 1/n, 2/n, …, 1 を割り当てる |
| `make_report_tables.py` ① | 同じ設定の周ごとの値をまとめる代表値。1周だけ外れた値に引きずられないもの |

**動かす**：作り物のデータで、図の見た目を先に確かめます。

```bash
python - <<'EOF'
from pathlib import Path
import pandas as pd
from evla.analysis.plots import pareto_plot, latency_tolerance_plot
df = pd.DataFrame({"runtime": ["torch", "ov", "ov", "vlacpp"], "precision": ["fp32", "fp16", "w8a16", "q8_0"],
                   "p50_ms": [900, 400, 350, 250], "success_rate": [0.9, 0.88, 0.8, 0.86],
                   "ci_lo": [0.79, 0.76, 0.67, 0.74], "ci_hi": [0.96, 0.94, 0.89, 0.93]})
pareto_plot(df, Path("reports/_try/pareto"))
curve = pd.DataFrame({"mode": ["sync"] * 3, "n_exec": [10] * 3, "latency_ms": [0, 500, 1000],
                      "success_rate": [0.9, 0.7, 0.4], "ci_lo": [0.79, 0.56, 0.27], "ci_hi": [0.96, 0.81, 0.54]})
latency_tolerance_plot(curve, Path("reports/_try/tolerance"))
EOF
ls reports/_try/
python scripts/make_report_tables.py
```

**確かめる**

- [ ] `reports/_try/` に `pareto.png/svg` と `tolerance.png/svg` ができる。開いて、エラーバーが上下非対称・帯が塗られている・パレート境界の破線が `vlacpp → ov(fp16) → torch` を結んでいることを確かめる
- [ ] `make_report_tables.py` が、26章で測った run からベンチマーク表を表示し、`reports/benchmark_table.md` を書く
- [ ] 確認が済んだら `reports/_try/` は消す

### Step 6：段階的な掃引を回す（D4〜D5の夜）

- [ ] `configs/sweep/stage1.yaml` の TODO（vla.cpp の行）を埋め、`--dry_run` で見積もる
- [ ] 4.2節の表に従い、`stage2.yaml`（スレッド数）、`stage3.yaml`（視点×解像度）を作る。**D4の夜に段1〜3**を回す
- [ ] 段4（モードB）は `mode: B`、`base: configs/closed_loop/modeB_lockstep.yaml` で作り、**D5の夜**に回す
- [ ] 朝、`results/sweeps/<name>/sweep_log.jsonl` で失敗（`rc != 0`）がないか確かめる。あれば `stderr_tail` を読み、直してから**同じコマンドで再開**する（終わった設定は飛ばされる）
- [ ] D7：`make_report_tables.py` の TODO（パレート図・遅延許容曲線）を埋め、`reports/` に出力する

### Step 7：仕上げ（10分）

- [ ] 7ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/sweep src/evla/analysis/{load,pareto,plots}.py scripts/sweep.py scripts/make_report_tables.py` を通す
- [ ] `pytest tests/sweep tests/analysis/test_pareto.py` がすべて通る
- [ ] コミットする

## 6. ベストプラクティス

- **夜に回す前に、昼に小さく回す**：`n_measure=3`、設定2つで、スイープ全体が最後まで走り、集計と図が出ることを確認してから本番を回す。
- **失敗しても止まらない**：一晩の実行で、3つ目の設定のエラーで残りが全部止まると1日を失う。失敗は記録して次へ進む。
- **再開できるようにする**：WSLが落ちる、Windows Updateで再起動する、などは実際に起こる。完了した設定は飛ばせるようにする。
- **生の結果から図を作り直せるようにする**：図はスクリプトで生成し、手で編集しない。データが増えたら再実行するだけにする。
- **パレート図には不確かさを描く**：成功率のエラーバーがない図は、差を過大に見せる。
- **「負けた」設定も表に残す**：Q4_0がだめだった、Iris Xeは遅かった、という結果もエッジ推論の知見として価値がある。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| モードAとモードBの結果が結合できない | キーにモード固有の設定（n_measureなど）が入っている | `policy_key` は `policy.*` だけから計算する |
| 同じ設定の3周の値が大きく違う | 温度、他プロセス、OSの電源設定の変化 | 冷却時間を延ばす。Windows側のログと突き合わせる |
| 掃引の途中でディスクが埋まる | 失敗動画を全部保存している | `save_video="failures"`、解像度を下げる |
| パレート境界がほぼ1点になる | 成功率がどれも同じで、差が区間内 | 試行数を増やすか、「同等」として扱い、レイテンシとメモリで選ぶ |

## 8. ヒント

<details><summary>L1：パレート境界の求め方</summary>

レイテンシの小さい順に並べ、上から見ていく。それまでに見た成功率の最大値を更新した点だけが境界に乗る。
</details>

<details><summary>L2：使う関数</summary>

`itertools.product`、`random.Random(seed + round).shuffle`、`subprocess.run(cmd, timeout=..., capture_output=True)`、`pd.json_normalize`、`df.sort_values`、`ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o")`、`ax.step(..., where="post")`（境界線）
</details>

<details><summary>L3：pareto_front の疑似コード</summary>

```
d = df.sort_values([minimize, maximize], ascending=[True, False])
best = -inf; keep = []
for idx, row in d.iterrows():
    if row[maximize] > best: keep.append(idx); best = row[maximize]
return d.loc[keep]
```
</details>

## 9. 完了条件（DoD）

- [ ] `scripts/sweep.py --dry_run configs/sweep/stage1.yaml` が設定数と予想所要時間を表示する
- [ ] 段1〜3の結果が揃い、ベンチマーク表（Markdown）が `reports/` にある
- [ ] 段4の結果と結合したパレート図（エラーバー付き）が `reports/` にある
- [ ] 最終候補の設定について、n=200のp99がある
- [ ] トラック2に渡す「最適設定」（ランタイム、精度、スレッド、視点、解像度）と、その根拠が `reports/track1_selection.md` に書かれている

## 10. 理解度チェック・発展課題

- 段階的な掃引は、全組み合わせの掃引と比べて何を見落とす可能性があるか（交互作用の観点から）。
- パレート境界上の設定が複数あるとき、トラック2に持ち込む1つをどう選ぶか。制御周期の要求（30Hz・n_exec=25なら0.83秒ごとに推論が必要）から考えよ。
- 発展：メモリを第3の軸にして、3目的のパレート集合を求めよ。2次元の図にどう表現するか（マーカーの大きさ、など）。
