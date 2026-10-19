# track1/28. 掃引の設計・実行と、パレート図

> **穴埋め版**（進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - 埋めるファイル（穴の数）：`src/evla/sweep/spec.py`（4）、`src/evla/sweep/runner.py`（3）、`src/evla/analysis/load.py`（1）、`src/evla/analysis/pareto.py`（2）、`src/evla/analysis/plots.py`（3）、`scripts/sweep.py`（1）、`scripts/make_report_tables.py`（1）
> - テスト：`pytest tests/sweep tests/analysis/test_pareto.py`
> - 「5. 実装手順」は、ゼロから書く場合の手順です。穴埋め版では、手順の中の**確認作業**（動作確認・測定・完了条件）を行い、コードは穴を埋めるだけで足ります。

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
| `configs/sweep/stage{1..4}.yaml` | 段階ごとの掃引の定義 |
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

## 5. 実装手順

1. `spec.py`：展開と除外、ハッシュ。テストで「展開数が期待どおりか」「同じ設定は同じkeyになるか」を確かめる（1.5時間）。
2. `runner.py`：子プロセスの実行、再開、シャッフル、冷却、失敗時の継続、進捗表示（2時間）。
3. `load.py`：`results/runs/*/config.yaml` と `summary.json` を読んで、列を平坦化したDataFrameにする（`policy.runtime` などのドット区切りの列名）（1時間）。
4. `pareto.py`：実装とテスト（手で描ける5点程度の例で確かめる）（1時間）。
5. `plots.py`（2時間）：
   - **ベンチマーク表**：ランタイム×精度ごとに、p50 / p90 / ピークRSS /（あれば）電力
   - **スレッド数の図**：横軸スレッド数、縦軸p50（エラーバーは周ごとのばらつき）
   - **パレート図**：横軸p50（対数軸にしてもよい）、縦軸成功率（Wilsonの95%区間のエラーバー）。パレート境界を線で結び、マーカーの形＝ランタイム、色＝精度
   - **遅延許容曲線**（27章の結果）：横軸遅延、縦軸成功率、線＝n_exec、帯＝信頼区間
6. 段1〜4を実行する（D4〜D5の夜）。
7. `make_report_tables.py` で `reports/` に出力する（D7）。

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
