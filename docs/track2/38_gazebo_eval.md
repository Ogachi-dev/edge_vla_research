# track2/38. Gazebo閉ループ評価と、把持閾値の感度分析

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**6個**（`ros2_ws/src/evla_nodes/evla_nodes/eval_runner.py` 2、`src/evla/analysis/rejudge.py` 4）
> - ゴール：基準（PyTorch FP32）とトラック1の最適設定の Gazebo での成功率（Wilson 区間つき）が出て、閾値の感度曲線と把持精度の ECDF の図がある
> - 進め方：Step 1 で評価のループを作り、Step 3〜4 を夜に回す。Step 5 の解析の穴（`rejudge.py`）は pytest で先に埋めておける

## 1. 目的と計画書の対応

- ファインチューニング済みSmolVLAを、トラック1で選んだ設定でGazebo上で動かし、成功率・把持精度・実効制御周期を測る。
- 把持判定の閾値を変えて成功率を算出し、**量子化モデルとFP32モデルの差がどの精度帯で開くか**を示す。
- 計画書の対応箇所：「把持判定仕様（閾値の決め方）」「閾値の感度分析」、評価指標（タスク成功率、把持精度、実効制御周期、RTF）、日程D8、成果物「把持閾値の感度分析」

## 2. 前提知識・事前に読む資料

- 32章（把持判定、`grasp_candidate` のログ）、27章（統計：Wilson、McNemar）、37章（ポリシーノード）
- 経験累積分布関数（ECDF）と、2つの分布の比較

## 3. 成果物

| パス | 役割 |
| --- | --- |
| `evla_nodes/eval_runner.py` | 評価のエピソード管理（リセット、タイムアウト、ロックステップ時の推論ループ） |
| （配置の集合） | `eval_runner` がシード 900000＋i から配置を作る（学習に使っていないシード。全設定で同じ配置になる） |
| `configs/policy/so101_*.yaml` | 評価する設定（モデル、ランタイム、精度）。36章 Step 5 で作ったもの |
| `src/evla/analysis/rejudge.py` | ログからの閾値の再判定 |
| `scripts/analyze_gazebo_eval.py`（自分で作る） | 成功率、失敗の分類、把持のずれの分布、感度曲線（`rejudge.sensitivity_curve` と `plots.grasp_ecdf_plot` を呼ぶ） |
| `reports/gazebo_eval.md`、図 | 結果 |

## 4. インタフェース仕様

### 4.1 評価の設計

| 項目 | 値（【判断】） |
| --- | --- |
| 配置 | 学習と重ならないシードで50配置を固定し、全設定で同じものを使う（対応のある比較） |
| 1エピソードの上限 | sim時刻で30秒（デモの最長の約2倍） |
| モード | ロックステップ（推論中はworldを止める）＝遅延と切り離した精度の比較。実時間＝遅延込みの比較 |
| 設定 | 基準：PyTorch FP32、トラック1の最適設定、精度違い（INT8 / Q8_0 / Q4_0 など）を2〜4個 |
| 試行 | 1設定×1モードあたり50エピソード |

### 4.2 失敗の分類

| 分類 | 判定方法 |
| --- | --- |
| `no_attempt` | `grasp_candidate` が1回も出なかった（閉じなかった） |
| `grasp_rejected:<条件>` | `grasp_candidate` は出たが、どれも条件を満たさなかった。最も多く落ちた条件を記録する |
| `dropped_or_misplaced` | attachしたが、箱に入らなかった |
| `timeout_after_release` | 解放後、静止の判定が出なかった |

### 4.3 閾値の再判定（感度分析）

```python
# src/evla/analysis/rejudge.py
@dataclass(frozen=True)
class ThresholdScale:
    position: float = 1.0      # 上下・前後の許容値に掛ける
    yaw: float = 1.0
    approach: float = 1.0
    # 開閉方向は (開き幅 − キューブ幅)/2 なので幾何で決まる。比較のため、これにも掛け算を入れてもよい

def rejudge_episode(events: Sequence[dict], base: GraspThresholds, scale: ThresholdScale) -> bool | None:
    """オンラインで成功したエピソードについて、実際に attach した grasp_candidate の幾何量が、
    より厳しい閾値（scale ≤ 1）でも条件を満たすかを判定する。満たせば True、満たさなければ False。
    scale > 1（より緩い閾値）は判定できないので None を返す。"""

def sensitivity_curve(episodes: pd.DataFrame, scales: Sequence[float]) -> pd.DataFrame:
    """設定ごと・scale ごとの成功率（Wilson区間つき）。"""
```

**再判定の限界（重要）**

- オンラインの閾値より**厳しい**閾値は、ログから判定できます。オンラインで掴んだときの幾何量が、厳しい閾値を満たすかを見ればよいからです。
- ただし、厳しい閾値なら「掴めずに、方策がやり直して後で成功した」可能性もあります。したがって、再判定した成功率は**下限（悲観的な推定）**です。
- オンラインより**緩い**閾値は判定できません（実際には掴まなかったので、その後どうなったかが分からない）。

→ **【判断】** 評価は、感度分析したい範囲の**最も緩い閾値でオンラインに実行し**、それより厳しい閾値はログから再判定する。データ生成に使った標準の閾値（scale=1.0）でも、別途オンラインで実行して、再判定の値と比べ、下限推定の偏りの大きさを確かめる。

### 4.4 把持精度の比較

成功・失敗に関わらず、`attached` の時点の幾何量（軸ごとのずれ[mm]、ヨー[°]）の分布を設定ごとに比べます。

- 図：軸ごとの |ずれ| のECDF（設定ごとに線を重ねる）。
- この図があると、「量子化で、平均的に何mmずれが大きくなったか」を、閾値の選び方に依存せずに示せる。成功率の感度曲線の説明になる。

## 5. ハンズオン

各 Step は「**読む → 埋める → 動かす → 確かめる**」の1サイクルです。この章は、これまでの部品（31〜37章）を全部つないで**数字を出す**章です。実行に時間がかかるので、**小さく試してから夜に回す**を守ってください。

### Step 0：準備（5分）

```bash
grep -n "【穴" ros2_ws/src/evla_nodes/evla_nodes/eval_runner.py src/evla/analysis/rejudge.py
```

- [ ] 穴6個が表示される
- [ ] 37章の Step 2（サーバーとクライアント）と Step 5（疎通確認）が済んでいる

### Step 1：評価のループ ― `eval_runner.py`（穴①②、3時間）

**読む**：ファイル冒頭の docstring。ロックステップでは**このノードが時計の所有者**で、「観測を集める → 推論サーバーに問い合わせる → 軌道を送る → n_exec フレーム step する」を繰り返す。実時間では world を動かし、制御は37章のポリシーノードに任せる。リセットは34章の `SimIO.reset_scene` を共有していることも確認する。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | 37章の `InferenceClient.predict` を、今の観測（画像・状態・指示文・sim 時刻）で呼ぶ。world は止まったままなので、推論時間は結果に影響しない |
| ② | 実時間係数（RTF）＝ sim 時刻の経過 ÷ 壁時計の経過。1 未満なら、シミュレーションが現実より遅い |

**動かす**：まず3エピソードだけ回します。

```bash
# 端末A（.venv-bench）
python scripts/serve_policy.py --config configs/policy/torch_fp32.yaml --bind tcp://127.0.0.1:5600 \
  --image_key_map '{"top": "top", "wrist": "wrist"}'     # SO-101 のモデルの設定に替える
# 端末B（ROS）
ros2 launch evla_sim sim.launch.py world:=/tmp/evla_world_0.sdf
# 端末C（ROS）
ros2 run evla_nodes eval_runner --ros-args -p use_sim_time:=true -p mode:=lockstep \
  -p config_name:=smoke -p n_episodes:=3
```

**確かめる**

- [ ] `[1/3] ok (1/1)` のようなログが3行出る（`ok` でなくても、`grasp_rejected:forward` などの分類が出ればループは動いている）
- [ ] `results/runs/<run_id>/episodes.jsonl` に3行あり、各行に `events`（把持判定のイベント）が入っている
- [ ] 1エピソードの所要時間から、50エピソード×設定数の実行時間を見積もる（Step 3〜4 の計画に使う）

### Step 2：評価の配置を確かめる（30分）

`eval_runner` は、シード `900000 + i`（学習のデータ生成と重ならない）から配置を作ります。同じ `i` なら全設定で同じ配置になるので、**対応のある比較**（McNemar）ができます。

```bash
python -c "
import numpy as np, yaml
from evla.planning.randomize import sample_scene
r = yaml.safe_load(open('configs/datagen/default.yaml'))
for i in range(3): print(sample_scene(np.random.default_rng(900000 + i), r).cube_xy_m)
"
```

- [ ] 何度実行しても同じ3点が出る
- [ ] 50配置のキューブ位置の散布図を作り、学習データ（34章）の範囲と同じ範囲に散らばっていることを確かめる

### Step 3：基準の評価（2時間＋実行時間）

- [ ] PyTorch FP32、ロックステップ、50エピソード（`-p config_name:=torch_fp32 -p n_episodes:=50`）。Step 1 の見積もりで、その日のうちに終わるか確かめてから回す

### Step 4：各設定の評価（D8、夜間）

- [ ] 4.1節の表の設定×モードを、夜のうちに順に回す（サーバーの `--config` と eval_runner の `config_name` を揃える）
- [ ] トラック1の最適設定は、**実時間**（`-p mode:=realtime`。37章のポリシーノードを同時に起動）でも回し、RTF と実効制御周期を記録する

### Step 5：解析 ― `rejudge.py`（穴①〜④、2時間）

**読む**：4.3節の「閾値の再判定」。オンラインより**厳しい**閾値だけが判定でき、結果は**下限**（悲観的な推定）になる理由を説明できるようにする。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `frozen=True` の dataclass は書き換えられない。一部のフィールドだけ変えたコピーを作る `dataclasses` の関数 |
| ② | どれか1つでも倍率が 1 を超えていれば「緩める方向」＝判定できない（None を返す） |
| ③ | 厳しくした閾値で、32章の `check_grasp` を同じ幾何量に通した結果 |
| ④ | pandas で設定ごとに行をまとめる |

**動かす**

```bash
source .venv-bench/bin/activate
pytest tests/analysis/test_rejudge.py -v
```

- [ ] 4つのテストが `PASSED`（scale=1 でオンラインと一致、厳しくすると失敗しうる、緩めると None、曲線が単調）

**解析のスクリプトを作る**：`scripts/analyze_gazebo_eval.py`（自分で作る）で、`results/runs/*_t2_gz-*/episodes.jsonl` を読み、次を `reports/gazebo_eval.md` と図に出します。

- [ ] 成功率（`wilson_ci`）と、基準との McNemar（27章の `stats.py`）
- [ ] 失敗の分類（`reason` の集計）の表
- [ ] 感度曲線（`sensitivity_curve`）と、把持精度の ECDF（28章の `plots.grasp_ecdf_plot`）
- [ ] scale=1.0 での再判定とオンラインの結果を比べ、下限推定の偏りを記録する

### Step 6：仕上げ（10分）

- [ ] 2ファイルの先頭の `from evla._blank import ___` を消し、`ruff check src/evla/analysis/rejudge.py` を通す
- [ ] コミットする

## 6. ベストプラクティス

- **同じ配置の集合で比較する**：設定ごとに配置を変えると、配置の難しさの違いが結果に混ざる。
- **失敗の動画を数本ずつ見る**：数値だけで分類すると、判定のバグ（成功しているのに失敗扱い）を見逃す。
- **評価の閾値はデータ生成と同じファイルから読む**：scale はその上に掛ける。閾値の定義が2か所にあると、必ずずれる。
- **RTFを一緒に記録する**：実時間のモードでは、RTFが1から外れると、推論の遅延の「sim時刻での大きさ」が変わる。RTFが低いと、実機よりも遅延が小さく見積もられる。
- **予算が足りないときの優先順位**：(1) 基準とトラック1の最適設定のロックステップ、(2) 最適設定の実時間、(3) 精度違い、(4) 感度曲線の点の追加。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 基準（FP32）の成功率が極端に低い | 学習の問題、前処理・関節の対応のずれ、タスク文の違い | 37章の対応表、35章のオフライン誤差を確認する。学習データの配置で評価して、学習自体ができているか切り分ける |
| 実時間のモードで成功率が大きく揺れる | RTFの揺らぎ、CPUの取り合い（Gazeboの描画と推論） | 計画書のリスク表のとおり、iGPU推論はロックステップを基本にする。RTFを記録する |
| 再判定の成功率がオンラインより高い | 再判定のロジックの誤り（下限のはず） | scale=1.0 で再判定して、オンラインの結果と一致するか確かめる |
| 量子化モデルでも差が出ない | 試行数不足、または本当に差がない | 信頼区間で判断する。把持精度のECDFで、細かい差を見る |

## 8. ヒント

<details><summary>L1：再判定に必要な情報は何か</summary>

各エピソードで「実際にattachした瞬間の幾何量」と「その後に成功したか」。32章で `grasp_candidate` と `attached` のイベントを、判定結果によらず全部残しておいたのはこのため。
</details>

<details><summary>L2：使う関数</summary>

`pandas.read_json(path, lines=True)`、`df.groupby(["config", "scale"])`、32章の `check_grasp`（閾値だけ差し替えて再利用する）、`dataclasses.replace(th, vertical_tol_m=th.vertical_tol_m * s, ...)`、ECDF：`np.sort(x)` と `np.arange(1, n + 1) / n`
</details>

<details><summary>L3：rejudge_episode の疑似コード</summary>

```
if any(v > 1.0 for v in asdict(scale).values()): return None
attached = [e for e in events if e["type"] == "attached"]
succ = any(e["type"] == "success" for e in events)
if not attached or not succ: return False
g = GraspGeometry(**attached[0]["geometry"])
th = scaled(base, scale)
return check_grasp(g, th).ok
```
</details>

## 9. 完了条件（DoD）

- [ ] 基準（PyTorch FP32）とトラック1の最適設定の、ロックステップでの成功率（Wilson区間）がある
- [ ] 最適設定の実時間での成功率、実効制御周期、RTFがある
- [ ] 精度違いの設定（2つ以上）との比較があり、McNemarの結果が書かれている
- [ ] 失敗の分類の表がある
- [ ] 感度曲線（scale→成功率、設定ごと）と、把持精度のECDFの図がある
- [ ] scale=1.0 での再判定とオンラインの結果を比べ、下限推定の偏りを記録した

## 10. 理解度チェック・発展課題

- 閾値を厳しくするほど、FP32と量子化モデルの成功率の差が開くとしたら、それは何を意味するか。逆に差が閉じるとしたら。
- 再判定が「下限」になる理由を、自分の言葉で説明せよ。
- 発展：閾値を変える代わりに、キューブの大きさや爪の開き幅を変えたら、同じ分析ができるか。どちらが実機への移行に意味があるか。
