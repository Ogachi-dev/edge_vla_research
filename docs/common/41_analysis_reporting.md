# common/41. 図表・Jetsonへの外挿・外部推論との比較・報告書

> **ハンズオン**（共通の進め方は [03_fill_in_guide](../03_fill_in_guide.md)）
> - この章で埋める穴：**4個**（`src/evla/analysis/extrapolate.py` 2、`scripts/netem/setup.sh` 2）
> - ゴール：`pytest tests/analysis/test_extrapolate_rtf.py` が通り、`scripts/make_all_figures.sh` 1本で全図表を作り直せて、報告書が4.3節の構成で書かれている
> - 進め方：まとめの章。穴は少なく、**これまでの結果を1本のスクリプトと1つの報告書にまとめる**作業が中心

## 1. 目的と計画書の対応

- 成果物（図表と報告書）の形を先に決め、データから自動で生成できるようにする。
- i7-1360Pでの結果を、**Jetson Orinに外挿する方法**を決める。
- （任意）外部推論（ネットワーク越しのGPU推論）と比較する実験を、tc netemで行う。
- 計画書の対応箇所：「成果物」7点、日程D9（外部推論との比較、Jetson Orinの公開値との照合）、D10（まとめ）、背景と動機（遅延・ジッタ・通信断への頑健性、台数スケール時のコスト）

## 2. 前提知識・事前に読む資料

- vla.cppのベンチ結果（Jetson AGX Orin / Orin NX / Orin Nano Super、各CPU）：<https://github.com/VinRobotics/vla.cpp/tree/main/docs/benchmark>
- tc netem：`man tc-netem`、<https://wiki.linuxfoundation.org/networking/netem>
- Linuxのネットワーク名前空間（`ip netns`）とveth
- 27章の遅延許容曲線（外挿の土台になる）

### 2.1 外挿に使える公開値（vla.cpp、SmolVLA、2視点・512px、`vla-bench` の最小レイテンシ）

| 端末 | ms | 備考 |
| --- | --- | --- |
| Jetson AGX Orin | 218.3 | CUDA |
| Jetson Orin NX 16GB | 411.2 | CUDA |
| Jetson Orin Nano Super | 462.8 | CUDA |
| Intel Core i7-14700F | 1,689.2 | CPU、65W |
| Intel Core Ultra X7 358H | 1,788.5 | CPU、25W |
| Intel Arc B390 iGPU | 499.8 | OpenVINO |

測定手順（公開ベンチ）：ウォームアップ3回＋計測20回を1プロセスとし、3プロセスのうち平均が最小のものを報告。**【要確認】** 表は「最小値」の列なので、自分の測定と比べるときは同じ統計量（最小値）を使い、p50も併記する。

## 3. 成果物

| パス | 内容（計画書の成果物との対応） |
| --- | --- |
| `reports/benchmark_table.md` | ①ベンチマーク表：モデル×ランタイム×精度×入力条件ごとのレイテンシ、メモリ、電力 |
| `reports/fig_pareto.{png,svg}` | ②パレート図 |
| `reports/fig_latency_tolerance.{png,svg}` | ③遅延許容曲線 |
| `reports/fig_threshold_sensitivity.{png,svg}`、`fig_grasp_ecdf.*` | ④把持閾値の感度分析 |
| `data/lerobot/so101_pickplace_v1/`、`data/models/smolvla_so101_v1/` | ⑤データセットとファインチューニング済みモデル（HFのHubに上げるかは別途判断） |
| `ros2_ws/src/evla_nodes/` | ⑥ROS 2ポリシーノードと把持判定ノード |
| `reports/report.md` | ⑦報告書（Jetsonへの外挿、外部推論との比較を含む） |
| `scripts/make_all_figures.sh` | 全図表を生データから作り直すスクリプト |
| `src/evla/analysis/extrapolate.py` | Jetsonへの外挿 |
| `scripts/netem/*.sh` | 外部推論の模擬実験（任意） |

## 4. インタフェース仕様

### 4.1 Jetsonへの外挿の方法

**【判断】** 2段階で外挿する。

**段階1：レイテンシの換算**（同じエンジン・同じモデル・同じ手順で比べる）

- vla.cppの `vla-bench` で、i7-1360Pを**公開ベンチと同じ手順**で測る（25章の `vlabench_reference.sh`）。
- 換算比 `r = L_Jetson / L_x86_ref` を公開値から作る。ただし、x86の参照は自分の端末そのもの（i7-1360P）なので、`L_Jetson_est = L_Jetson_pub`（公開値をそのまま使う）とし、`L_i7-1360P_measured` との比を報告する。
- 自作のランタイム（OV、PyTorch）での値は、vla.cpp CPUとの比を使って補正する（「OV-GPUはvla.cpp CPUの何倍速いか」という比は、Jetsonには移らないことに注意する）。

**段階2：成功率の推定**

- 27章の遅延許容曲線（遅延→成功率、n_execごと、同期/非同期）に、Jetsonの推定レイテンシを代入して、Jetsonでの成功率を読み取る。
- 前提：Jetsonと i7-1360P で、モデルの数値（行動）が同じとみなせること（同じGGUFを使い、精度も揃える）。

```python
# src/evla/analysis/extrapolate.py
@dataclass(frozen=True)
class DeviceLatency:
    device: str
    latency_ms: float
    source: Literal["measured", "published"]
    note: str = ""

def interpolate_success(curve: pd.DataFrame, latency_ms: float, n_exec: int, mode: str) -> tuple[float, float, float]:
    """遅延許容曲線を線形補間し、(成功率, 下限, 上限) を返す。曲線の範囲外なら外挿せず NaN（範囲外であることを明示）。"""

def extrapolation_table(curve: pd.DataFrame, devices: Sequence[DeviceLatency], n_exec: int, mode: str) -> pd.DataFrame: ...
```

**限界として報告書に書くこと**：CUDAとCPUで数値の誤差の出方が違う。Jetsonの電力モード（15〜60W）やサーマルによって性能が変わる。メモリ帯域の差。カメラ入力（実機）の前処理の時間はベンチに含まれない。

### 4.2 外部推論との比較（D9、任意）

計画書の動機（「帯域だけでは反論に弱い。遅延・ジッタ・通信断への頑健性が主な論点」）を、定量的に示す実験です。

**構成**

```
[ROS / LIBERO 評価プロセス] --veth0-- (netem: 遅延・ジッタ・損失) --veth1-- [ns "remote": 推論サーバー]
```

- 推論サーバーをネットワーク名前空間 `remote` の中で動かし、vethペアの片側に netem をかける。ループバック（lo）に netem をかけると、他の通信もすべて遅くなるので使わない。
- WSL2のカーネルでは netem がモジュール（`CONFIG_NET_SCH_NETEM=m`）なので、`sudo modprobe sch_netem` が必要（**【要確認】** 読み込めない場合は、27章の `LatencyModel` で遅延・ジッタ・損失を模擬する方式に切り替える）。

**比較する条件**

| 条件 | 推論の時間 | 通信 |
| --- | --- | --- |
| オンボード（i7-1360P） | 実測（数秒） | なし |
| 外部GPU（理想） | GPUでの実測（学習機のRTX PRO 6000、またはvla.cppの公開値） | 遅延 0 |
| 外部GPU＋良好な無線 | 同上 | 遅延 20ms ± 5ms |
| 外部GPU＋混雑した無線 | 同上 | 遅延 100ms ± 50ms、損失 1〜5% |
| 外部GPU＋通信断 | 同上 | 一定時間（0.5〜2秒）通信が途絶える |

**仮想時間方式との関係**：上の条件は、27章の `LatencyModel` に「推論の時間＋通信の遅延の分布＋損失（タイムアウト→再送）」として与えれば、netemなしでも同じ実験ができます。**【判断】** 主な結果は仮想時間方式で出し、netemは「実装上の挙動（タイムアウトや再接続を含む）の確認」として1〜2条件だけ行う。

```bash
# scripts/netem/setup.sh（雛形。コマンドの意味を調べてから使う）
sudo ip netns add remote
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth1 netns remote
# TODO(you): IPアドレスの割り当て、link up、netem の設定
# sudo tc qdisc add dev veth0 root netem delay 50ms 10ms distribution normal loss 1%
```

### 4.3 報告書の構成（`reports/report.md`）

1. 要旨（問い、方法、主な数値3つ）
2. 背景と動機（計画書の背景を、結果を踏まえて書き直す）
3. 実験環境（端末、OS、WSL2の制約、ソフトの版、コミット）
4. 方法：トラック1（ランタイム、測定モード、掃引）、トラック2（シーン、把持判定、データ、学習、評価）
5. 結果：ベンチマーク表、パレート図、遅延許容曲線、同期と非同期、Gazeboでの結果、閾値の感度
6. 考察：どの条件なら実用的な制御ができるか（計画書の問いへの答え）、外部推論との比較、Jetsonへの外挿
7. 限界：WSL2でのP/Eコア、simでの把持の代替、衝突の無効化、LIBEROと実機の差、試行数
8. 結論と次の一手
9. 付録：再現手順、設定ファイル、一致の表、品質レポート

## 5. ハンズオン

各 Step は「**埋める／作る → 動かす → 確かめる**」の1サイクルです。報告書の数値は、**すべて生成された表から引用**します（手で転記しない）。

### Step 0：準備（5分）

```bash
grep -n "【穴" src/evla/analysis/extrapolate.py scripts/netem/setup.sh
ls reports/
```

- [ ] 穴4個が表示される
- [ ] 28章・38章・40章で作った表と図が `reports/` にある

### Step 1：図表スクリプトの統合（2時間）

`scripts/make_all_figures.sh`（自分で作る）に、28章の `make_report_tables.py`、38章の `analyze_gazebo_eval.py`、40章の電力の集計を順に並べます。

```bash
bash scripts/make_all_figures.sh
ls -la reports/
```

- [ ] 3節の表の図表（ベンチマーク表、パレート図、遅延許容曲線、閾値の感度、把持精度の ECDF）が全部できる

### Step 2：Jetson への外挿 ― `extrapolate.py`（穴①②、2時間）

**読む**：4.1節の外挿の考え方（「Jetson のレイテンシ（公開値や推定値）を、この端末で測った遅延許容曲線に当てはめる」。曲線の外は外挿しない）。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `n_exec` と `mode` の両方が一致する行。pandas の論理積は `&` で、各条件を括弧で囲む |
| ② | 遅延が曲線の測定範囲（最小〜最大）の内側か。外なら NaN を返して「範囲外」を明示する |

**動かす**

```bash
source .venv-bench/bin/activate
pytest tests/analysis/test_extrapolate_rtf.py -v
```

- [ ] 3つのテストが `PASSED`（区間の内側は線形補間、外側や存在しない `n_exec` は NaN）
- [ ] `extrapolation_table` で、Jetson（Orin Nano / AGX Orin など）の推定レイテンシと推定成功率・区間・前提の表を作り、`make_all_figures.sh` に足す

### Step 3：外部推論の比較（任意、D9、半日） ― `scripts/netem/setup.sh`（穴①②）

まず仮想時間方式（27章の `ConstantLatency`）で4.2節の条件を回し、そのあと netem で1条件だけ実際の通信で確かめます。

**埋める**

| 穴 | ねらい |
| --- | --- |
| ① | `ip link set` で、veth の片側を別のネットワーク名前空間に移す指定 |
| ② | `tc qdisc ... netem` で、遅延とそのゆらぎ（`$DELAY`、`$JITTER`）を与える指定 |

**動かす**

```bash
sudo bash scripts/netem/setup.sh 50ms 10ms 1%
ping -c 5 10.200.0.2
sudo bash scripts/netem/teardown.sh
```

- [ ] `ping` の往復が約 50ms ±ゆらぎになる（netem は veth0 から**出る**向きにだけかかるので、遅延は片道ぶん）
- [ ] 行わなかった場合は、その理由を報告書に書く

### Step 4：報告書（D10）

- [ ] `reports/report.md` を4.3節の構成で書く。数値は `reports/` の表から引用する
- [ ] 計画書の問い（どの条件なら、何Hzで、どの程度の成功率か、パレート境界はどこか）に、1節ずつ答えている

### Step 5：再現性の確認（1時間）

- [ ] 新しいシェルで `bash scripts/make_all_figures.sh` を実行し、同じ図が出る
- [ ] `extrapolate.py` の先頭の `from evla._blank import ___` を消し、コミットする

## 6. ベストプラクティス

- **図はすべてスクリプトで作る**：手で直した図は、データが変わったときに作り直せない。
- **図には不確かさを描く**：成功率には信頼区間、レイテンシには分布（箱ひげ図やp50とp99）。
- **外挿は範囲を明示する**：曲線の外側には外挿しない。前提を必ず書く。
- **「できなかったこと」も成果として書く**：SYCLが動かなかった、P/E固定ができなかった、なども、エッジ推論の実務的な知見として価値がある。
- **数値の出典を追えるようにする**：報告書の各数値の横に、run_id や表のファイル名を（付録でもよいので）書く。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 公開値と自分の値の比較が不公平になる | 統計量（最小値とp50）や手順（ウォームアップの回数）の違い | 公開ベンチと同じ手順で測った値を、比較用に別に用意する |
| netemをかけたら全部遅くなった | loにかけた | vethと名前空間を使う |
| 外挿した成功率に過信が生じる | 曲線の信頼区間を無視した | 区間も一緒に外挿して表に書く |
| 報告書の数値と図が食い違う | 手で転記した | 表と図を同じデータから生成し、報告書は表を参照する |

## 8. ヒント

<details><summary>L1：遅延許容曲線からJetsonの成功率を読むには</summary>

曲線は（遅延、成功率）の点の列。Jetsonの推定遅延がどの2点の間にあるかを見つけて、線形補間する。信頼区間の上限と下限も同じように補間する。
</details>

<details><summary>L2：使う関数・コマンド</summary>

`np.interp(x, xp, fp, left=np.nan, right=np.nan)`、`ip netns exec remote <cmd>`、`tc qdisc show dev veth0`、`tc qdisc del dev veth0 root`（後片付け）、`ping -c 100`（netemの効果の確認）
</details>

<details><summary>L3：extrapolation_table の疑似コード</summary>

```
rows = []
c = curve[(curve.n_exec == n_exec) & (curve.mode == mode)].sort_values("latency_ms")
for d in devices:
    s, lo, hi = (np.interp(d.latency_ms, c.latency_ms, c[col], left=nan, right=nan) for col in ["success", "lo", "hi"])
    rows.append({"device": d.device, "latency_ms": d.latency_ms, "source": d.source, "success_est": s, "lo": lo, "hi": hi})
return DataFrame(rows)
```
</details>

## 9. 完了条件（DoD）

- [ ] 成果物の7点が揃っている（3節の表のとおり）
- [ ] `scripts/make_all_figures.sh` で全図表を作り直せる
- [ ] Jetsonへの外挿の表（推定レイテンシと推定成功率、区間、前提）がある
- [ ] 外部推論の比較を行った、または行わなかった理由が報告書に書かれている
- [ ] 報告書が4.3の構成で書かれ、計画書の問い（どの条件なら、何Hzで、どの程度の成功率か、パレート境界はどこか）に答えている

## 10. 理解度チェック・発展課題

- 帯域（カメラ2台・640×480・30fpsで約440Mbps）を動機にすると反論に弱い、と計画書が述べている理由は何か。この研究の結果は、どの論点で「オンボード推論」を支持（または否定）したか。
- 台数をN台に増やしたときの、オンボード推論と外部推論のコストを比べる簡単なモデルを作れ（ボードの単価、GPUサーバーの単価と1台あたりの同時処理数）。
- 発展：Jetson Orinを実際に借りられたら、何を最初に測るか。この研究の基盤のうち、そのまま使えるものと、作り直しが必要なものを分けよ。
