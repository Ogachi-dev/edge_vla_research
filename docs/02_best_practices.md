# 02. ベストプラクティス：実装・テスト・計測の作法

> **この章の位置づけ**：穴埋めはありません。全章のハンズオンに共通する作業サイクル（5節）と、計測の作法（6節）を最初に一度読んでおきます。

## 1. 目的と計画書の対応

- 全章に共通するコーディング規約と作法をまとめる。各章の「ベストプラクティス」節は、この章を前提にしている。
- 計画書の対応箇所：「測定条件の統一」「リスクと対策」

この研究では、**計測の信頼性**が成果そのものです。速いが不正確なコードより、遅くても正しく測れるコードを優先します。

## 2. 前提知識・事前に読む資料

- Python型ヒント：<https://docs.python.org/3/library/typing.html>
- pytest入門：<https://docs.pytest.org/en/stable/getting-started.html>
- rclpyの実行モデル（Executor・Callback Group）：<https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Executors.html>
- QoS：<https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html>
- ROSの時刻とsim time：<https://design.ros2.org/articles/clock_and_time.html>

## 3. 成果物

この章には、固有の成果物はありません。`ruff` と `pytest` の設定は [01](01_repository_design.md) の `pyproject.toml` に含めています。

## 4. インタフェース仕様（コード規約）

### 4.1 Python

| 規約 | 内容 | 理由 |
| --- | --- | --- |
| 型ヒント | 公開関数にはすべて型ヒントを付ける。配列は `npt.NDArray[np.float32]` | IDEの補完と `mypy` で、形や型の取り違えを早期に検出できる |
| 配列の形 | docstringに形を書く（例：`(T, 6) float32, 単位: deg`） | 形と単位の取り違えは、ロボット系で最も多いバグ |
| 単位 | 変数名に単位を付ける（`latency_ms`、`q_rad`、`q_deg`、`pos_m`） | rad/degやm/mmの混在を防ぐ |
| データ構造 | 値の組は `@dataclass(frozen=True)` にする。dictの多用は避ける | キーのタイポを防ぎ、不変にして副作用を防ぐ |
| 例外 | 想定外の状態は握りつぶさず、例外にする。`except Exception: pass` は禁止 | 測定が黙って壊れることを防ぐ |
| ログ | `print` ではなく `logging` を使う。ROSノードでは `self.get_logger()` | レベル分けとファイル出力ができる |
| 乱数 | `np.random.default_rng(seed)` を引数で受け取る。グローバルな `np.random.seed` は使わない | 再現性と並列実行時の独立性を保つ |
| 時刻 | 区間の計測には `time.perf_counter_ns()` を使う。`time.time()` は使わない | `time.time()` はNTPの補正で戻ることがあり、分解能も低い |

### 4.2 ROS 2（rclpy）

| 規約 | 内容 | 理由 |
| --- | --- | --- |
| sim time | Gazeboと連動するノードは、すべて `use_sim_time:=true` にする | 壁時計とsim時刻が混ざると、TFの補間やタイムアウトが壊れる |
| QoS | カメラ画像は `SensorDataQoS`（BEST_EFFORT・depth小）。イベント（attach/detach）は RELIABLE | 画像は最新の1枚があればよい。イベントは取りこぼすと致命的 |
| パラメータ | 閾値などは `declare_parameter` で宣言し、YAMLから渡す | コードを変えずに感度分析ができる |
| コールバック | コールバックの中で重い処理やブロッキング呼び出し（サービスの同期呼び出しなど）をしない | シングルスレッドのExecutorでは、デッドロックや処理落ちの原因になる |
| サービス呼び出し | `call_async` ＋ Future を使う。待つ必要があるなら、別のCallback Groupと `MultiThreadedExecutor` を使う | コールバック内で `call()` するとデッドロックする（典型的な落とし穴） |
| ロジックの分離 | ノードクラスにはI/Oだけを書き、判定や計算は `evla` の純粋関数を呼ぶ | ROSなしでテストできる |

### 4.3 テスト

- **純粋関数は境界値でテストする**：閾値ちょうど、閾値±ε、角度の周期の境界（±45°、±90°）など。
- **既知の答えでテストする**：統計関数は、手計算できる小さな入力（例：`[1,2,3,4]` のp50）で確かめる。
- **プロパティテスト**：「ヨー角に90°を足しても判定は変わらない」のような不変条件は、ランダムな入力で多数回確かめる。（`hypothesis` を入れてもよいし、`rng` のループでもよい）
- **計測コードもテストする**：`sleep(0.01)` を計測して約10msになるか、などの健全性チェックを入れる。

## 5. ハンズオンの作業サイクル（毎日の進め方）

各章の5節（ハンズオン）は、すべて同じ型で書いてあります。1つの Step を次の順で回します。

1. **Step 0 で出発点を確かめる**：`grep -n "【穴" <ファイル>` で穴の一覧を出し、その章のテストが `NotImplementedError` で**失敗する**ことを見ておく（テストが正しく穴を検出している、という確認になる）。
2. **読む**：Step の「読む」に挙げた節（インタフェース仕様、docstring、テストのコード）を読む。テストは「何ができれば正解か」の仕様書として読む。
3. **埋める**：表の「ねらい」を読んでから、コードのヒントを見て穴を埋める。詰まったら、章末のヒントを L1 から開く。10分以上止まったら `answers/` の該当行だけを見る（[03](03_fill_in_guide.md) の2節）。
4. **動かす**：Step の「動かす」のコマンドを、そのまま実行する。
5. **確かめる**：「確かめる」のチェック項目と、実際の出力を1つずつ見比べる。合わなければ、次の Step に進まない。
6. **仕上げる**：章の最後の Step で、`from evla._blank import ___` を消し、`ruff check` と章のテストを通して、1つの機能ごとにコミットする。メッセージは「何を・なぜ」。
7. **メモを残す**：ハマった点と解決方法を、`docs/notes/YYYY-MM-DD.md`（自分で作る）に3行でよいので書く。報告書を書くときの材料になる。

**AIエージェントとの付き合い方（学習目的の場合）**：エージェントに頼む場合も、「この関数のレビュー」「このエラーの意味」「このAPIの使い方」のように、**自分のコードに対する質問**の形にする。実装の丸投げはしない。自分で書いたコードを説明できることが、このプロジェクトのゴールの1つです。

## 6. ベストプラクティス：計測の作法

計測にはこの研究の成果がかかっているので、ルールを明文化しておきます。

1. **ウォームアップを捨てる**：最初の数回は、JITコンパイルやキャッシュ、メモリ確保のために遅い。OpenVINO GPUは初回コンパイルに数十秒〜数分かかることもある。
2. **分布で報告する**：平均値だけでなく、p50・p90・p99・最小・最大を出す。制御で効くのはp99（最悪側）。
3. **1回の測定で結論を出さない**：同じ設定をプロセスを分けて3回測り、ばらつきを確認する（vla.cppの公開ベンチも「3プロセス×20回」で測っている）。
4. **環境を固定・記録する**：電源プラン、スレッド数、他に動いているプロセス、温度。すべて `env.json` に残す。
5. **測定対象以外を止める**：ブラウザ、IDEのインデックス作成、Windows Update。少なくともログに記録する。
6. **条件を交互に測る**：A→A→A→B→B→B ではなく A→B→A→B と測り、時間とともに進む温度上昇の影響を均す。
7. **成功率には信頼区間を付ける**：50試行で70%なら、95%信頼区間は約±13ポイントある。差を主張する前に区間が重ならないか確認する（[27](track1/27_closed_loop_b_c.md)）。

## 7. 落とし穴

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| 測るたびに値が大きく変わる | サーマルスロットリング、ターボブーストの持続時間切れ | 測定間に冷却時間を入れる。温度とクロックを記録する（[40](common/40_metrics_measurement.md)） |
| スレッドを増やすと遅くなる | ライブラリが二重にスレッドを立て、論理コア数を超えた（over-subscription） | `OMP_NUM_THREADS` などを明示し、実行中のスレッド数を `ps -L` で確認する |
| ROSノードが固まる | コールバック内でサービスを同期呼び出しした | `call_async` を使う。Callback Groupを分ける |
| sim時刻が0のまま | `/clock` がbridgeされていない、または `use_sim_time` が未設定 | `ros2 topic echo /clock` で確認する |
| テストは通るが実機（sim）で動かない | 単位、座標系、配列の形の取り違え | docstringに単位と形を書き、境界で `assert arr.shape == (...)` する |

## 8. ヒント

<details><summary>L1：コールバック内でサービス結果を待ちたい</summary>

「待つ」設計そのものを避けられないか考える。状態機械にして、サービスの完了コールバックで次の状態に進めば、待つ必要はなくなる。
</details>

<details><summary>L2：使うAPI</summary>

`client.call_async(req)` → `future.add_done_callback(cb)`。どうしても待つ場合は `MutuallyExclusiveCallbackGroup` を2つ作ってタイマーとクライアントを分け、`MultiThreadedExecutor` で `spin` する。
</details>

<details><summary>L3：状態機械の疑似コード</summary>

```
state = IDLE
on_timer():
    if state == IDLE and 条件成立: future = cli.call_async(req); future.add_done_callback(on_done); state = WAITING
on_done(fut):
    if fut.result().success: state = DONE else: state = IDLE
```
</details>

## 9. 完了条件（DoD）

- [ ] `uv run ruff check src tests` がエラーなしで通る
- [ ] `uv run pytest` が通る
- [ ] この章の「計測の作法」7項目を、自分のベンチスクリプトがどう満たしているか説明できる

## 10. 理解度チェック・発展課題

- `time.time()` と `time.perf_counter_ns()` と `time.monotonic()` の違いを説明できるか。
- rclpyのシングルスレッドExecutorで、コールバックの中から `client.call()` するとなぜデッドロックするか。
- p50が同じ2つの設定のうち、ロボット制御に向くのはどちらか。どの指標で判断するか。
