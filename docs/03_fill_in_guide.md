# 03. 穴埋め版の進め方

各章の「インタフェース仕様」から自分でコードを書き起こすと、コードを考えるだけで時間がなくなります。
そこで、**動くコードを先に用意し、学習の要になる行だけを空欄（穴）にしました**。
穴を埋めてテストを通しながら、各章の本文で「なぜそう書くのか」を確認する、という進め方にします。

- 穴の数：**281 個**（79 ファイル）。1つの穴は、関数名・引数・1つの式のどれかです。
- 答え：`answers/` に、穴を埋めた完成版が同じパスで置いてあります（git の管理外）。
- 章ごとの本文（インタフェース仕様、ベストプラクティス、落とし穴）は、そのまま「穴を埋めるための資料」として使います。

---

## 1. 穴の見た目

```python
digest = ___(text.encode("utf-8")).hexdigest()  # 【穴②】ハッシュ関数（hashlib モジュール、160bit）
```

- `___` の部分を、コメントのヒントに合う**式**に置き換えます。行のほかの部分は変えなくてよいです。
- キーワード引数は値だけが穴になっています（例：`d.mkdir(parents=True, exist_ok=___)`）。
- YAML・XML・シェルにも同じ形の穴があります（例：`update_rate: ___  # 【穴①】...`）。
- Python のファイルの先頭には次の行があります。`___` を「呼ぶと例外を出す特別な値」として定義しているので、**穴が残っていてもファイル自体は import でき**、穴の行に来たところで場所が分かるエラーになります。

```python
from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい
```

```text
NotImplementedError: 穴埋め（___）が残っています。トレースバックの行を埋めてください
```

**便利なコマンド**

```bash
grep -rn "【穴" src scripts configs ros2_ws/src/evla_*      # 残っている穴の一覧
grep -rn "【穴" src/evla/grasp/                            # 1つのパッケージだけ
ruff check src/evla/common/run_id.py                       # F401（未使用の import）が出ていたら、その import が穴の答えのヒント
```

## 2. 1つの穴の埋め方（手順）

1. 対応する章の「4. インタフェース仕様」と、ファイルの docstring を読み、その関数が**何を返すべきか**を確認する。
2. 穴のヒントを読む。API 名が分からなければ、公式ドキュメントか `python -c "import X; help(X.Y)"` で調べる。
3. 埋めたら、その章のテストを実行する（4節の表）。テストのないもの（ROS・モデルが必要なもの）は、章の「9. 完了条件」で確かめる。
4. 埋め終わったファイルの先頭の `from evla._blank import ___` を消し、`ruff check` を通す。

**答え合わせのルール（おすすめ）**：同じ穴で10分以上止まったら、章のヒント（L1→L2→L3）を開く。それでも分からなければ、`answers/` の該当行だけを見る。

```bash
diff answers/src/evla/common/run_id.py src/evla/common/run_id.py    # 自分の答えとの差分（埋め終わってから）
```

## 3. 環境ごとに、どこまで確認できるか

| 対象 | 確認できる環境 | 方法 |
| --- | --- | --- |
| `src/evla/` のうち、ROS・モデル不要の部分（common、grasp、planning、dataset、ipc、sim の executor、analysis、sweep） | `.venv-bench`（D1 から） | `pytest`（4節の表） |
| モデルが必要な部分（torch / ov / vlacpp の Policy、LIBERO） | `.venv-bench`＋モデル | 章の完了条件（スモークテスト、一致の表） |
| ROS ノード（`ros2_ws/src/evla_nodes`）と Gazebo の設定（`evla_sim`） | Ubuntu 24.04＋ROS 2 Jazzy | `colcon build` → launch → 章の完了条件 |

ROS ノードは `evla`（`src/evla`）を import します。ROS 側からは `source scripts/ros_env.sh` で `PYTHONPATH` に `src/` を通してください（Ubuntu 24.04 のシステム Python には `pip install` できないため）。

## 4. 進める順番と、各章の穴

日程（[00_schedule.md](00_schedule.md)）の順に並べています。同じ行の中では、上のファイルから順に埋めてください（下のファイルが上のファイルを使います）。テストは `.venv-bench` を有効にして、リポジトリの直下で実行します。

| 順 | 章 | 内容 | 穴埋めファイル（穴の数） | テスト |
| --- | --- | --- | --- | --- |
| 1 | [01](01_repository_design.md) | 共通部品（run_id・設定・環境記録） | `src/evla/common/run_id.py`（5）<br>`src/evla/common/config.py`（6）<br>`src/evla/common/sysinfo.py`（7）<br>`src/evla/common/cli.py`（1） | `pytest tests/common/test_run_id.py tests/common/test_config.py tests/common/test_sysinfo.py` |
| 2 | [12](env/12_inference_envs.md) | vla.cpp のプロトコル生成 | `scripts/gen_vla_proto.sh`（1） | なし（章の完了条件で確認） |
| 3 | [21](track1/21_libero_setup.md) | LIBERO のラッパと固定観測 | `src/evla/sim/libero_env.py`（9）<br>`src/evla/sim/fixed_obs.py`（2）<br>`scripts/libero_smoke.py`（2）<br>`scripts/dump_fixed_obs.py`（1） | なし（章の完了条件で確認） |
| 4 | [22](track1/22_policy_interface.md) | Policy 抽象・前処理・一致指標 | `src/evla/policy/factory.py`（1）<br>`src/evla/policy/preprocess.py`（5）<br>`src/evla/policy/equivalence.py`（6）<br>`scripts/check_equivalence.py`（2） | `pytest tests/policy` |
| 5 | [23](track1/23_runtime_pytorch.md) | PyTorch 版 | `src/evla/policy/torch_policy.py`（7） | なし（章の完了条件で確認） |
| 6 | [24](track1/24_runtime_openvino.md) | OpenVINO 版 | `src/evla/policy/ov_policy.py`（7）<br>`scripts/ov_download.py`（1）<br>`scripts/ov_inspect.py`（2） | なし（章の完了条件で確認） |
| 7 | [25](track1/25_runtime_vlacpp.md) | vla.cpp 版 | `src/evla/policy/vlacpp_server.py`（7）<br>`src/evla/policy/vlacpp_policy.py`（11）<br>`scripts/vlacpp_quantize.sh`（1）<br>`scripts/vlabench_reference.sh`（1） | なし（章の完了条件で確認） |
| 8 | [26](track1/26_bench_mode_a.md) | モードA（単体ベンチ） | `src/evla/bench/timing.py`（6）<br>`src/evla/bench/memory.py`（4）<br>`src/evla/bench/mode_a.py`（5）<br>`scripts/bench_mode_a.py`（1） | `pytest tests/bench` |
| 9 | [27](track1/27_closed_loop_b_c.md) | モードB/C（executor・遅延・統計） | `src/evla/sim/latency.py`（2）<br>`src/evla/sim/executor.py`（7）<br>`src/evla/sim/closed_loop.py`（2）<br>`src/evla/analysis/stats.py`（7）<br>`scripts/eval_closed_loop.py`（1） | `pytest tests/sim tests/analysis/test_stats.py` |
| 10 | [28](track1/28_sweep_and_pareto.md) | 掃引とパレート図 | `src/evla/sweep/spec.py`（4）<br>`src/evla/sweep/runner.py`（3）<br>`src/evla/analysis/load.py`（1）<br>`src/evla/analysis/pareto.py`（2）<br>`src/evla/analysis/plots.py`（3）<br>`scripts/sweep.py`（1）<br>`scripts/make_report_tables.py`（1） | `pytest tests/sweep tests/analysis/test_pareto.py` |
| 11 | [31](track2/31_gazebo_scene.md) | シーン・URDF・bridge・launch | `ros2_ws/src/evla_sim/CMakeLists.txt`（1）<br>`ros2_ws/src/evla_sim/config/bridge.yaml`（3）<br>`ros2_ws/src/evla_sim/config/controllers.yaml`（1）<br>`ros2_ws/src/evla_sim/urdf/so101_evla.urdf.xacro`（4）<br>`ros2_ws/src/evla_sim/worlds/evla_pickplace.sdf.jinja`（1）<br>`ros2_ws/src/evla_sim/scripts/render_world.py`（3）<br>`ros2_ws/src/evla_sim/launch/sim.launch.py`（3） | なし（章の完了条件で確認） |
| 12 | [32](track2/32_grasp_judge_node.md) | 把持判定（純粋関数＋ノード） | `src/evla/grasp/geometry.py`（11）<br>`src/evla/grasp/judge.py`（11）<br>`src/evla/grasp/events.py`（3）<br>`ros2_ws/src/evla_nodes/evla_nodes/ros_utils.py`（2）<br>`ros2_ws/src/evla_nodes/evla_nodes/grasp_judge_node.py`（5）<br>`ros2_ws/src/evla_nodes/setup.py`（1） | `pytest tests/grasp` |
| 13 | [33](track2/33_moveit_demo_gen.md) | IK・ランダム化・区間の計画 | `src/evla/planning/ik_so101.py`（7）<br>`src/evla/planning/randomize.py`（5）<br>`src/evla/planning/episode_plan.py`（1）<br>`src/evla/planning/build.py`（3）<br>`ros2_ws/src/evla_nodes/evla_nodes/demo_generator.py`（2） | `pytest tests/planning` |
| 14 | [34](track2/34_episode_recorder.md) | ロックステップ記録とデータセット | `src/evla/common/stamped_buffer.py`（3）<br>`src/evla/dataset/conventions.py`（6）<br>`src/evla/dataset/raw_episode.py`（6）<br>`src/evla/dataset/quality.py`（5）<br>`src/evla/dataset/to_lerobot.py`（4）<br>`ros2_ws/src/evla_nodes/evla_nodes/lockstep.py`（4）<br>`ros2_ws/src/evla_nodes/evla_nodes/sim_io.py`（3）<br>`ros2_ws/src/evla_nodes/evla_nodes/episode_runner.py`（5） | `pytest tests/common/test_stamped_buffer.py tests/dataset` |
| 15 | [36](track2/36_model_conversion.md) | OpenVINO IR / GGUF への変換 | `scripts/export_ov.py`（3）<br>`scripts/export_gguf.sh`（1） | なし（章の完了条件で確認） |
| 16 | [37](track2/37_policy_node_zmq.md) | ZeroMQ 通信とポリシーノード | `src/evla/ipc/protocol.py`（6）<br>`src/evla/ipc/client.py`（4）<br>`src/evla/ipc/server.py`（4）<br>`scripts/serve_policy.py`（1）<br>`ros2_ws/src/evla_nodes/evla_nodes/action_mapping.py`（3）<br>`ros2_ws/src/evla_nodes/evla_nodes/policy_node.py`（3） | `pytest tests/ipc` |
| 17 | [38](track2/38_gazebo_eval.md) | Gazebo 評価と閾値の再判定 | `ros2_ws/src/evla_nodes/evla_nodes/eval_runner.py`（2）<br>`src/evla/analysis/rejudge.py`（4） | `pytest tests/analysis/test_rejudge.py` |
| 18 | [40](common/40_metrics_measurement.md) | 電力・RTF | `src/evla/analysis/power.py`（4）<br>`src/evla/analysis/rtf.py`（2）<br>`scripts/attach_power.py`（1）<br>`scripts/clock_check.sh`（1） | `pytest tests/analysis/test_power.py tests/analysis/test_extrapolate_rtf.py` |
| 19 | [41](common/41_analysis_reporting.md) | Jetson への外挿・netem | `src/evla/analysis/extrapolate.py`（2）<br>`scripts/netem/setup.sh`（2） | `pytest tests/analysis/test_extrapolate_rtf.py` |

**テストどうしの依存**：たとえば `tests/grasp/test_judge.py` は `geometry.py` の穴も使います。テストが `NotImplementedError` で落ちたら、トレースバックの**一番下の行のファイル**が、まだ埋めていない場所です。

**穴のないファイル**：`src/evla/_blank.py`、`policy/base.py`、`sim/types.py`、`planning/episode_plan.py` の一部、`tests/` の全部、`configs/` の大半は、穴なしの完成版です。読んで理解するだけでよいです（特に `tests/` は「何ができれば正解か」の仕様書として読む）。

## 5. 穴以外に、自分で決めて書くところ（`TODO(you)`）

穴埋めとは別に、**実測や判断が必要な値**には `TODO(you)` を付けています。答えはなく、`answers/` にも仮の値しか入っていません。

| ファイル | 内容 | 章 |
| --- | --- | --- |
| `pyproject.toml` | 依存パッケージの版の固定 | [env/12](env/12_inference_envs.md) |
| `ros2_ws/src/evla_sim/urdf/so101_evla.urdf.xacro` | TCP と手首カメラの位置、衝突マスクの入れ方 | [31](track2/31_gazebo_scene.md) |
| `ros2_ws/src/evla_sim/config/scene.yaml` | キューブの大きさ | [31](track2/31_gazebo_scene.md)、[32](track2/32_grasp_judge_node.md) |
| `ros2_ws/src/evla_sim/config/grasp.yaml`、`opening_table.csv` | 把持判定の閾値、グリッパ角→開き幅の表 | [32](track2/32_grasp_judge_node.md) |
| `configs/datagen/so101_geometry.yaml` | IK に使うリンク長・零点・関節の範囲（URDF から読む） | [33](track2/33_moveit_demo_gen.md) |
| `configs/datagen/default.yaml` | home 姿勢、閉じ角、ランダム化の範囲 | [33](track2/33_moveit_demo_gen.md) |
| `src/evla/dataset/conventions.py` | 関節名・符号・グリッパの角度範囲の確定（【要確認】） | [34](track2/34_episode_recorder.md) |
| `src/evla/sim/closed_loop.py` | 失敗エピソードの動画の保存 | [27](track1/27_closed_loop_b_c.md) |
| `src/evla/dataset/to_lerobot.py` | 追加メタデータの残り（cameras.json など） | [34](track2/34_episode_recorder.md) |
| `configs/sweep/stage1.yaml`、`scripts/make_report_tables.py` | vla.cpp の掃引の行、図の出力 | [28](track1/28_sweep_and_pareto.md) |

**用意していないスクリプト**（必要になったら、近いものを写して作る）

| 章で挙げたもの | 代わりに使うもの |
| --- | --- |
| `tests/sim/test_libero_obs.py`、`tests/policy/test_{torch,ov,vlacpp}_policy.py` | モデルが要るので、章の完了条件のスモークテストで確かめる |
| `scripts/check_equivalence_so101.py`（36章） | `scripts/check_equivalence.py` を写し、観測を LeRobotDataset から作る |
| `scripts/analyze_gazebo_eval.py`（38章） | `evla.analysis.rejudge.sensitivity_curve` と `evla.analysis.plots.grasp_ecdf_plot` を呼ぶ |
| `configs/eval/scenes_v1.yaml`（38章） | `eval_runner` がシード 900000＋i から配置を作る（全設定で同じ配置になる） |
| `src/evla/sim/realtime.py`（27章、任意） | なし（必要なら `AsyncExecutor` をスレッドで書き直す） |

## 6. 章の本文と、穴埋め版のコードの違い

穴埋め版を作るときに、テストしやすさのために章の仕様から少し変えたところがあります。コードの方を正としてください。

| 章 | 変更点 | 理由 |
| --- | --- | --- |
| [22](track1/22_policy_interface.md) | `Policy` を `with` 文で使えるようにした。`warmup()` は各回のレイテンシのリストを返す | 終了処理の漏れを防ぐ。初回の時間を記録するため |
| [26](track1/26_bench_mode_a.md) | `ModeAConfig.chunk_size`（固定ノイズの形）を追加 | OV / vla.cpp は1回推論するまでチャンク長が分からないため |
| [27](track1/27_closed_loop_b_c.md) | `ExecutorConfig` に `gripper_dim`・`noise_dim` を追加。hold="zero" でもグリッパは直前の値を保つ。非同期の最初の1回は同期で推論する | 保持中にグリッパが勝手に開閉しないように。最初はロボットが止まっているので、チャンクの先頭から使えるため |
| [28](track1/28_sweep_and_pareto.md) | `expand()` は `SweepSpec`（設定の一覧＋周回数など）を返す。再開の印は `results/sweeps/<name>/<key>_r<周>.done` | 実行に必要な情報を1つにまとめるため |
| [32](track2/32_grasp_judge_node.md) | `grasp_candidate` は「閉じ指令の立ち上がり」ごとに1回だけ出す。失敗の分類 `classify_failure()` を `events.py` に置いた | 30Hz で同じ候補を出し続けないように。記録と評価で同じ分類を使うため |
| [33](track2/33_moveit_demo_gen.md) | 区間の組み立てを ROS 非依存の `evla/planning/build.py` に移した。上下の直線移動は自作の IK、関節空間の移動は MoveIt（`demo_generator.MoveItJointPlanner`）か `joint_interp` | MoveIt なしでテストできるように。MoveIt が動かなくても先に進めるように |
| [34](track2/34_episode_recorder.md) | 時刻つきのバッファ `StampedBuffer` を `evla/common/` に置いた。Gazebo とのやり取りを `sim_io.SimIO` にまとめ、`episode_runner` と `eval_runner` で共有した | ROS なしでテストできるように。リセットの手順を1か所にするため |
| [37](track2/37_policy_node_zmq.md) | 画像は `image.<カメラ名>` という名前の配列で送る。行動チャンク→関節名の変換 `chunk_to_urdf()` を `conventions.py` に置いた | 対応表を1か所にするため（記録と制御で同じ変換を使う） |

## 7. コード中の【要確認】

LeRobot・physicalai・MoveItPy などは版によって API が変わります。執筆時に実物で確かめられなかった行には `【要確認】` を付けました。**その行で動かなかったら、まず自分の環境の版のソースを確認**してください（`python -c "import lerobot, inspect; print(inspect.getsourcefile(lerobot))"` で場所が分かります）。

```bash
grep -rn "【要確認】" src scripts ros2_ws/src/evla_*
```
