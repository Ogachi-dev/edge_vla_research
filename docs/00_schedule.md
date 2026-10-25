# 00. 日程索引（D1〜D10）と判定ゲート

> 実装はハンズオン形式です。各日のタスクで開く章の5節「ハンズオン」を、Step 0 から順に進めます（表の「Step」は、その5節の Step 番号）。穴埋めの共通ルールと全体の一覧は [03_fill_in_guide.md](03_fill_in_guide.md)。

計画書の「日程と判定ゲート」を、**その日にやるタスクのチェックリスト**にしたものです。各タスクから、実装の章へリンクしています。

## 1. 1日の進め方

- 一人で2トラックを並行するので、**午前＝トラック1、午後＝トラック2、夜＝長時間の実行**を基本にします。
- 夜に回すもの（掃引、データ生成、学習、評価）は、**夕方までに小さく試運転しておく**（[28](track1/28_sweep_and_pareto.md) の6節）。
- 毎日の終わりに、`docs/notes/YYYY-MM-DD.md` に「やったこと・ハマったこと・明日の最初の作業」を3〜5行で書く。

凡例：⏱ 目安時間、🌙 夜間の実行、★ その日の最低ライン（これだけは終わらせる）

## 2. 事前準備（D0、半日〜1日）

| タスク | 章 | ⏱ |
| --- | --- | --- |
| [ ] リポジトリの骨組み、`pyproject.toml`、`src/evla/common/` | [01](01_repository_design.md) | 3h |
| [ ] 規約を読む | [02](02_best_practices.md) | 1h |
| [ ] `.wslconfig`、使うディストロ（Ubuntu 22.04）の確認 | [env/10](env/10_wsl2_and_igpu.md) Step 1〜2 | 1h |

## 3. 前半（D1〜D5）：計測基盤とデータ

### D1

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 1 | [ ] ★ iGPUの認識：`/dev/dxg` → `clinfo` → OpenVINOのデバイス一覧 | [env/10](env/10_wsl2_and_igpu.md) Step 3 | 2h |
| 1 | [ ] `.venv-bench` を作る（CPU版torch、lerobot、openvino、physicalai） | [env/12](env/12_inference_envs.md) Step 1 | 2h |
| 1 | [ ] vCPUのP/Eコア推定実験 | [env/10](env/10_wsl2_and_igpu.md) Step 4 | 1.5h |
| 2 | [ ] ★ ROS 2 Humble＋Gazebo Fortress、WSLgの描画確認 | [env/11](env/11_ros2_gazebo.md) Step 1〜2 | 2h |
| 2 | [ ] ★ lerobot_wsをビルドし、SO-101をGazeboとMoveItで動かす | [env/11](env/11_ros2_gazebo.md) Step 3〜4 | 2h |
| 2 | [ ] 関節の対応と符号の表 | [env/11](env/11_ros2_gazebo.md) Step 5 | 1h |
| 🌙 | vla.cpp CPUのビルド（時間がかかれば夜に） | [env/12](env/12_inference_envs.md) Step 3 | |

### D2

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 1 | [ ] ★ LIBEROの起動（MuJoCoの描画の確認）、`lerobot-eval` で1回成功 | [21](track1/21_libero_setup.md) Step 1〜2 | 2h |
| 1 | [ ] `LiberoTaskEnv` と一致テスト | [21](track1/21_libero_setup.md) Step 3〜4 | 3h |
| 1 | [ ] ★ `Policy` 抽象＋PyTorch CPUベースライン | [22](track1/22_policy_interface.md)、[23](track1/23_runtime_pytorch.md) | 4h |
| 2 | [ ] ★ Gazeboシーン：机・キューブ・箱・2カメラ・TCP | [31](track2/31_gazebo_scene.md) Step 1〜5、7 | 4h |
| 2 | [ ] 衝突マスク、DetachableJoint、サービスの確認 | [31](track2/31_gazebo_scene.md) Step 6、8 | 2h |

### D3

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 1 | [ ] ★ モードAのベンチ基盤（計時、メモリ、結果の保存） | [26](track1/26_bench_mode_a.md) | 4h |
| 1 | [ ] OpenVINO（CPU / GPU）のアダプタ | [24](track1/24_runtime_openvino.md) | 3h |
| 1 | [ ] vla.cpp（CPU、可能ならSYCL / OV）のアダプタ | [25](track1/25_runtime_vlacpp.md)、[env/12](env/12_inference_envs.md) Step 4〜5 | 3h |
| 1 | [ ] 一致の表（全ランタイム） | [22](track1/22_policy_interface.md) | 1h |
| 2 | [ ] ★ 把持判定のコア（純粋関数）とテスト → ROSノード | [32](track2/32_grasp_judge_node.md) | 4h |
| 2 | [ ] ★ MoveItで、固定配置のpick&placeを1回成功させる | [33](track2/33_moveit_demo_gen.md) Step 1〜2、4〜5 | 3h |
| 2 | [ ] ロックステップの記録ノード | [34](track2/34_episode_recorder.md) Step 1〜4 | 4h |

D3は作業量が最も多い日です。遅れたら、vla.cppのSYCL / OVとトラック1の一致の表をD4に回します。

### D4

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 1 | [ ] ★ 掃引の仕組み（展開、子プロセス実行、再開） | [28](track1/28_sweep_and_pareto.md) Step 1〜3 | 3h |
| 1 | [ ] 段1〜3の試運転（`n_measure=3`） | [28](track1/28_sweep_and_pareto.md) | 1h |
| 2 | [ ] ★ ランダム化、立て直しの注入、頑健化 | [33](track2/33_moveit_demo_gen.md) Step 3、6 | 4h |
| 2 | [ ] 変換スクリプト（生 → LeRobotDataset）、品質チェック | [34](track2/34_episode_recorder.md) Step 5〜6 | 3h |
| 2 | [ ] 20エピソードの試運転 → 所要時間の見積もり | [33](track2/33_moveit_demo_gen.md) Step 6 | 1h |
| 🌙 | **デモ生成（成功100〜200エピソード）** と **掃引の段1〜3**（CPUを取り合うので、どちらを優先するか決める。**【判断】** データ生成を優先し、掃引はD5の昼に回す） | | |

### D5

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 2 | [ ] ★ データの品質確認 → **判定ゲート**（下記） | [34](track2/34_episode_recorder.md) 6節 | 2h |
| 2 | [ ] ★ （ゲート通過なら）学習機へ転送し、学習を開始する | [35](track2/35_finetune_smolvla.md) Step 1〜4 | 2h |
| 1 | [ ] 掃引の段1〜3（昼の間、学習機で学習している間に） | [28](track1/28_sweep_and_pareto.md) | （実行） |
| 1 | [ ] executor（同期 / 非同期）と偽の環境でのテスト | [27](track1/27_closed_loop_b_c.md) Step 1〜3 | 4h |
| 🌙 | **LIBEROの閉ループ（ロックステップ）：上位3設定の成功率** | [27](track1/27_closed_loop_b_c.md) Step 4〜5 | |

## 4. D5 判定ゲート

計画書：「デモ生成が安定 → ファインチューニングへ ／ 不安定 → トラック2はROS 2経由のレイテンシ測定のみに縮小」

**【判断】**「安定」を、次の**すべて**を満たすことと定義する。

| # | 基準 | 確認方法 |
| --- | --- | --- |
| 1 | 成功エピソードが100以上ある | 変換後のデータセットのエピソード数 |
| 2 | 生成の成功率が、直近50エピソードで60%以上 | 生成ログ（`generation.json`） |
| 3 | 夜間の実行が、手で介入せずに完走した（または再開の仕組みで自動回復した） | 生成ログ |
| 4 | 品質チェックで致命的な問題がない（フレーム落ちなし、黒画像なし、LeRobotDatasetとして読める） | [34](track2/34_episode_recorder.md) の品質レポート |
| 5 | 目視で10エピソードを確認し、明らかにおかしな動きがない | 動画 |

**通過しなかった場合（縮小案）**

- トラック2は「**ROS 2経由のレイテンシ測定**」に縮小する。LIBEROのモデルを流用し、成功率は問わない。
- やること：[37](track2/37_policy_node_zmq.md)（ポリシーノードとZeroMQ）を作り、Gazebo上で推論の遅延・実効制御周期・キュー枯渇を、トラック1の上位設定で測る。
- 浮いた時間はトラック1（モードCの点の追加、上限確認のπ0 / GR00T）に回す。
- 判断の理由を `docs/decisions.md` に記録する。

## 5. 後半（D6〜D10）：閉ループ評価とまとめ

### D6

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 1 | [ ] ★ 遅延注入（仮想時間）：遅延×n_exec→成功率 | [27](track1/27_closed_loop_b_c.md) 6.1節 | （実行） |
| 2 | [ ] ★ ZeroMQのプロトコル、推論サーバー、クライアント | [37](track2/37_policy_node_zmq.md) Step 1〜3 | 4h |
| 2 | [ ] ポリシーノード（推論は別プロセス） | [37](track2/37_policy_node_zmq.md) Step 4 | 3h |
| 🌙 | 遅延許容曲線の残りの点 | | |

### D7

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 1 | [ ] ★ 同期と非同期の比較 | [27](track1/27_closed_loop_b_c.md) | （実行） |
| 1 | [ ] ★ パレート図、ベンチマーク表 | [28](track1/28_sweep_and_pareto.md) Step 4〜6 | 3h |
| 1 | [ ] トラック2に渡す最適設定を決めて記録する | [28](track1/28_sweep_and_pareto.md) DoD | 1h |
| 2 | [ ] ★ Gazebo閉ループの疎通確認（LIBEROのモデルでパイプラインを検証） | [37](track2/37_policy_node_zmq.md) Step 5 | 3h |
| 2 | [ ] 学習の監視、チェックポイントのオフライン評価 | [35](track2/35_finetune_smolvla.md) Step 5 | 1h |

### D8

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 2 | [ ] ★ 学習済みモデルをOV / GGUFへ変換し、一致を確認する | [36](track2/36_model_conversion.md) | 4h |
| 2 | [ ] ★ Gazebo閉ループ評価（基準＋最適設定、ロックステップ） | [38](track2/38_gazebo_eval.md) | （実行） |
| 1 | [ ] （任意）上限確認：π0・GR00T（Q4 / INT8） | [25](track1/25_runtime_vlacpp.md) 10節 | 2h |
| 🌙 | 精度違いの設定の評価、実時間モードの評価 | [38](track2/38_gazebo_eval.md) | |

### D9

| トラック | タスク | 章 | ⏱ |
| --- | --- | --- | --- |
| 共通 | [ ] ★ 閾値の感度分析、把持精度のECDF | [38](track2/38_gazebo_eval.md) Step 5 | 3h |
| 共通 | [ ] ★ 電力・温度の突き合わせ | [40](common/40_metrics_measurement.md) | 2h |
| 共通 | [ ] Jetsonへの外挿 | [41](common/41_analysis_reporting.md) 4.1 | 2h |
| 共通 | [ ] （任意）外部推論との比較（tc netem / 仮想時間） | [41](common/41_analysis_reporting.md) 4.2 | 3h |

### D10

| タスク | 章 | ⏱ |
| --- | --- | --- |
| [ ] ★ 全図表をスクリプトで作り直す | [41](common/41_analysis_reporting.md) | 1h |
| [ ] ★ 報告書 | [41](common/41_analysis_reporting.md) 4.3 | 5h |
| [ ] 再現性の確認、リポジトリの整理、README | [01](01_repository_design.md) | 2h |

## 6. 計画書との対応表

計画書の項目が、どの章で実装されるかの一覧です。漏れがないことを確認するために使います。

### 6.1 トラック1

| 計画書の項目 | 章 |
| --- | --- |
| 対象モデル：SmolVLA LIBERO（OV版・GGUF版） | [23](track1/23_runtime_pytorch.md)、[24](track1/24_runtime_openvino.md)、[25](track1/25_runtime_vlacpp.md) |
| 対象モデル：上限確認 π0 / π0.5 / GR00T | [25](track1/25_runtime_vlacpp.md) 10節 |
| 掃引軸：ランタイム（PyTorch CPU、OV CPU / GPU、vla.cpp CPU / SYCL） | [23](track1/23_runtime_pytorch.md)〜[25](track1/25_runtime_vlacpp.md)、[28](track1/28_sweep_and_pareto.md) |
| 掃引軸：精度（FP32、FP16、INT8、Q8_0、Q4_0） | [24](track1/24_runtime_openvino.md) 2.2、[25](track1/25_runtime_vlacpp.md) 5.1 |
| 掃引軸：入力（解像度、視点数） | [23](track1/23_runtime_pytorch.md)、[25](track1/25_runtime_vlacpp.md)、[28](track1/28_sweep_and_pareto.md) 段3 |
| 掃引軸：CPU（スレッド数、P/E） | [env/10](env/10_wsl2_and_igpu.md) Step 4、[26](track1/26_bench_mode_a.md) |
| 測定モードA / B / C | [26](track1/26_bench_mode_a.md)、[27](track1/27_closed_loop_b_c.md) |
| 遅延注入、許容遅延×チャンク長→成功率 | [27](track1/27_closed_loop_b_c.md) |
| 同期 vs 非同期 | [27](track1/27_closed_loop_b_c.md) |

### 6.2 トラック2

| 計画書の項目 | 章 |
| --- | --- |
| SO-101のURDF、ros2_control、MoveIt 2、TCPフレーム | [env/11](env/11_ros2_gazebo.md)、[31](track2/31_gazebo_scene.md) |
| データセット仕様（v3.0、6次元、関節名・順序、度、top＋wrist、480×640、30fps、100〜200、ロックステップ） | [34](track2/34_episode_recorder.md) |
| 追加メタデータ（カメラ、URDF版・単位・符号、初期位置・シード、成否・ずれ・閾値） | [34](track2/34_episode_recorder.md) 4.4 |
| データの多様化（配置、軌道、見た目、リカバリー） | [33](track2/33_moveit_demo_gen.md)、[31](track2/31_gazebo_scene.md) |
| ポリシー実行の構成（カメラ→推論→チャンク→JTC） | [37](track2/37_policy_node_zmq.md) |
| ファインチューニング | [35](track2/35_finetune_smolvla.md) |
| OV / GGUFへの変換 | [36](track2/36_model_conversion.md) |
| ROS 2閉ループ評価 | [38](track2/38_gazebo_eval.md) |

### 6.3 把持判定とDetachableJoint

| 計画書の項目 | 章 |
| --- | --- |
| 把持条件1（軸ごとの箱、開閉方向は開き幅から） | [32](track2/32_grasp_judge_node.md) |
| 把持条件2（閉じ指令または閉じ角） | [32](track2/32_grasp_judge_node.md) |
| 把持条件3（ヨー±15°、90°周期） | [32](track2/32_grasp_judge_node.md) |
| 把持条件4（任意：接近方向30°以内） | [32](track2/32_grasp_judge_node.md) |
| 解放、タスク成功（箱の内側・縁の下・Nフレーム静止） | [32](track2/32_grasp_judge_node.md) |
| 閾値の決め方（幾何から、メタデータに記録） | [32](track2/32_grasp_judge_node.md)、[34](track2/34_episode_recorder.md) |
| 注意点：起動時は接続済み → 開始時にdetach | [31](track2/31_gazebo_scene.md)、[32](track2/32_grasp_judge_node.md)、[34](track2/34_episode_recorder.md) |
| 注意点：爪とキューブの衝突 → collide_bitmask | [31](track2/31_gazebo_scene.md) 4.4 |
| 注意点：DARTの初期状態の衝突 → 初期配置を離す | [31](track2/31_gazebo_scene.md) |
| 注意点：キューブは1つにしてset_poseで再配置 | [31](track2/31_gazebo_scene.md)、[34](track2/34_episode_recorder.md) |
| 注意点：bridge経由の遅延 → 接続確認後の相対姿勢を記録 | [32](track2/32_grasp_judge_node.md) |
| 判定ノードの流れ（TF、bridge、joint_states → attach / detach） | [32](track2/32_grasp_judge_node.md) |

### 6.4 評価指標・測定条件・リスク・成果物

| 計画書の項目 | 章 |
| --- | --- |
| 指標8つ（レイテンシ、実効制御周期、成功率、把持精度、メモリ、電力、RTF、温度・クロック） | [40](common/40_metrics_measurement.md) 4.1 |
| 測定条件の統一（電源プラン、P/E固定、ウォームアップ、温度） | [02](02_best_practices.md) 6節、[26](track1/26_bench_mode_a.md)、[40](common/40_metrics_measurement.md) 6節 |
| 閾値の感度分析 | [38](track2/38_gazebo_eval.md) |
| リスク：シミュレーションと推論のiGPU競合 | [20](track1/20_overview.md) 4.1（モード分け）、[27](track1/27_closed_loop_b_c.md)（仮想時間方式）、[38](track2/38_gazebo_eval.md) |
| リスク：WSL2での描画 | [env/11](env/11_ros2_gazebo.md)、[21](track1/21_libero_setup.md) Step 1、[31](track2/31_gazebo_scene.md) Step 7 |
| リスク：ROS 2とLeRobotの依存衝突 | [01](01_repository_design.md) 4.1、[34](track2/34_episode_recorder.md)、[37](track2/37_policy_node_zmq.md) |
| リスク：MoveItのデモの単調さ | [33](track2/33_moveit_demo_gen.md) |
| リスク：DetachableJointの挙動 | [31](track2/31_gazebo_scene.md)、[32](track2/32_grasp_judge_node.md) |
| リスク：OpenVINO Physical AIのAPI変更 | [24](track1/24_runtime_openvino.md) 6節、[36](track2/36_model_conversion.md) |
| リスク：D5時点でデモ生成が不安定 | 本章 4節 |
| リスク：サーマルスロットリング | [40](common/40_metrics_measurement.md) |
| 成果物7点 | [41](common/41_analysis_reporting.md) 3節 |
| D9：外部推論との比較（tc netem）、Jetson Orinの公開値との照合 | [41](common/41_analysis_reporting.md) 4.1〜4.2 |
