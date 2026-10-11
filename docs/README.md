# エッジVLA推論検証 実装ガイド（docs/）

[`edge_vla_research_plan.md`](../edge_vla_research_plan.md)（以下「計画書」）を、**自分の手で実装できる粒度**まで分解したドキュメント群です。
コードは原則として自分で書く前提です。このガイドが書くのは「何を・どのファイルに・どのAPIで・どの順で・どこまでできたら完了か」までで、本体ロジックの完成コードは載せていません。

---

## 1. 読み方

1. まず [00_schedule.md](00_schedule.md) で、D1〜D10のどこに今いるかを確認する。
2. その日のタスクからリンクされた章を開き、上から順に進める。
3. 詰まったら章末の「ヒント」を **L1 → L2 → L3** の順に1段ずつ開く。いきなりL3を開かないこと。
4. 「完了条件（DoD）」をすべて満たしたら、チェックを付けて次へ進む。

横断的な決めごとは、最初に一度読んでおく。

- [01_repository_design.md](01_repository_design.md)：ディレクトリ構成、venvの分け方、実験IDと結果ファイルの形式
- [02_best_practices.md](02_best_practices.md)：コーディング規約、テスト、ロギング、計測の作法

## 2. 章一覧

| 区分 | 章 | 内容 |
| --- | --- | --- |
| 全体 | [00_schedule](00_schedule.md) | 日程索引、D5判定ゲート、計画書との対応表 |
| | [01_repository_design](01_repository_design.md) | リポジトリ構成、プロセスとvenvの分離、データ形式 |
| | [02_best_practices](02_best_practices.md) | 実装・テスト・計測の作法 |
| 環境 | [env/10_wsl2_and_igpu](env/10_wsl2_and_igpu.md) | WSL2の設定、iGPU（Iris Xe）、CPUトポロジの罠 |
| | [env/11_ros2_gazebo](env/11_ros2_gazebo.md) | Ubuntu 24.04、ROS 2 Jazzy、Gazebo Harmonic、lerobot_ws |
| | [env/12_inference_envs](env/12_inference_envs.md) | 推論用venv（uv）、vla.cppのビルド（CPU / SYCL / OpenVINO） |
| トラック1 | [track1/20_overview](track1/20_overview.md) | 全体像、測定モードA/B/C |
| | [track1/21_libero_setup](track1/21_libero_setup.md) | LIBEROの導入とヘッドレス描画 |
| | [track1/22_policy_interface](track1/22_policy_interface.md) | ランタイム共通のPolicy抽象 |
| | [track1/23_runtime_pytorch](track1/23_runtime_pytorch.md) | PyTorch CPUベースライン |
| | [track1/24_runtime_openvino](track1/24_runtime_openvino.md) | OpenVINO（CPU / GPU）、INT8 |
| | [track1/25_runtime_vlacpp](track1/25_runtime_vlacpp.md) | vla.cpp（GGUF、CPU / SYCL / OV） |
| | [track1/26_bench_mode_a](track1/26_bench_mode_a.md) | 単体ベンチ（計時、統計、メモリ） |
| | [track1/27_closed_loop_b_c](track1/27_closed_loop_b_c.md) | ロックステップ、リアルタイム、遅延注入、同期/非同期 |
| | [track1/28_sweep_and_pareto](track1/28_sweep_and_pareto.md) | 掃引設計、集計、パレート図 |
| トラック2 | [track2/30_overview](track2/30_overview.md) | ROSグラフとプロセス境界 |
| | [track2/31_gazebo_scene](track2/31_gazebo_scene.md) | world、カメラ、TCP、衝突マスク、bridge |
| | [track2/32_grasp_judge_node](track2/32_grasp_judge_node.md) | 把持判定とDetachableJoint |
| | [track2/33_moveit_demo_gen](track2/33_moveit_demo_gen.md) | MoveItでのデモ自動生成 |
| | [track2/34_episode_recorder](track2/34_episode_recorder.md) | ロックステップ記録とLeRobotDataset v3への変換 |
| | [track2/35_finetune_smolvla](track2/35_finetune_smolvla.md) | RTX PRO 6000でのファインチューニング |
| | [track2/36_model_conversion](track2/36_model_conversion.md) | OpenVINO IR / GGUFへの変換と数値一致の検証 |
| | [track2/37_policy_node_zmq](track2/37_policy_node_zmq.md) | ROS 2ポリシーノードとZeroMQ IPC |
| | [track2/38_gazebo_eval](track2/38_gazebo_eval.md) | 閉ループ評価と閾値の感度分析 |
| 共通 | [common/40_metrics_measurement](common/40_metrics_measurement.md) | 指標の取得方法と測定条件の統一 |
| | [common/41_analysis_reporting](common/41_analysis_reporting.md) | 図表、Jetsonへの外挿、外部推論との比較、報告書 |

## 3. 各章の共通テンプレート

すべての実装章は、次の10節で構成しています。

| # | 節 | 中身 |
| --- | --- | --- |
| 1 | 目的と計画書の対応 | この章で何を作るか。計画書のどの節に対応するか |
| 2 | 前提知識・事前に読む資料 | 実装前に目を通す公式ドキュメント |
| 3 | 成果物 | 作るファイルのパスと役割 |
| 4 | インタフェース仕様 | シグネチャ、dataclass、設定スキーマ、トピック定義 |
| 5 | 実装手順 | 番号付きの小さなステップ（1つ30分〜2時間が目安） |
| 6 | ベストプラクティス | 推奨するやり方と、その理由 |
| 7 | 落とし穴 | ハマりやすい点と対処 |
| 8 | ヒント | `<details>` で折りたたんだ段階的ヒント |
| 9 | 完了条件（DoD） | 確認コマンドと期待結果、テスト観点 |
| 10 | 理解度チェック・発展課題 | 学習の確認用の問い |

## 4. 凡例

| 表記 | 意味 |
| --- | --- |
| **【要確認】** | 執筆時点（2026-10-12）で一次情報を確認しきれなかった事項。バージョンで変わりうる。添えてある確認方法で、実装前に確かめる |
| **【判断】** | 計画書に書かれておらず、このガイドで決めた設計判断。変えてもよいが、理由を `docs/decisions.md`（自分で作る）に残す |
| **【計画書との差分】** | 調査の結果、計画書の前提を修正したほうがよい箇所 |
| `TODO(you)` | 雛形の中で、自分で埋める箇所 |
| L1 / L2 / L3 | ヒントの深さ。L1＝方針、L2＝使うAPIや関数名、L3＝疑似コード |

## 5. 調査で判明した重要事項（計画書との差分のまとめ）

執筆前に一次情報（各リポジトリのソース・ドキュメント・HFのモデルカード）を確認しました。計画書に影響する事項を先にまとめます。詳細は各章にあります。

1. **WSL2ではP/Eコアを区別できない**：WSL2のゲストからは「8コア×2スレッド」の均質なCPUに見え、`/sys/devices/cpu_core` も存在しない（この端末で確認済み）。WSL内の `taskset` ではPコアだけに固定できない。→ [env/10](env/10_wsl2_and_igpu.md#7-落とし穴)、[26](track1/26_bench_mode_a.md)
2. **WSL2に見えているメモリは15GB**：64GB搭載だが、`.wslconfig` の既定でWSLには約半分しか割り当てられない。→ [env/10](env/10_wsl2_and_igpu.md)
3. **LeRobotはPython 3.12以上が必須**（v0.6系）。現在のUbuntu 22.04（Python 3.10）では動かないので、Ubuntu 24.04に統一する。→ [env/11](env/11_ros2_gazebo.md)、[env/12](env/12_inference_envs.md)
4. **OpenVINO版SmolVLAの配布元が変わった**：計画書の `modelapi/smolvla-libero-fp16-ov-catalog` は現在アクセスできない（401）。代わりに `OpenVINO/smolvla-libero-{fp16,bf16,w8a16}-ov` が公開されている（Physical AIフレームワーク `physicalai` 向け）。→ [24](track1/24_runtime_openvino.md)
5. **iGPUを使う経路は3つある**：OpenVINO（physicalai）、vla.cpp＋SYCL、vla.cpp＋OpenVINOバックエンド（`GGML_OPENVINO_DEVICE=GPU`）。→ [25](track1/25_runtime_vlacpp.md)
6. **CPU推論は1回あたり約2秒かかる見込み**：vla.cppの公開値では、デスクトップCPU（i7-14700F）でSmolVLA（2視点・512px）のp50は約1.7秒で、うち約1.1秒がビジョン部。i7-1360Pはさらに遅いと見込む。チャンク50・30Hzの1チャンク分（1.67秒）より長いので、**非同期実行と解像度・視点数の掃引が研究の核心**になる。→ [20](track1/20_overview.md)
7. **lerobot_wsのSO-101は関節名が "1"〜"6"**：公開データの `shoulder_pan.pos` などとは名前が違う。コントローラの `update_rate` は10Hzで、TCPリンクもない。腕は5自由度なので、6自由度の姿勢を指定するIKは解けないことが多い。→ [31](track2/31_gazebo_scene.md)、[33](track2/33_moveit_demo_gen.md)
8. **公開SO-101データのグリッパは0〜100の正規化値**：計画書は「度単位で統一」としているが、公開データのグリッパ列は度ではない（統計値0.9〜70.2）。→ [34](track2/34_episode_recorder.md)
9. **ROS 2 Jazzy（NumPy 1.26）とLeRobot（NumPy 2以上が必須）は同じプロセスに同居できない**。記録ノードは生データを書き出すだけにし、LeRobotDatasetへの変換は別venvのスクリプトで行う。→ [34](track2/34_episode_recorder.md)
10. **WSL2ではRAPLが見えない**（`/sys/class/powercap` がない）。電力・温度・クロックはWindows側で測る。→ [40](common/40_metrics_measurement.md)
