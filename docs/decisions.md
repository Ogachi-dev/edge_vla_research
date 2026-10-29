# 設計判断の記録

【判断】や計画書からの変更を、決めた日付と理由とともに1件ずつ残します（[01](01_repository_design.md) 5節、[README](README.md) の【判断】の凡例）。

## 2026-10-26：検証端末の環境を Ubuntu 22.04＋ROS 2 Humble＋Gazebo Fortress にする

**決めたこと**：計画書の「WSL2 に Ubuntu 24.04 を追加し、ROS 2 Jazzy＋Gazebo Harmonic で構築する」をやめ、ROS 2 Humble（`ros-humble-desktop`）を入れてある既存の WSL（Ubuntu 22.04）で進める。Gazebo は Humble の apt の標準の組み合わせである Fortress にする。

**理由**：検証端末の WSL には 22.04＋Humble がすでに入っていて、`colcon build` も通っている。ディストロを追加して環境を作り直すより、この環境に合わせて条件を直すほうが早い。

**確かめたこと**（2026-10、ソースと apt で確認）

| 項目 | 結果 |
| --- | --- |
| 端末 | Ubuntu 22.04.5、カーネル 6.6（WSL2）、16 vCPU、メモリ 23GB（WSL から見える量）、`/dev/dxg` あり |
| ROS 側の Python | システムの 3.10.12。apt の NumPy は 1.21.5。`~/.local` に `pip install --user` した NumPy 2.2.6 があり、何もしないとこちらが先に読まれる |
| bench 側の Python | uv が入れた 3.12（`.venv-bench`）。LeRobot 0.6 の要求（3.12 以上）を満たす |
| Gazebo | apt で入るのは Fortress（`ignition-fortress`）だけ。Humble 向けの Harmonic のパッケージ（`ros-humble-ros-gzharmonic`）は apt に見つからない |
| `gz_ros2_control` | `ros-humble-gz-ros2-control` 0.7.x。URDF の書き方（`gz_ros2_control-system`、`gz_ros2_control/GazeboSimSystem`）は lerobot_ws と同じ |
| SDF のプラグイン名 | Fortress（ign-gazebo6）は `gz-sim-*` のファイル名を読み替え、`gz::sim::systems::*` の別名も登録しているので、`evla_sim` の world と URDF はそのまま読める |
| DetachableJoint | Fortress にも `attach_topic`、`detach_topic`、`output_topic` がある |
| ros_gz_bridge | humble 0.244.26（apt）。トピックの型名は `gz.msgs.*` と `ignition.msgs.*` のどちらも通る。`ControlWorld` と `SetEntityPose` のサービスもbridgeできる（0.244.17 で backport） |
| JointTrajectoryController | Humble でも `~/controller_state` の `reference` に目標値が入る（`sim_io.py` が使う） |
| MoveItPy | `ros-humble-moveit-py` 2.5.10 が apt にある |
| vla.cpp | CMake 3.22・C++17 を要求する。22.04 の CMake 3.22.1・gcc 11.4 で足りる。ビルド済みバイナリは使えない（glibc 2.35） |

**変えたもの**

- `src/evla`：Python 3.12 でしか使えない構文（`def f[T]`、`class C[T]`、`datetime.UTC`）を、3.10 でも動く書き方にした。`pyproject.toml` は `requires-python = ">=3.10"`、ruff の `target-version = "py310"`。
- `scripts/ros_env.sh`：`/opt/ros/humble` を読み、`ROS_LOCALHOST_ONLY=1`、`PYTHONNOUSERSITE=1`（`~/.local` の NumPy 2 を読まない）。
- docs：環境の記述（README、00、01、03、env/10〜12、track2、40）と CLI（`gz` → `ign`）。

**影響と残っている【要確認】**

- lerobot_ws は Jazzy＋Harmonic 向けなので、Humble でビルド・起動できるかは env/11 の Step 3〜4 で確かめる。直すときは lerobot_ws を書き換えず、`evla_sim` 側のオーバーレイで吸収する。
- メッシュが見えないときは `IGN_GAZEBO_RESOURCE_PATH` を確かめる（Fortress はこちらの変数を見る）。
- vla.cpp の SYCL 版：oneAPI 2026.x は 22.04 に対応していないので、2025.x で試す。動かなければ、iGPU は OpenVINO と vla.cpp の OpenVINO バックエンドで測る。
- Intel compute runtime：24.04 向けの PPA は使えない。22.04 向けの Intel のリポジトリか、22.04 の glibc で動く版の `.deb` を使う（env/10 Step 3）。
- 物理エンジンは Fortress でも DART（既定）なので、DetachableJoint と DART の注意点（31章）はそのまま当てはまる。Harmonic との細かな挙動の差は、31章 Step 6 で確かめる。

## 2026-10-26：Gazebo の描画エンジンを ogre（1.9）にする

**決めたこと**：GUI もカメラのセンサも、Fortress の既定の ogre2 ではなく ogre（1.9）で描画する。world テンプレートの `<render_engine>` を `ogre`、`sim.launch.py` に `render_engine` 引数（既定 `ogre`）を足し、GUI の既定（`~/.ignition/gazebo/6/gui.config`）も `ogre` にする。

**理由**：WSLg の D3D12 ドライバ（Mesa 23.2.1）は OpenGL 4.1 までで、ogre-next 2.2 の GL3Plus が使う機能がない。`ign gazebo shapes.sdf` の GUI と、ヘッドレス起動（`-s`）の上方カメラの両方が `Ogre::UnimplementedException`（`GL3PlusTextureGpu::copyTo`）で落ちた。ogre にすると、GUI は落ちず、上方カメラは 640×480 の RGB 画像を出した（`ros2 launch ros_gz_sim gz_sim.launch.py` 経由でも確認）。

**影響**：データ生成と評価の画像は ogre（1.9）の描画になる（ogre2 の PBR とは見た目が違う）。学習と評価で同じエンジンを使うので比較には影響しないが、報告書の「環境」には描画エンジンを書く。

**ついでに直したこと**：world テンプレートが光の明るさのランダム化（0.6〜1.2）を `<diffuse>` に入れていて、1 を超える値で Fortress の SDF の読み込み（sdformat 12）が失敗した。色は白のまま、明るさは `<intensity>` で与えるようにした。

## 2026-10-26：描画はソフトウェア描画（llvmpipe）＋ogre2 にする（上の「ogre（1.9）にする」を置き換える）

**決めたこと**：ogre（1.9）をやめ、Gazebo の描画エンジンを ogre2 に戻す。代わりに `scripts/ros_env.sh` で `LIBGL_ALWAYS_SOFTWARE=1` にして、ROS のシェルで起動する Gazebo と RViz を Mesa の llvmpipe（OpenGL 4.5）で描画する。

**理由**：ogre（1.9）を D3D12 で動かすと、lerobot_ws の SO-101 を表示したときにロボットが消えたり現れたりした。D3D12 には ogre2 が使う `GL_ARB_copy_image` がないが、llvmpipe にはある。llvmpipe＋ogre2 では、同じ launch でちらつきがなくなった（目視で確認）。カメラ付きの world をヘッドレスで起動しても、上方カメラは 640×480 の画像を出し、RTF は 0.98（D3D12＋ogre は 0.95）だった。

**影響**：描画が CPU を使うので、推論のベンチ（トラック1）と Gazebo を同時に動かすと CPU を取り合う。Gazebo の評価はロックステップ（sim 時刻）で行うので結果は変わらないが、所要時間は延びる。報告書の「環境」には「Gazebo は llvmpipe＋ogre2 で描画」と書く。上の判断で書き換えた `~/.ignition/gazebo/6/gui.config` は `ogre2` に戻した。

## 2026-10-27：vla.cpp の SYCL 版は oneAPI 2025.3 で、Level Zero API なしでビルドする

**決めたこと**：oneAPI は Base Toolkit 2025.3.1 を `~/intel/oneapi` に入れる。vla.cpp の SYCL 版は `-DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF` でビルドする。iGPU の compute runtime は、Intel の jammy 向けリポジトリの 24.39 を使う（env/10 Step 3。下の 2026-10-27 の判断で 25.13 に替えた）。

**理由**：Level Zero のヘッダ（`libze-dev`）がある環境では、ggml-sycl は起動時に `zesInit`（Sysman）を呼ぶ。WSL では compute runtime の中でここが segfault した（gdb で確認）。OFF にしてビルドし直すと、`backend = SYCL (device 0: Intel(R) Graphics [0xa7a0])` で `vla-bench` が最後まで動いた（`smolvla-libero.gguf`、2画像、512px：最短 3469 ms、中央値 5722 ms、p90 7636 ms）。上の「2026-10-26：検証端末の環境を…」の【要確認】（SYCL 版は 2025.x で試す、compute runtime の入手元）はこれで決まった。

**影響**：OFF にすると、ggml-sycl は GPU の空きメモリの問い合わせなどに Level Zero ではなく SYCL の API を使う。推論の処理自体は Level Zero のアダプタで GPU で動く。GPU の名前は製品名ではなく `Intel(R) Graphics [0xa7a0]` と出るので、報告書には「Iris Xe（i7-1360P、96EU、compute runtime 24.39）」と書く。

## 2026-10-27：iGPU の compute runtime を 25.13 にする（上の「24.39 を使う」を置き換える）

**決めたこと**：compute runtime は、GitHub のリリースの 25.13.33276.16（IGC 2.10.8、gmmlib 22.7.0）を `.deb` で入れる。Level Zero の loader（`libze1` 1.17）は Intel の jammy 向けリポジトリのまま使う。

**理由**：compute runtime の latest（26.x）は 24.04 以降向けで、glibc 2.38・libstdc++ 13 を要求するので 22.04 では動かない。25.13 は `libc6 (>= 2.34)`、`libstdc++6 (>= 12)` で、22.04 で動く一番新しい版である。jammy 向けのリポジトリは 24.39 で止まっている。

**確かめたこと**：`sycl-ls` に `[level_zero:gpu]`（1.6.33276）と `[opencl:gpu]`（25.13.33276.16）が出た。`build-sycl` は作り直さずに `backend = SYCL (device 0: Intel(R) Graphics [0xa7a0])` で動いた（2画像、512px、warmup 2・reps 10：最短 3229 ms、中央値 3844 ms、p90 5359 ms、vision 1721 ms）。24.39 のとき（warmup 3・reps 20：中央値 5722 ms）より速いが、回数と Windows 側のメモリの状態がそろっていないので、runtime の差とは言えない。比べるときは同じ条件で測り直す。

**影響**：報告書の「環境」には「compute runtime 25.13.33276.16（GitHub のリリース）」と書く。apt の `libze-intel-gpu1` を入れると 25.13 の `intel-level-zero-gpu` とぶつかるので入れない。

## 2026-10-28：提出の範囲をトラック1に縮め、学習をやめる

**決めたこと**：締め切り（10/30、レポートの作成を含む）に合わせて、提出をトラック1に絞る。この端末で、公開されている SmolVLA LIBERO（PyTorch 版、Intel の OV IR、GGUF）を各ランタイムで動かし、パラメータを少し変えて比べる。トラック2は学習（35章）以降をやめ、Gazebo・MoveIt の環境構築までの記録としてレポートに書く。

**理由**：残りの作業は計画の時間で約 90 時間あり、締め切りまでに使える時間（1日 4 時間 × 2 日）を大きく超える。トラック1は、学習なしで公開モデルだけで成り立つ。

**影響**：計画書の「トラック2：Gazebo での閉ループ評価」と「D5 の判定ゲート」は行わない。レポートでは、トラック2は「環境構築と、そこで分かった WSL での制約（描画、Humble・Fortress への置き換え）」として書く。

## 2026-10-29：vla.cpp SYCL 版の掃引は、夜間の実行の最後に単独で回す

**起きたこと**：10/28 の夜間の実行で、モードB（OV GPU、vla.cpp OV。合わせて約 70 分）のあとに掃引の段1を始めた。最初の実行（vla.cpp SYCL 版）がサーバーを起動した直後（02:31）に `dxgkio_reserve_gpu_va: Ioctl failed: -75` が出て、02:35 に WSL ごと止まった。Linux 側の OOM の記録はない。掃引とモードBの PyTorch 基準は回らなかった。

**原因の見立て**：Windows のコミット済みメモリが足りなくなった。iGPU は GPU のメモリもホストのメモリから取り、WSL からの確保（dxg）もホストのコミットに数えられる。WSL の VM は使ったメモリをすぐには返さないので、それと SYCL の確保が重なり、ページファイルの自動拡張が追い付かなかったと考える（確かめてはいない）。

**決めたこと**：
- Windows のページファイルを 100GB にし、`.wslconfig` の `memory` を 24GB から 16GB に下げる（ユーザーが設定する）。
- SYCL 版は `configs/sweep/stage1_sycl.yaml` に分け、`scripts/run_overnight.sh` の最後に単独で回す。工程の間には 60 秒の待ちを入れる。

**影響**：SYCL 版の掃引がまた止まっても、その前の結果は残る。止まった場合は、試運転の値（p50 6349 ms）をレポートに載せ、「長く回すと WSL ごと止まった」ことも結果として書く。`.wslconfig` を変えたので、10/28 の結果（memory=24GB）と 10/29 以降の結果では VM のメモリの上限が違う。モードAの数字を比べるときは、どちらの条件で測ったかを書く。
