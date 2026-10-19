#!/usr/bin/env bash
# 外部推論の模擬：ネットワーク名前空間 remote と veth ペアを作り、veth0 に netem をかける。docs/common/41 4.2
# 例: sudo bash scripts/netem/setup.sh 50ms 10ms 1%
set -euo pipefail
DELAY=${1:-20ms}; JITTER=${2:-5ms}; LOSS=${3:-0%}
modprobe sch_netem  # WSL2 では netem がモジュール
ip netns add remote 2>/dev/null || true
ip link add veth0 type veth peer name veth1 2>/dev/null || true
ip link set veth1 ___  # 【穴①】veth1 を名前空間 remote に移す指定
ip addr add 10.200.0.1/24 dev veth0 2>/dev/null || true
ip netns exec remote ip addr add 10.200.0.2/24 dev veth1 2>/dev/null || true
ip link set veth0 up
ip netns exec remote ip link set veth1 up
ip netns exec remote ip link set lo up
tc qdisc replace dev veth0 root ___ distribution normal loss "$LOSS"  # 【穴②】遅延とそのゆらぎを与える netem の指定
echo "推論サーバー: sudo ip netns exec remote <python> scripts/serve_policy.py --bind tcp://10.200.0.2:5600 ..."
echo "確認: ping -c 5 10.200.0.2"
