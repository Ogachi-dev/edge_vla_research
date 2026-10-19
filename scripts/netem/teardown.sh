#!/usr/bin/env bash
set -euo pipefail
tc qdisc del dev veth0 root 2>/dev/null || true
ip link del veth0 2>/dev/null || true
ip netns del remote 2>/dev/null || true
