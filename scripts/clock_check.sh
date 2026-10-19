#!/usr/bin/env bash
# WSL と Windows の時計のずれを確認する。docs/common/40_metrics_measurement.md 5
set -euo pipefail
wsl_ns=$(date +%s%N)
win=$(powershell.exe -NoProfile -Command '[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()' | tr -d '\r')
diff_ms=$(( ___ - win ))  # 【穴①】ns → ms
echo "WSL - Windows = ${diff_ms} ms（powershell の起動時間ぶん、数百msは WSL が先になる）"
