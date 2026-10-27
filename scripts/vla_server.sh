#!/usr/bin/env bash
# vla-server をバックエンドの環境を読んでから起動する（docs/env/12_inference_envs.md Step 4〜5）。
# 使い方: VLA_BACKEND={cpu|sycl|ov} scripts/vla_server.sh <vla-server の引数...>
# 設定では server.binary にこのスクリプト、server.env.VLA_BACKEND にバックエンドを書く
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# setupvars.sh を引数なしで source すると、このスクリプトの引数を読んで消費するので、先に退避しておく
ARGS=("$@")
set --
case "${VLA_BACKEND:-cpu}" in
  cpu)  BIN="$REPO/third_party/vla.cpp/build/vla-server" ;;
  sycl) set +u; source "$HOME/intel/oneapi/setvars.sh" --force >/dev/null; set -u
        BIN="$REPO/third_party/vla.cpp/build-sycl/vla-server" ;;
  ov)   set +u; source /opt/intel/openvino/setupvars.sh >/dev/null; set -u
        export GGML_OPENVINO_DEVICE="${GGML_OPENVINO_DEVICE:-GPU}"
        BIN="$REPO/third_party/vla.cpp/build-ov/vla-server" ;;
  *) echo "unknown VLA_BACKEND=${VLA_BACKEND}" >&2; exit 2 ;;
esac
exec "$BIN" "${ARGS[@]}"
