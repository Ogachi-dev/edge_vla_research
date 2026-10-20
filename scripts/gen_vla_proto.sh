#!/usr/bin/env bash
# vla.cpp の vla.proto から Python のモジュール（vla_pb2.py）を生成する。docs/track1/25_runtime_vlacpp.md
# 使い方: .venv-bench を有効にして  bash scripts/gen_vla_proto.sh
set -euo pipefail
PROTO_DIR=third_party/vla.cpp/src/serving
OUT=src/evla/policy/_proto
# grpcio-tools に入っている protoc を使う（apt の protoc と protobuf ランタイムの版がずれる事故を避ける）
python -m grpc_tools.protoc -I "$PROTO_DIR" --python_out="$OUT" "$PROTO_DIR/vla.proto"  # 【穴①】Python のコードを出力するオプション
echo "generated: $OUT/vla_pb2.py"
