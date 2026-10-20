#!/usr/bin/env python3
"""推論サーバーを起動する（.venv-bench で実行）。docs/track2/37_policy_node_zmq.md

例: python scripts/serve_policy.py --config configs/policy/so101_ov_cpu.yaml --bind tcp://127.0.0.1:5600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evla.common.config import load_config
from evla.ipc.server import InferenceServer
from evla.policy.base import PolicyConfig
from evla.policy.factory import make_policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bind", default="tcp://127.0.0.1:5600")
    p.add_argument("--image_key_map", default='{"top": "image", "wrist": "image2"}',
                   help='ROSのカメラ名 → Policy の視点名（JSON）。SO-101 のモデルなら {"top": "top", "wrist": "wrist"}')
    a = p.parse_args()
    policy = make_policy(load_config(a.config, PolicyConfig))
    server = InferenceServer(policy, bind=a.bind, image_key_map=json.loads(a.image_key_map))  # 【穴①】JSON の文字列を辞書にする
    print(f"serving {a.config} on {a.bind}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        policy.close()


if __name__ == "__main__":
    main()
