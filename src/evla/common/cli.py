"""scripts/ で共通に使う引数の定義。scripts は薄く保ち、中身は src/evla を呼ぶだけにする。"""

from __future__ import annotations

import argparse
from pathlib import Path


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True, help="設定YAML")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",  # 【穴①】同じオプションを何回も書けて、値がリストにたまる指定
                   help="設定の上書き（例 --set policy.n_threads=8）。何回でも書ける")
    p.add_argument("--tag", default="", help="run の目印（掃引名など）")
    return p
