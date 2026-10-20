"""実験ID（run_id）と結果ディレクトリ。docs/01_repository_design.md 4.2"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def config_hash(cfg: Mapping[str, Any], n: int = 6) -> str:
    """設定辞書から決定的なハッシュを作る（キー順に依存しない）。"""
    # Path などJSONにできない値は default=str で文字列にする
    text = json.dumps(cfg, sort_keys=True, default=str)  # 【穴①】キー順に依存しないJSONにする json.dumps の引数
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()  # 【穴②】ハッシュ関数（hashlib モジュール、160bit）
    return digest[0:n]  # 【穴③】先頭 n 文字


def new_run_id(track: str, mode: str, cfg: Mapping[str, Any]) -> str:
    """'20261014-103012_t1_modeA_3fa9c1' 形式のIDを返す。"""
    stamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')  # 【穴④】20261014-103012 になる書式文字列
    return f"{stamp}_{track}_{mode}_{config_hash(cfg)}"


def make_run_dir(run_id: str, root: Path = Path("results/runs")) -> Path:
    """ディレクトリを作成して返す。既存なら FileExistsError（上書き事故防止）。"""
    d = root / run_id
    d.mkdir(parents=True, exist_ok=False)  # 【穴⑤】既存なら FileExistsError にする引数
    (d / "artifacts").mkdir()
    return d
