"""推論サーバーとの通信プロトコル（ZeroMQ マルチパート＋msgpack）。docs/track2/37_policy_node_zmq.md 4.1

フレーム0 = msgpack のヘッダ、フレーム1.. = 配列の生のバイト列（ヘッダの "arrays" の順）。
ROS のプロセス（NumPy 1.26）からも import するので、NumPy 2 専用の API は使わない。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import msgpack
import numpy as np

from evla._blank import ___  # 穴埋め用。全部埋めたらこの行は消してよい

PROTOCOL_VERSION = 1


def encode(header: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> list[bytes | memoryview]:
    """ヘッダに配列のメタ情報を足して msgpack 化し、[header, buf1, buf2, ...] を返す。
    配列は C 連続にしてから memoryview で渡す（コピーを避ける）。"""
    metas, bufs = [], []
    for name, arr in arrays.items():
        a = ___  # 【穴①】メモリ上で連続な配列にする（転置などで非連続だと、生のバイト列の並びが形と合わない）
        metas.append({"name": name, "dtype": ___, "shape": list(a.shape)})  # 【穴②】エンディアンも含む型の文字列（'<f4' など）
        # 要素数0の配列は memoryview の cast ができないので空のバイト列にする
        bufs.append(___.cast("B") if a.size else b"")  # 【穴③】コピーせずにバッファを参照するオブジェクト
    head = {**header, "v": PROTOCOL_VERSION, "arrays": metas}
    return [___(head, use_bin_type=True), *bufs]  # 【穴④】辞書を msgpack のバイト列にする関数


def decode(frames: Sequence[Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """np.frombuffer(...).reshape(shape)。v が違えば ValueError。"""
    head = ___(_bytes(frames[0]), raw=False)  # 【穴⑤】msgpack のバイト列を辞書に戻す関数
    if head.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"プロトコルの版が違う: {head.get('v')} != {PROTOCOL_VERSION}")
    metas = head.pop("arrays", [])
    if len(metas) != len(frames) - 1:
        raise ValueError(f"配列の数が合わない: header {len(metas)}, frames {len(frames) - 1}")
    arrays = {}
    for meta, fr in zip(metas, frames[1:], strict=True):
        arr = ___(_bytes(fr), dtype=np.dtype(meta["dtype"]))  # 【穴⑥】バイト列をコピーせずに配列として見る関数
        arrays[meta["name"]] = arr.reshape(meta["shape"])
    return head, arrays


def _bytes(fr: Any) -> bytes | memoryview:
    """zmq.Frame（copy=False で受けたとき）なら .buffer を、bytes ならそのまま返す。"""
    return fr.buffer if hasattr(fr, "buffer") else fr
