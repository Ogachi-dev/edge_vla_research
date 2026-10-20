"""ランタイム間で共通の前処理。docs/track1/22_policy_interface.md、25_runtime_vlacpp.md 2.1"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from evla.sim.types import Observation


def select_views(obs: Observation, views: tuple[str, ...]) -> dict[str, npt.NDArray[np.uint8]]:
    """cfg.views にある視点だけを、その順で取り出す。無い視点を要求したら KeyError。"""
    missing = [v for v in views if v not in obs.images]
    if missing:
        raise KeyError(f"観測に無い視点: {missing}（ある視点: {list(obs.images)}）")
    return {v: obs.images[v] for v in views}


def hwc_uint8_to_chw_float(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """(H, W, 3) uint8 [0,255] → (3, H, W) float32 [0,1]"""
    x = img.astype(np.float32) / 255.0  # 【穴①】[0,255] → [0,1]
    return np.ascontiguousarray(x.transpose(2, 0, 1))  # 【穴②】HWC → CHW の軸の並べ替え


def resize_with_pad(img_chw: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float32]:
    """アスペクト比を保って長辺を size にし、左と上を0で埋めて size×size にする。
    【要確認】LeRobot の SmolVLA（modeling_smolvla.py の resize_with_pad）は「左と上」を埋める実装だった。
    使う版のソースと、vla.cpp のクライアントの実装を読んで、埋める側が同じか確かめる。"""
    import torch
    import torch.nn.functional as F  # noqa: N812

    c, h, w = img_chw.shape
    ratio = max(h, w) / size  # 【穴③】縮小率（長辺が size になる）
    nh, nw = int(h / ratio), int(w / ratio)
    t = torch.from_numpy(img_chw)[None]
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)  # 【穴④】出力の (高さ, 幅)
    t = F.pad(t, (size - nw, 0, size - nh, 0), value=0.0)  # 【穴⑤】(左, 右, 上, 下) の埋め幅。左と上を埋める
    return t[0].numpy()
