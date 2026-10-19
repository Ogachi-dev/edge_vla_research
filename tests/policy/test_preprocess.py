import numpy as np
import pytest

from evla.policy.preprocess import hwc_uint8_to_chw_float, resize_with_pad, select_views
from evla.sim.types import Observation


def test_hwc_to_chw():
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 255
    x = hwc_uint8_to_chw_float(img)
    assert x.shape == (3, 4, 6) and x.dtype == np.float32
    assert x[0].max() == 1.0 and x[1].max() == 0.0
    assert x.flags["C_CONTIGUOUS"]


def test_select_views_order_and_missing():
    obs = Observation(images={"image": np.zeros((2, 2, 3), np.uint8), "image2": np.ones((2, 2, 3), np.uint8)},
                      state=np.zeros(8, np.float32), task="t")
    assert list(select_views(obs, ("image2", "image"))) == ["image2", "image"]
    with pytest.raises(KeyError):
        select_views(obs, ("wrist",))


def test_resize_with_pad_keeps_aspect():
    pytest.importorskip("torch")
    x = np.ones((3, 480, 640), np.float32)
    y = resize_with_pad(x, 512)
    assert y.shape == (3, 512, 512)
    assert y[:, :128, :].max() == 0.0     # 上側に埋め（512 - 384 = 128 行）
    assert y[:, 128:, :].min() > 0.99
