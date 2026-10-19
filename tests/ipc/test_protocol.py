import numpy as np
import pytest

from evla.ipc.protocol import decode, encode


def roundtrip(arrays, header=None):
    frames = encode(header or {"type": "x"}, arrays)
    frames = [bytes(f) for f in frames]   # 実際の送受信と同じく bytes にする
    return decode(frames)


def test_roundtrip_mixed_dtypes():
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    st = np.arange(6, dtype=np.float32)
    head, arr = roundtrip({"top": img, "state": st}, {"type": "predict", "task": "pick"})
    assert head["task"] == "pick"
    assert arr["top"].dtype == np.uint8 and np.array_equal(arr["top"], img)
    assert np.array_equal(arr["state"], st)


def test_noncontiguous_array():
    a = np.arange(12, dtype=np.float32).reshape(3, 4).T   # 転置で非連続
    _, arr = roundtrip({"a": a})
    assert np.array_equal(arr["a"], a)


def test_empty_array_and_no_arrays():
    _, arr = roundtrip({"e": np.zeros((0, 6), np.float32)})
    assert arr["e"].shape == (0, 6)
    head, arr = roundtrip({})
    assert arr == {}


def test_version_mismatch():
    import msgpack
    with pytest.raises(ValueError):
        decode([msgpack.packb({"v": 999, "arrays": []})])
