import numpy as np
import pytest

from evla.dataset.conventions import (
    joint_state_to_vector,
    lerobot_to_joint_targets,
    rad_vector_to_lerobot,
)


def test_reorders_by_name():
    names = ["6", "1", "3", "2", "5", "4"]
    pos = [0.6, 0.1, 0.3, 0.2, 0.5, 0.4]
    assert np.allclose(joint_state_to_vector(names, pos), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_missing_joint():
    with pytest.raises(KeyError):
        joint_state_to_vector(["1", "2"], [0.0, 0.0])


def test_roundtrip_rad_lerobot_rad():
    q = np.radians([10, -20, 30, -40, 50, 60])
    v = rad_vector_to_lerobot(q)
    assert v.dtype == np.float32
    back = lerobot_to_joint_targets(v)
    assert np.allclose([back[k] for k in "123456"], q, atol=1e-5)


def test_batch_shape():
    assert rad_vector_to_lerobot(np.zeros((7, 6))).shape == (7, 6)


def test_chunk_to_urdf():
    from evla.dataset.conventions import chunk_to_urdf

    chunk = np.tile(rad_vector_to_lerobot(np.radians([10, 20, 30, 40, 50, 60])), (4, 1))
    d = chunk_to_urdf(chunk)
    assert set(d) == set("123456")
    assert d["3"].shape == (4,)
    assert np.allclose(d["3"], np.radians(30), atol=1e-5)
