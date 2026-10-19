import math

import numpy as np
import pytest

from evla.grasp.geometry import fold_deg
from evla.planning.ik_so101 import So101Geometry, fk_tcp, ik_top_down

# 仮の寸法（URDF から読んだ値に置き換えたら、このテストもそのまま使える）
GEO = So101Geometry(base_height_m=0.12, shoulder_offset_m=0.03, upper_arm_m=0.116,
                    forearm_m=0.135, wrist_to_tcp_m=0.10)


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_ik_fk_roundtrip(elbow):
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(200):
        p = np.array([rng.uniform(0.12, 0.30), rng.uniform(-0.15, 0.15), rng.uniform(0.0, 0.10)])
        yaw = rng.uniform(-math.pi, math.pi)
        q = ik_top_down(p, yaw, GEO, elbow=elbow)
        if q is None:
            continue
        n_ok += 1
        x, y, z, phi, yaw_fk = fk_tcp(q, GEO)
        assert np.allclose([x, y, z], p, atol=1e-9)
        assert phi == pytest.approx(-math.pi / 2)
        assert fold_deg(math.degrees(yaw_fk - yaw)) == pytest.approx(0.0, abs=1e-6)  # 90°の倍数のずれは許す
    assert n_ok > 100


def test_unreachable_returns_none():
    assert ik_top_down([1.0, 0.0, 0.0], 0.0, GEO) is None


def test_joint_limits_respected():
    geo = So101Geometry(**{**GEO.__dict__, "lower_rad": (-0.1,) * 5, "upper_rad": (0.1,) * 5})
    assert ik_top_down([0.2, 0.1, 0.02], 0.0, geo) is None
