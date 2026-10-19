import math

import numpy as np
import pandas as pd

from evla.analysis.extrapolate import interpolate_success
from evla.analysis.rtf import rtf_series

CURVE = pd.DataFrame({"latency_ms": [0, 500, 1000], "n_exec": 10, "mode": "async",
                      "success_rate": [0.9, 0.7, 0.3], "ci_lo": [0.8, 0.6, 0.2], "ci_hi": [1.0, 0.8, 0.4]})


def test_interpolate_inside():
    s, lo, hi = interpolate_success(CURVE, 250, 10, "async")
    assert math.isclose(s, 0.8) and math.isclose(lo, 0.7) and math.isclose(hi, 0.9)


def test_outside_is_nan():
    assert math.isnan(interpolate_success(CURVE, 2000, 10, "async")[0])
    assert math.isnan(interpolate_success(CURVE, 250, 25, "async")[0])


def test_rtf_half_speed():
    t_wall = np.arange(0, 20, 0.1)
    t_sim = t_wall * 0.5
    r = rtf_series(t_sim, (t_wall * 1e9).astype(np.int64), window_s=5.0)
    assert np.isnan(r[0])
    assert np.allclose(r[60:], 0.5)
