import math
import time

import numpy as np
import pytest

from evla.bench.memory import PeakRssMonitor
from evla.bench.timing import Stopwatch, summarize_latencies


def test_summary_values():
    s = summarize_latencies(np.arange(1, 101))
    assert s["n"] == 100 and s["min"] == 1 and s["max"] == 100
    assert s["p50"] == pytest.approx(50.5)
    assert not math.isnan(s["p99"])


def test_p99_nan_when_few_samples():
    assert math.isnan(summarize_latencies([1.0, 2.0, 3.0])["p99"])


def test_stopwatch_sleep():
    with Stopwatch() as sw:
        time.sleep(0.05)
    assert 45 < sw.ms < 150


def test_peak_rss_sees_allocation():
    import os
    with PeakRssMonitor([os.getpid()], interval_s=0.01) as m:
        before = m.peak_mb[os.getpid()]
        x = np.ones(50_000_000 // 8)   # 50MB
        time.sleep(0.1)
        del x
    assert m.peak_mb[os.getpid()] > before + 30
