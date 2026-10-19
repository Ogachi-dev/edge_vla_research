import pytest

from evla.analysis.stats import mcnemar, required_n, wilson_ci


def test_wilson_known_value():
    lo, hi = wilson_ci(8, 10)   # statsmodels の proportion_confint(8, 10, method="wilson") と同じ値
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_edges_stay_in_range():
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
    assert 0 < wilson_ci(10, 10)[0] < 1


def test_mcnemar_counts_and_p():
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 8 + [False] * 2
    r = mcnemar(a, b)
    assert r["b01"] == 8 and r["b10"] == 0
    assert r["p_value"] == pytest.approx(2 * 0.5**8)


def test_mcnemar_no_discordant():
    assert mcnemar([True, False], [True, False])["p_value"] == 1.0


def test_required_n_grows_as_gap_shrinks():
    assert required_n(0.8, 0.6) < required_n(0.8, 0.7)
    assert required_n(0.8, 0.6) == pytest.approx(82, abs=2)
