import pandas as pd

from evla.analysis.pareto import pareto_front


def test_pareto_simple():
    df = pd.DataFrame({
        "name": ["a", "b", "c", "d", "e"],
        "lat": [100, 200, 150, 300, 200],
        "succ": [0.5, 0.8, 0.4, 0.9, 0.7],
    })
    front = pareto_front(df, minimize="lat", maximize="succ")
    assert list(front["name"]) == ["a", "b", "d"]


def test_pareto_ties():
    df = pd.DataFrame({"lat": [100, 100], "succ": [0.5, 0.6]})
    assert list(pareto_front(df, "lat", "succ")["succ"]) == [0.6]
