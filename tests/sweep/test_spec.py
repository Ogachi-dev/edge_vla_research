import yaml

from evla.sweep.spec import expand, policy_key


def write(tmp_path, d):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_expand_and_exclude(tmp_path):
    spec = expand(write(tmp_path, {
        "name": "s", "mode": "A", "base": "configs/bench/base.yaml", "rounds": 3,
        "grid": [
            {"policy.runtime": ["torch"], "policy.precision": ["fp32", "bf16"]},
            {"policy.runtime": ["ov"], "policy.device": ["CPU", "GPU"], "policy.n_threads": [8, 16]},
        ],
        "exclude": [{"policy.device": "GPU", "policy.n_threads": 16}],
    }))
    assert len(spec.items) == 2 + 4 - 1
    assert spec.rounds == 3
    assert len({it.key for it in spec.items}) == len(spec.items)


def test_policy_key_ignores_non_policy():
    a = {"policy.runtime": "ov", "n_measure": 50}
    b = {"policy.runtime": "ov", "n_measure": 200}
    assert policy_key(a) == policy_key(b)
    assert policy_key(a) != policy_key({"policy.runtime": "torch"})
