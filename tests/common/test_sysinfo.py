import json

from evla.common.sysinfo import capture_env, save_env


def test_capture_env_basic():
    s = capture_env()
    assert s.n_logical_cpus > 0
    assert len(s.cpu_affinity) > 0
    assert s.mem_total_gb > 0
    assert "numpy" in s.packages


def test_save_env(tmp_path):
    save_env(tmp_path / "env.json")
    d = json.loads((tmp_path / "env.json").read_text())
    assert "git_commit" in d and "cpu_model" in d
