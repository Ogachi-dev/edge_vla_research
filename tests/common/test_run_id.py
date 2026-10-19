import re

import pytest

from evla.common.run_id import config_hash, make_run_dir, new_run_id


def test_hash_is_order_independent():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash({"b": {"d": 3, "c": 2}, "a": 1})


def test_hash_changes_with_value():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_hash_length():
    assert len(config_hash({"a": 1})) == 6
    assert len(config_hash({"a": 1}, n=10)) == 10


def test_run_id_format():
    rid = new_run_id("t1", "modeA", {"x": 1})
    assert re.fullmatch(r"\d{8}-\d{6}_t1_modeA_[0-9a-f]{6}", rid), rid


def test_make_run_dir_refuses_overwrite(tmp_path):
    d = make_run_dir("r1", root=tmp_path)
    assert d.is_dir() and (d / "artifacts").is_dir()
    with pytest.raises(FileExistsError):
        make_run_dir("r1", root=tmp_path)
