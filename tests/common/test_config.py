from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from evla.common.config import dump_config, load_config


@dataclass(frozen=True)
class Inner:
    n_threads: int | None = None
    views: tuple[str, ...] = ("image", "image2")


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner = field(default_factory=Inner)
    path: Path = Path("data/x.npz")


def write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_nested_and_types(tmp_path):
    cfg = load_config(write(tmp_path, {"name": "a", "inner": {"views": ["image"]}, "path": "p.npz"}), Outer)
    assert cfg.inner.views == ("image",)
    assert isinstance(cfg.path, Path)


def test_override(tmp_path):
    cfg = load_config(write(tmp_path, {"name": "a"}), Outer, ["inner.n_threads=8", "name=b"])
    assert cfg.inner.n_threads == 8 and cfg.name == "b"


def test_unknown_key_is_error(tmp_path):
    with pytest.raises(ValueError, match="n_thread"):
        load_config(write(tmp_path, {"name": "a", "inner": {"n_thread": 8}}), Outer)


def test_dump_roundtrip(tmp_path):
    cfg = load_config(write(tmp_path, {"name": "a"}), Outer, ["inner.n_threads=4"])
    out = tmp_path / "out.yaml"
    dump_config(cfg, out)
    assert load_config(out, Outer) == cfg
