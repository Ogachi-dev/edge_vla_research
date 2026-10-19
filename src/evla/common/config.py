"""dataclass＋YAML の設定。docs/01_repository_design.md 4.4

使い方:
    cfg = load_config(Path("configs/bench/base.yaml"), ModeAConfig, ["policy.n_threads=8"])
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def parse_override(item: str) -> tuple[list[str], Any]:
    """'a.b=3' → (['a', 'b'], 3)。値は YAML として解釈する（3 → int、[1,2] → list、null → None）。"""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"override は key=value 形式で書く: {item!r}")
    return key.split("."), yaml.safe_load(raw)  # 【穴①】ドットで分割したキーのリスト、YAMLとして解釈した値


def set_by_path(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """d["a"]["b"] = value。途中の辞書がなければ作る。"""
    for k in keys[:-1]:
        d = d.setdefault(k, {})  # 【穴②】無ければ空の辞書を入れて、それを返す dict のメソッド
    d[keys[-1]] = value


def _convert(tp: Any, value: Any, where: str) -> Any:
    """型ヒント tp に合わせて value を変換する（dataclass・Path・tuple・Optional に対応）。"""
    if value is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):  # int | None など
        non_none = [a for a in args if a is not type(None)]
        return _convert(non_none[0], value, where) if len(non_none) == 1 else value
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, where)
    if tp is Path:
        return Path(value)
    if origin is tuple:
        elem = args[0] if args else Any
        return tuple(_convert(elem, v, where) for v in value)
    return value


def from_dict[T](cls: type[T], data: Mapping[str, Any], where: str = "") -> T:
    """辞書 → dataclass。未知のキーがあれば ValueError（タイポの早期検出）。"""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # 【穴③】dataclass のフィールド一覧を返す関数
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}{where}: 未知のキー {sorted(unknown)}")  # 【穴④】送出する例外の型
    kwargs = {k: _convert(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def load_config[T](path: Path, cls: type[T], overrides: list[str] | None = None) -> T:
    """YAMLを読み、`a.b=3` 形式のoverridesを適用して、dataclass cls に変換する。
    未知のキーがあれば ValueError（タイポの早期検出）。"""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}  # 【穴⑤】YAMLを読む（任意のPythonオブジェクトを作らない安全な方）
    for item in overrides or []:
        keys, value = parse_override(item)
        set_by_path(data, keys, value)
    return from_dict(cls, data)


def to_plain(obj: Any) -> Any:
    """dataclass / Path / tuple を、YAMLやJSONにできる素の型に変換する。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return  str(obj) # 【穴⑥】Path を文字列に
    return obj


def dump_config(cfg: Any, path: Path) -> None:
    """dataclassをYAMLに書き出す（デフォルト値も含めて全部）。"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False, allow_unicode=True)
