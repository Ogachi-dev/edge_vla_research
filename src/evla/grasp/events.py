"""把持イベント（/grasp/event に JSON で出し、JSONL にも保存する）。docs/track2/32_grasp_judge_node.md 4"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

EVENT_TYPES = ("grasp_candidate", "attach_sent", "attached", "detach_sent", "released", "success", "failure")


def make_event(type_: str, t_sim: float, **payload: Any) -> dict[str, Any]:
    if type_ not in EVENT_TYPES:
        raise ValueError(f"未知のイベント: {type_}")
    return {"type": type_, "t_sim": float(t_sim), **payload}


def _default(o: Any) -> Any:
    """json.dumps が知らない型の変換。"""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, np.ndarray):
        return o.tolist()  # 【穴①】ndarray を Python のリストに
    if isinstance(o, np.generic):
        return o.item()  # np.float64 → float など
    raise TypeError(f"JSONにできない型: {type(o)}")


def dumps(event: dict[str, Any]) -> str:
    return json.dumps(event, default=_default, ensure_ascii=False)  # 【穴②】知らない型に出会ったときに呼ぶ関数を指定する引数


def classify_failure(events: list[dict[str, Any]]) -> str:
    """失敗の分類（docs/track2/38_gazebo_eval.md 4.2）。成功なら "ok"。"""
    types = [e["type"] for e in events]
    if "success" in types:
        return "ok"
    if "grasp_candidate" not in types:
        return "no_attempt"
    if "attached" not in types:
        failed = [f for e in events if e["type"] == "grasp_candidate" for f in e["decision"]["failed"]]
        most = max(set(failed), key=failed.count) if failed else "?"  # 【穴③】最も多く出てきた条件の名前（出現回数が最大の要素）
        return f"grasp_rejected:{most}"
    if "failure" in types:
        return "dropped_or_misplaced"
    return "timeout_after_release"
