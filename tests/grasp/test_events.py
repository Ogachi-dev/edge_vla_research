import json

import numpy as np

from evla.grasp.events import classify_failure, dumps, make_event


def cand(*failed):
    return {"type": "grasp_candidate", "decision": {"ok": not failed, "failed": list(failed)}}


def test_dumps_numpy():
    s = dumps(make_event("released", 1.0, cube_pos=np.array([1.0, 2.0]), n=np.int64(3)))
    assert json.loads(s)["cube_pos"] == [1.0, 2.0]


def test_classify():
    assert classify_failure([]) == "no_attempt"
    assert classify_failure([cand("yaw"), cand("yaw", "forward")]) == "grasp_rejected:yaw"
    assert classify_failure([cand(), {"type": "attached"}, {"type": "failure"}]) == "dropped_or_misplaced"
    assert classify_failure([cand(), {"type": "attached"}]) == "timeout_after_release"
    assert classify_failure([{"type": "success"}]) == "ok"
