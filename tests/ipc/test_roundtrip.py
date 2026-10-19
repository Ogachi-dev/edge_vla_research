import threading

import numpy as np
import pytest

from evla.ipc.client import InferenceClient
from evla.ipc.server import InferenceServer
from evla.policy.base import ChunkResult, Policy, PolicyConfig


class EchoPolicy(Policy):
    """state を全行にコピーしたチャンクを返す。"""

    def __init__(self):
        self.cfg = PolicyConfig(runtime="torch", model="echo")
        self.last_images = None

    chunk_size = 5
    action_dim = 6

    def predict_chunk(self, obs, noise=None):
        self.last_images = sorted(obs.images)
        return ChunkResult(actions=np.tile(obs.state, (5, 1)).astype(np.float32), latency_ms=1.0)


@pytest.fixture
def server():
    addr = "tcp://127.0.0.1:5699"
    policy = EchoPolicy()
    srv = InferenceServer(policy, bind=addr, image_key_map={"top": "top", "wrist": "wrist"})
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield addr, policy
    srv.stop()


def test_predict_roundtrip(server):
    addr, policy = server
    c = InferenceClient(addr, timeout_ms=2000)
    assert c.ping()
    img = np.zeros((48, 64, 3), np.uint8)
    rep = c.predict({"top": img, "wrist": img}, np.arange(6, dtype=np.float32), "pick", t_obs_sim=1.0)
    assert rep.actions.shape == (5, 6)
    assert np.array_equal(rep.actions[3], np.arange(6))
    assert policy.last_images == ["top", "wrist"]
    assert rep.rtt_ms > 0
    c.close()


def test_timeout_when_no_server():
    c = InferenceClient("tcp://127.0.0.1:5698", timeout_ms=100, retries=1)
    with pytest.raises(TimeoutError):
        c.predict({}, np.zeros(6, np.float32), "x", 0.0)
    c.close()
