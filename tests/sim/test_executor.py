import numpy as np

from evla.sim.executor import AsyncExecutor, ExecutorConfig, SyncExecutor
from evla.sim.latency import ConstantLatency

from .fakes import OBS, FakeEnv, FakePolicy


def run(cls, mode, latency_ms, max_steps=40, **kw):
    cfg = ExecutorConfig(mode=mode, noise_dim=None, **kw)
    env = FakeEnv()
    policy = FakePolicy()
    ex = cls(cfg, policy, ConstantLatency(latency_ms), np.random.default_rng(0))
    ok, trace = ex.run(env, OBS, max_steps)
    return env, policy, trace


def test_lockstep_executes_chunks_back_to_back():
    env, _, trace = run(SyncExecutor, "lockstep", 999.0, max_steps=25)
    assert trace.hold_steps == 0
    expected = [(0, j) for j in range(10)] + [(1, j) for j in range(10)] + [(2, j) for j in range(5)]
    assert env.tags() == expected


def test_sync_inserts_k_holds_before_each_chunk():
    env, _, trace = run(SyncExecutor, "sync", 150.0, max_steps=26)  # k = 3
    expected = [None] * 3 + [(0, j) for j in range(10)] + [None] * 3 + [(1, j) for j in range(10)]
    assert env.tags() == expected
    assert trace.hold_steps == 6


def test_sync_uses_observation_at_inference_start():
    _, policy, _ = run(SyncExecutor, "sync", 150.0, max_steps=26)
    assert policy.seen_steps == [0, 13]


def test_async_no_hold_after_first_and_discards_stale():
    env, _, trace = run(AsyncExecutor, "async", 150.0, max_steps=40, async_threshold=5)
    tags = env.tags()
    assert tags[:3] == [None] * 3                       # 最初の1回だけ同期
    assert None not in tags[3:]                         # その後は保持なし
    assert tags[3:14] == [(0, j) for j in range(8)] + [(1, 3), (1, 4), (1, 5)]
    assert trace.discarded_actions >= 3


def test_async_queue_starves_when_latency_is_large():
    env, _, trace = run(AsyncExecutor, "async", 2000.0, max_steps=60, async_threshold=5)  # k = 40
    assert trace.hold_steps > 40
    assert None in env.tags()[50:]


def test_episode_ends_on_done():
    cfg = ExecutorConfig(mode="lockstep", noise_dim=None)
    env = FakeEnv(done_at=7, success=True)
    ok, trace = SyncExecutor(cfg, FakePolicy(), ConstantLatency(0), np.random.default_rng(0)).run(env, OBS, 100)
    assert ok and trace.steps == 7
