"""モードA：推論単体のベンチ。1設定×1プロセス。docs/track1/26_bench_mode_a.md"""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import psutil

from evla.bench.memory import PeakRssMonitor
from evla.bench.timing import Stopwatch, summarize_latencies
from evla.common.config import dump_config
from evla.common.sysinfo import save_env
from evla.policy.base import PolicyConfig
from evla.policy.equivalence import make_fixed_noise
from evla.policy.factory import make_policy
from evla.policy.vlacpp_server import VlaServerConfig, VlaServerProcess
from evla.sim.fixed_obs import load_fixed_obs

MAX_ACTION_DIM = 32


@dataclass(frozen=True)
class ModeAConfig:
    policy: PolicyConfig
    fixed_obs: Path = Path("data/fixed_obs/libero_object_t0.npz")
    n_warmup: int = 3
    n_measure: int = 50
    obs_cycle: bool = True              # 固定観測を順番に使い回す（毎回同じ1枚にしない）
    fixed_noise_seed: int | None = 0
    server: VlaServerConfig | None = None   # runtime == "vlacpp" のときに起動する
    cooldown_s: float = 0.0             # 測定前の待機（温度を揃える）
    tag: str = ""
    chunk_size: int = 50


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    load_s: float                  # モデルの読み込み（GPUならコンパイルを含む）
    first_infer_ms: float          # ウォームアップの1回目
    summary: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float = float("nan")
    peak_rss_server_mb: float | None = None


def run_mode_a(cfg: ModeAConfig, run_dir: Path) -> RunResult:
    run_id = run_dir.name
    dump_config(cfg, run_dir / "config.yaml")
    save_env(run_dir / "env.json")
    obs_list = load_fixed_obs(cfg.fixed_obs)
    noise = None
    if cfg.fixed_noise_seed is not None:
        noise = make_fixed_noise(cfg.chunk_size, MAX_ACTION_DIM, cfg.fixed_noise_seed)

    rows: list[dict] = []
    with ExitStack() as stack:  # 後片付けを登録した逆順で確実に行う
        server = None
        if cfg.policy.runtime == "vlacpp":
            if cfg.server is None:
                raise ValueError("runtime=vlacpp には server の設定が必要")
            server = stack.enter_context(VlaServerProcess(cfg.server))  # 【穴①】with 文に入るのと同じことをして、抜けるときの後片付けを登録する ExitStack のメソッド
        pids = [os.getpid()] + ([server.pid] if server else [])
        mon = stack.enter_context(PeakRssMonitor(pids))

        with Stopwatch() as sw_load:
            policy = stack.enter_context(make_policy(cfg.policy))
        time.sleep(cfg.cooldown_s)

        n_total = cfg.n_warmup + cfg.n_measure
        first_ms = float("nan")
        for i in range(n_total):
            k = i % len(obs_list) if cfg.obs_cycle else 0  # 【穴②】観測を順番に使い回す番号
            t_start_ns = time.time_ns()  # Windows側のログとの突き合わせ用（壁時計）
            vcpu = psutil.Process().cpu_num()  # os.sched_getcpu は Python 3.12 にない
            r = policy.predict_chunk(obs_list[k], noise)
            if i == 0:
                first_ms = r.latency_ms
            row = {
                "run_id": run_id,
                "iter": i - cfg.n_warmup,  # 【穴③】ウォームアップが負の番号、本番が0から
                "is_warmup": i < cfg.n_warmup,
                "obs_index": k,
                "latency_ms": r.latency_ms,
                "t_start_ns": t_start_ns,
                "vcpu": vcpu,
            }
            row.update({f"bd_{name}_ms": v for name, v in r.breakdown_ms.items()})
            rows.append(row)
        describe = policy.describe()

    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "samples.parquet", index=False)  # 【穴④】列指向の Parquet 形式で保存する DataFrame のメソッド
    measured = df.loc[~df["is_warmup"], "latency_ms"]  # 【穴⑤】ウォームアップでない行（真偽の反転）
    result = RunResult(
        run_dir=run_dir,
        load_s=sw_load.ms / 1000,
        first_infer_ms=first_ms,
        summary=summarize_latencies(measured.to_numpy()),
        peak_rss_mb=mon.peak_mb[os.getpid()],
        peak_rss_server_mb=mon.peak_mb.get(server.pid) if server else None,
    )
    out = {**asdict(result), "run_dir": str(run_dir), "describe": describe,
           "server_banner": server.banner if server else None}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return result


