"""環境スナップショット（env.json）。docs/01_repository_design.md 4.3"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "torch", "openvino", "lerobot", "physicalai", "pyzmq", "nncf", "transformers")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VLA_N_THREADS",
              "MUJOCO_GL", "GGML_OPENVINO_DEVICE")


@dataclass(frozen=True)
class EnvSnapshot:
    git_commit: str            # `git rev-parse HEAD`。未コミットの変更があれば末尾に "-dirty"
    hostname: str
    cpu_model: str             # /proc/cpuinfo の model name
    n_logical_cpus: int
    cpu_affinity: list[int]    # os.sched_getaffinity(0)
    mem_total_gb: float
    python: str
    packages: dict[str, str]   # torch, openvino, lerobot, numpy などの版（importlib.metadata）
    thread_env: dict[str, str] # OMP_NUM_THREADS, MKL_NUM_THREADS, VLA_N_THREADS など
    ov_devices: list[str]      # openvino.Core().available_devices（入っていれば）
    timestamp_utc: str


def _git_commit() -> str:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")  # 【穴①】現在のコミットIDを表示する git のサブコマンドと引数
        dirty = run("status", "--porcelain") != ""  # 【穴②】変更ファイルを機械向けの形式で一覧する git のサブコマンドと引数
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return str(line.split(": ", 1)[-1])  # 【穴③】"model name	: 13th Gen ..." のコロンより後ろ
    except OSError:
        pass
    return platform.processor()


def _mem_total_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return round(kib / (1024**2), 2)  # 【穴④】KiB → GiB
    return float("nan")


def _packages() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)  # 【穴⑤】import せずにインストール済みの版を得る（importlib.metadata）
        except metadata.PackageNotFoundError:
            continue
    return out


def _ov_devices() -> list[str]:
    try:
        import openvino as ov  # 入っていない環境（ROS側）もあるので関数の中で import する
    except ImportError:
        return []
    return list(ov.Core().available_devices)


def capture_env() -> EnvSnapshot:
    return EnvSnapshot(
        git_commit=_git_commit(),
        hostname=socket.gethostname(),
        cpu_model=_cpu_model(),
        n_logical_cpus=os.cpu_count() or 0,
        cpu_affinity=sorted(os.sched_getaffinity(0)),  # 【穴⑥】このプロセスが動いてよいCPU番号の集合（os モジュール）
        mem_total_gb=_mem_total_gb(),
        python=platform.python_version(),
        packages=_packages(),
        thread_env={k: os.environ[k] for k in THREAD_ENV if k in os.environ},
        ov_devices=_ov_devices(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_env(path: Path) -> EnvSnapshot:
    snap = capture_env()
    path.write_text(json.dumps(dataclasses.asdict(snap), indent=2, ensure_ascii=False))  # 【穴⑦】dataclass を辞書に変換する関数
    return snap
