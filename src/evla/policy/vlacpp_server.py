"""vla-server の起動と終了。docs/track1/25_runtime_vlacpp.md 4"""

from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VlaServerConfig:
    binary: Path                       # third_party/vla.cpp/build{,-sycl,-ov}/vla-server
    gguf: Path
    bind: str = "tcp://127.0.0.1:5555"
    env: Mapping[str, str] = field(default_factory=dict)   # VLA_N_THREADS, GGML_OPENVINO_DEVICE, VLA_DEVICE ...
    args: tuple[str, ...] = ()         # ("--weight-dtype", "f16", "--num-steps", "10")
    expect_backend: str | None = None  # "CPU" | "SYCL" | "OPENVINO"。バナーと一致しなければ例外
    ready_timeout_s: float = 300.0     # OV-GPUはコンパイルで遅いので長めにする
    log_path: Path | None = None


# バナーの行から、実際に使われたバックエンドを推定する。【要確認】実際の出力を見て正規表現を直す
BACKEND_PATTERNS = {
    "SYCL": re.compile(r"backend\s*=\s*SYCL|ggml_sycl", re.I),
    "OPENVINO": re.compile(r"OpenVINO: using device", re.I),
}


def detect_backend(banner: list[str]) -> str:
    text = "\n".join(banner)
    for name, pat in BACKEND_PATTERNS.items():
        if pat.search(text):  # 【穴①】文字列のどこかに一致するか（re のメソッド）
            return name
    return "CPU"


class VlaServerProcess:
    """with文で使う。__enter__ で起動して ready まで待ち、__exit__ で SIGTERM → 待機 → SIGKILL。"""

    def __init__(self, cfg: VlaServerConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._ready = threading.Event()

    def _cmd(self) -> list[str]:
        # チェックポイントは位置引数で、オプションの後ろに置く（`vla-server --help`）
        return [str(self.cfg.binary), "--bind", self.cfg.bind, *self.cfg.args, str(self.cfg.gguf)]

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        log = open(self.cfg.log_path, "w") if self.cfg.log_path else None  # noqa: SIM115
        for line in self._proc.stdout:  # プロセスが終わるまで1行ずつ読む
            self._lines.append(line.rstrip("\n"))
            if log:
                log.write(line)
                log.flush()
            if "ready" in line:  # 【穴②】起動完了の行に含まれる語（"... bound to tcp://*:5555. ready."）
                self._ready.set()  # 【穴③】待っているスレッドに「準備完了」を知らせる（threading.Event のメソッド）
        if log:
            log.close()

    def __enter__(self) -> VlaServerProcess:
        env = {**os.environ, **self.cfg.env}
        self._proc = subprocess.Popen(
            self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env  # 【穴④】標準エラーも標準出力にまとめる指定
        )
        atexit.register(self._terminate)  # 例外で抜けてもゾンビを残さない
        threading.Thread(target=self._reader, daemon=True).start()
        deadline = time.monotonic() + self.cfg.ready_timeout_s
        while not self._ready.wait(timeout=1.0):  # 【穴⑤】イベントを待つ（来なければ False）
            if self._proc.poll() is not None:  # ready の前に落ちたら、待たずに失敗にする
                raise RuntimeError(f"vla-server が起動中に終了した（rc={self._proc.returncode}）。"
                                   f"ログ末尾: {self._lines[-20:]}")
            if time.monotonic() > deadline:
                self._terminate()
                raise TimeoutError(f"vla-server が ready にならない。ログ末尾: {self._lines[-20:]}")
        if self.cfg.expect_backend is not None:
            actual = detect_backend(self.banner)
            if actual != self.cfg.expect_backend:
                self._terminate()
                raise RuntimeError(f"バックエンドが違う: 期待 {self.cfg.expect_backend}、実際 {actual}")
        return self

    def __exit__(self, *exc: object) -> None:
        self._terminate()

    def _terminate(self) -> None:
        p = self._proc
        if p is None or p.poll() is not None:
            return
        p.send_signal(signal.SIGTERM)  # 【穴⑥】まずは行儀よく終了を頼むシグナル
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()  # 【穴⑦】応じなければ強制終了（SIGKILL）
            p.wait()

    @property
    def pid(self) -> int:
        assert self._proc is not None
        return self._proc.pid

    @property
    def banner(self) -> list[str]:
        return list(self._lines[:200])
