"""掃引の実行：子プロセスで1設定ずつ。再開・冷却・シャッフル対応。docs/track1/28_sweep_and_pareto.md 4.1"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

from evla.sweep.spec import SweepItem, expand

SCRIPTS = {"A": "scripts/bench_mode_a.py", "B": "scripts/eval_closed_loop.py"}


def _done_marker(out_root: Path, spec_name: str, item: SweepItem, rnd: int) -> Path:
    return out_root / spec_name / f"{item.key}_r{rnd}.done"


def run_sweep(spec_path: Path, resume: bool = True, dry_run: bool = False,
              out_root: Path = Path("results/sweeps"), sec_per_item: float = 300.0) -> None:
    """各周で順序をシャッフルし、各設定を子プロセスとして実行する。
    resume=True なら、同じ key・同じ周の完了マーカーがある設定を飛ばす。
    子プロセスが失敗したら記録して次へ進む（一晩の実行が1つの失敗で止まらないように）。"""
    spec = expand(spec_path)
    total = len(spec.items) * spec.rounds
    print(f"{spec.name}: {len(spec.items)} 設定 × {spec.rounds} 周 = {total} 実行")
    if dry_run:
        hours = total * (sec_per_item + spec.cooldown_s) / 3600
        print(f"見積もり: 約 {hours:.1f} 時間（1実行 {sec_per_item:.0f} 秒と仮定）")
        for it in spec.items:
            print(f"  {it.key}  {it.overrides}")
        return

    log_path = out_root / spec.name / "sweep_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for rnd in range(spec.rounds):
        order = list(spec.items)
        random.Random(rnd).shuffle(order)  # 【穴①】周の番号をシードにして順序を並べ替える（同じ周なら再開しても同じ順）
        for i, item in enumerate(order):
            marker = _done_marker(out_root, spec.name, item, rnd)
            if resume and marker.exists():  # 【穴②】再開モードで、既に終わった印がある
                continue
            cmd = [sys.executable, SCRIPTS[spec.mode], "--config", str(spec.base), "--tag", f"{spec.name}_r{rnd}"]
            for k, v in item.overrides.items():
                cmd += ["--set", f"{k}={json.dumps(v)}"]
            print(f"[round {rnd} {i + 1}/{len(order)}] {item.overrides}", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)  # 【穴③】子プロセスを実行して終わるまで待ち、出力を文字列で受け取る
            rec = {"round": rnd, "key": item.key, "overrides": item.overrides, "rc": proc.returncode,
                   "sec": time.time() - t0, "stderr_tail": proc.stderr[-2000:]}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if proc.returncode == 0:
                marker.touch()
            else:
                print(f"  失敗 rc={proc.returncode}（ログ: {log_path}）", flush=True)
            time.sleep(spec.cooldown_s)
