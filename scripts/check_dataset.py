#!/usr/bin/env python3
"""生エピソードの品質チェックと、D5 判定ゲートの数値。docs/track2/34_episode_recorder.md、docs/00_schedule.md"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evla.dataset.quality import check_dataset, gate_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw/so101_pickplace_v1"))
    p.add_argument("--out", type=Path, default=Path("reports/dataset_quality.md"))
    a = p.parse_args()
    df = check_dataset(a.raw)
    g = gate_summary(df)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bad = df[~df["ok"]]
    a.out.write_text(
        "# データセット品質\n\n```json\n" + json.dumps(g, indent=2) + "\n```\n\n## 問題のあるエピソード\n\n"
        + (bad.to_markdown(index=False) if len(bad) else "なし") + "\n"
    )
    print(json.dumps(g, indent=2))
    print(f"report → {a.out}")


if __name__ == "__main__":
    main()
