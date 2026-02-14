#!/usr/bin/env python3
"""Aggregate all training CSV logs into results/results.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_run_summary import RUN_NAME_RE, summarize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate all run logs into one CSV.")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_config.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(
        p for p in args.logs_dir.glob("*.csv") if RUN_NAME_RE.match(p.name)
    )
    if not csv_paths:
        raise FileNotFoundError(f"No training CSV logs found in: {args.logs_dir}")

    rows = [summarize_run(csv_path=path, config_path=args.config) for path in csv_paths]
    df = pd.DataFrame(rows)
    df = df.sort_values(["framework", "dataset", "model", "precision", "seed"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"aggregated_logs={len(csv_paths)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
