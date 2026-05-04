#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fold-level TRUST-Bench summaries.")
    parser.add_argument("--runs-dir", required=True, help="Directory containing fold_00, fold_01, ... subdirectories.")
    parser.add_argument("--summary-name", default="summary.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    fold_metrics = []
    for fold_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")):
        summary_path = fold_dir / args.summary_name
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        val = summary["metrics"]["val"]
        fold_metrics.append(
            {
                "fold": fold_dir.name,
                "guarded": float(val["guarded"]),
                "joint": float(val["joint"]),
                "amr": float(val["amr"]),
                "bmr": float(val["bmr"]),
                "rnr": float(val["rnr"]),
                "acnr": float(val["acnr"]),
            }
        )

    if not fold_metrics:
        raise RuntimeError(f"No fold summaries found under {runs_dir}")

    aggregate = {
        "folds": fold_metrics,
        "guarded_mean": round(statistics.mean(x["guarded"] for x in fold_metrics), 2),
        "guarded_std": round(statistics.stdev(x["guarded"] for x in fold_metrics) if len(fold_metrics) > 1 else 0.0, 2),
        "joint_mean": round(statistics.mean(x["joint"] for x in fold_metrics), 2),
        "joint_std": round(statistics.stdev(x["joint"] for x in fold_metrics) if len(fold_metrics) > 1 else 0.0, 2),
    }
    out_path = runs_dir / "aggregate_summary.json"
    out_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

