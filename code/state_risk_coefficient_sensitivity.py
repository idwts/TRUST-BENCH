#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

from risk_utils import (
    DEFAULT_RHO,
    build_group_key,
    extract_state_features,
    guarded_metrics_from_decisions,
    optimize_threshold,
    read_folds,
    read_records,
    write_json,
)


DEFAULT_WEIGHTS = {"trig": 0.24, "probe": 0.20, "mis": 0.18, "drift": 0.12, "warn": 0.10}
BASE_BIAS = 0.08
WEIGHT_ORDER = ["trig", "probe", "mis", "drift", "warn"]


@dataclass(frozen=True)
class Config:
    name: str
    group: str
    description: str
    weights: dict[str, float]


def compute_state_score(counts: dict[str, int], weights: dict[str, float]) -> float:
    score = BASE_BIAS + sum(weights[k] * counts[k] for k in WEIGHT_ORDER)
    return max(0.0, min(1.0, float(score)))


def build_configs() -> list[Config]:
    configs = [Config("default", "reference", "Released implementation constants.", dict(DEFAULT_WEIGHTS))]
    for key in WEIGHT_ORDER:
        weights = dict(DEFAULT_WEIGHTS)
        weights[key] = 0.0
        configs.append(Config(f"no_{key}", "leave_one_out", f"Zero out c_{key}.", weights))
    for factor, tag in [(0.5, "half"), (1.5, "plus50")]:
        for key in WEIGHT_ORDER:
            weights = dict(DEFAULT_WEIGHTS)
            weights[key] = round(weights[key] * factor, 6)
            configs.append(Config(f"{tag}_{key}", "local_sensitivity", f"Multiply c_{key} by {factor}.", weights))
    total = sum(DEFAULT_WEIGHTS.values())
    uniform = round(total / len(WEIGHT_ORDER), 6)
    configs.append(Config("uniform_same_sum", "structure_check", "Uniform weights with the same total mass.", {k: uniform for k in WEIGHT_ORDER}))
    configs.append(Config("reverse_emphasis", "structure_check", "Reverse the original weight ordering.", {"trig": 0.10, "probe": 0.12, "mis": 0.18, "drift": 0.20, "warn": 0.24}))
    configs.append(Config("signed_benign_noise", "diagnostic_signed", "Diagnostic only: flip drift/warn negative.", {"trig": 0.24, "probe": 0.20, "mis": 0.18, "drift": -0.12, "warn": -0.10}))
    configs.append(Config("signed_warn_only", "diagnostic_signed", "Diagnostic only: warning negative.", {"trig": 0.24, "probe": 0.20, "mis": 0.18, "drift": 0.12, "warn": -0.10}))
    configs.append(Config("signed_drift_only", "diagnostic_signed", "Diagnostic only: drift negative.", {"trig": 0.24, "probe": 0.20, "mis": 0.18, "drift": -0.12, "warn": 0.10}))
    return configs


def compute_auc(rows: list[dict[str, Any]]) -> float | None:
    if roc_auc_score is None or not rows:
        return None
    y_true = [1 if r["mode_served"] == "malicious" else 0 for r in rows]
    if len(set(y_true)) < 2:
        return None
    y_score = [float(r["score"]) for r in rows]
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity sweep for deterministic state_risk coefficients.")
    parser.add_argument("--data-file", default=str(Path(__file__).resolve().parents[1] / "data" / "episodes_paper_id_v3.json"))
    parser.add_argument("--folds-file", default=str(Path(__file__).resolve().parents[1] / "data" / "folds_paper_id_v3.json"))
    parser.add_argument("--method-filter", default="cwm_optimized")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "analysis" / "state_risk_coefficient_sensitivity"))
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = read_records(args.data_file)
    records = [r for r in all_records if not args.method_filter or str(r.get("method", "")) == args.method_filter]
    folds = read_folds(args.folds_file)
    methods = sorted({str(r.get("method", "")) for r in records}, key=len, reverse=True)
    attacks = sorted({str(r.get("attack_profile", "")) for r in records}, key=len, reverse=True)

    rows = []
    global_signal_totals = defaultdict(int)
    global_nonzero = defaultdict(int)
    for r in all_records:
        sf = extract_state_features(r.get("trajectory") if isinstance(r.get("trajectory"), list) else [])
        counts = {
            "trig": int(sf.get("triggered_count", 0)),
            "probe": int(sf.get("probe_detect_count", 0)),
            "mis": int(sf.get("high_mismatch_count", 0)),
            "drift": int(sf.get("id_drift_count", 0)),
            "warn": int(sf.get("warning_count", 0)),
        }
        for k, v in counts.items():
            global_signal_totals[k] += v
            if v > 0:
                global_nonzero[k] += 1
        if args.method_filter and str(r.get("method", "")) != args.method_filter:
            continue
        rows.append(
            {
                "group_key": build_group_key(r, methods, attacks),
                "mode_served": str(r.get("mode_served", "normal")),
                "counts": counts,
            }
        )

    results = []
    default_result = None
    for cfg in build_configs():
        scored_rows = [{**r, "score": compute_state_score(r["counts"], cfg.weights)} for r in rows]
        fold_guarded = []
        fold_joint = []
        fold_aucs = []
        per_fold = []
        pooled_predictions = []
        for fold_idx, val_keys in enumerate(folds):
            val_key_set = set(val_keys)
            train_rows = [r for r in scored_rows if r["group_key"] not in val_key_set]
            val_rows = [r for r in scored_rows if r["group_key"] in val_key_set]
            threshold, train_metrics = optimize_threshold(train_rows, rho=args.rho)
            val_predictions = []
            for row in val_rows:
                prediction = "reject" if float(row["score"]) >= threshold else "execute"
                val_predictions.append({"mode_served": row["mode_served"], "prediction": prediction})
                pooled_predictions.append({"mode_served": row["mode_served"], "prediction": prediction})
            val_metrics = guarded_metrics_from_decisions(val_predictions, rho=args.rho)
            auc = compute_auc(val_rows)
            fold_guarded.append(float(val_metrics["guarded"]))
            fold_joint.append(float(val_metrics["joint"]))
            if auc is not None:
                fold_aucs.append(float(auc))
            per_fold.append(
                {
                    "fold": fold_idx,
                    "threshold": round(threshold, 6),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "val_auc": round(auc, 4) if auc is not None else None,
                }
            )
        summary = {
            "name": cfg.name,
            "group": cfg.group,
            "description": cfg.description,
            "weights": cfg.weights,
            "guarded_mean": round(statistics.mean(fold_guarded), 2),
            "guarded_std": round(statistics.stdev(fold_guarded) if len(fold_guarded) > 1 else 0.0, 2),
            "joint_mean": round(statistics.mean(fold_joint), 2),
            "joint_std": round(statistics.stdev(fold_joint) if len(fold_joint) > 1 else 0.0, 2),
            "auc_mean": round(statistics.mean(fold_aucs), 4) if fold_aucs else None,
            "pooled": guarded_metrics_from_decisions(pooled_predictions, rho=args.rho),
            "per_fold": per_fold,
        }
        if cfg.name == "default":
            default_result = summary
        results.append(summary)

    assert default_result is not None
    for result in results:
        result["delta_guarded_mean_vs_default"] = round(result["guarded_mean"] - default_result["guarded_mean"], 2)
        result["delta_auc_mean_vs_default"] = round((result["auc_mean"] or 0.0) - (default_result["auc_mean"] or 0.0), 4)
    results.sort(key=lambda x: (x["guarded_mean"], x["auc_mean"] or -1.0), reverse=True)

    summary = {
        "rho": args.rho,
        "base_bias": BASE_BIAS,
        "default_weights": DEFAULT_WEIGHTS,
        "n_rows": len(rows),
        "n_groups": len({r["group_key"] for r in rows}),
        "diagnostics": {
            "global_signal_activity_all_methods": {
                "total_rows": len(all_records),
                "sum_counts": dict(global_signal_totals),
                "nonzero_row_pct": {k: round(100.0 * global_nonzero[k] / max(1, len(all_records)), 2) for k in WEIGHT_ORDER},
            }
        },
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "group", "guarded_mean", "guarded_std", "joint_mean", "joint_std", "auc_mean", "delta_guarded_mean_vs_default", "delta_auc_mean_vs_default", "trig", "probe", "mis", "drift", "warn"])
        for result in results:
            w = result["weights"]
            writer.writerow([result["name"], result["group"], result["guarded_mean"], result["guarded_std"], result["joint_mean"], result["joint_std"], result["auc_mean"], result["delta_guarded_mean_vs_default"], result["delta_auc_mean_vs_default"], w["trig"], w["probe"], w["mis"], w["drift"], w["warn"]])
    print(json.dumps({"top3": results[:3], "default": default_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

