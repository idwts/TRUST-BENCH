#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from risk_utils import read_folds, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped cross-validation with the generic full-parameter trainer.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-file", default=str(Path(__file__).resolve().parents[1] / "data" / "episodes_paper_id_v3.json"))
    parser.add_argument("--folds-file", default=str(Path(__file__).resolve().parents[1] / "data" / "folds_paper_id_v3.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-filter", default="")
    parser.add_argument("--attack-profile-filter", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=1.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = read_folds(args.folds_file)
    train_script = Path(__file__).resolve().parent / "train_risk_model_full.py"

    fold_runs = []
    for fold_idx in range(len(folds)):
        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        cmd = [
            sys.executable,
            str(train_script),
            "--model-name-or-path",
            args.model_name_or_path,
            "--data-file",
            args.data_file,
            "--folds-file",
            args.folds_file,
            "--validation-fold",
            str(fold_idx),
            "--output-dir",
            str(fold_dir),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--warmup-ratio",
            str(args.warmup_ratio),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--per-device-eval-batch-size",
            str(args.per_device_eval_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--max-length",
            str(args.max_length),
            "--seed",
            str(args.seed + fold_idx),
            "--rho",
            str(args.rho),
        ]
        if args.method_filter:
            cmd.extend(["--method-filter", args.method_filter])
        if args.attack_profile_filter:
            cmd.extend(["--attack-profile-filter", args.attack_profile_filter])
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if args.bf16:
            cmd.append("--bf16")
        if args.fp16:
            cmd.append("--fp16")
        if args.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
        subprocess.run(cmd, check=True)
        summary_path = fold_dir / "summary.json"
        if summary_path.exists():
            fold_runs.append(json.loads(summary_path.read_text(encoding="utf-8")))

    write_json(out_dir / "fold_runs.json", fold_runs)
    print(json.dumps({"completed_folds": len(fold_runs), "output_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

