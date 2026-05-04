#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from risk_utils import (
    DEFAULT_RHO,
    RISK_HIGH,
    RISK_LOW,
    build_group_key,
    build_prompt,
    build_structured_observation,
    guarded_metrics_from_decisions,
    label_to_decision,
    mode_to_label,
    optimize_threshold,
    read_folds,
    read_records,
    write_json,
)


@dataclass
class Example:
    prompt: str
    label: str
    mode_served: str
    group_key: str
    benchmark: str
    task_id: str
    variant_id: str


class PromptOnlyLossDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        label_text = " " + ex.label
        prompt_ids = self.tokenizer.encode(ex.prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(label_text, add_special_tokens=False)
        max_prompt_len = max(8, self.max_length - len(label_ids) - 1)
        prompt_ids = prompt_ids[:max_prompt_len]
        input_ids = prompt_ids + label_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + label_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class Collator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_label_logprob(model, tokenizer, prompt: str, label: str) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_ids = tokenizer.encode(" " + label, add_special_tokens=False)
    input_ids = prompt_ids + label_ids
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tensor).logits
    total = 0.0
    for pos in range(len(prompt_ids), len(input_ids)):
        log_probs = torch.log_softmax(logits[0, pos - 1], dim=-1)
        total += log_probs[input_ids[pos]].item()
    return float(total)


def score_example(model, tokenizer, example: Example) -> float:
    hi = compute_label_logprob(model, tokenizer, example.prompt, RISK_HIGH)
    lo = compute_label_logprob(model, tokenizer, example.prompt, RISK_LOW)
    return float(hi - lo)


def prepare_examples(records: list[dict[str, Any]]) -> list[Example]:
    methods = sorted({str(r.get("method", "")) for r in records}, key=len, reverse=True)
    attacks = sorted({str(r.get("attack_profile", "")) for r in records}, key=len, reverse=True)
    examples: list[Example] = []
    for ep in records:
        obs = build_structured_observation(ep)
        examples.append(
            Example(
                prompt=build_prompt(obs),
                label=mode_to_label(str(ep.get("mode_served", "normal"))),
                mode_served=str(ep.get("mode_served", "normal")),
                group_key=build_group_key(ep, methods, attacks),
                benchmark=str(ep.get("benchmark", "unknown")),
                task_id=str(ep.get("task_id", "")),
                variant_id=str(ep.get("variant_id", "")),
            )
        )
    return examples


def split_examples(examples: list[Example], folds_file: Path | None, validation_fold: int | None) -> tuple[list[Example], list[Example]]:
    if folds_file is None or validation_fold is None:
        n = len(examples)
        cut = max(1, int(0.2 * n))
        return examples[cut:], examples[:cut]
    folds = read_folds(folds_file)
    val_keys = set(folds[validation_fold])
    train_examples = [x for x in examples if x.group_key not in val_keys]
    val_examples = [x for x in examples if x.group_key in val_keys]
    return train_examples, val_examples


def evaluate_model(model, tokenizer, train_examples: list[Example], val_examples: list[Example], rho: float) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    train_rows = [{"score": score_example(model, tokenizer, ex), "mode_served": ex.mode_served} for ex in train_examples]
    threshold, train_metrics = optimize_threshold(train_rows, rho=rho)
    predictions = []
    for ex in val_examples:
        score = score_example(model, tokenizer, ex)
        label = RISK_HIGH if score >= threshold else RISK_LOW
        predictions.append(
            {
                "benchmark": ex.benchmark,
                "task_id": ex.task_id,
                "variant_id": ex.variant_id,
                "mode_served": ex.mode_served,
                "score": round(score, 6),
                "prediction_label": label,
                "prediction": label_to_decision(label),
            }
        )
    val_metrics = guarded_metrics_from_decisions(predictions, rho=rho)
    return threshold, {"train": train_metrics, "val": val_metrics}, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic full-parameter risk-model fine-tuning for TRUST-Bench.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-file", default=str(Path(__file__).resolve().parents[1] / "data" / "episodes_paper_id_v3.json"))
    parser.add_argument("--folds-file", default=str(Path(__file__).resolve().parents[1] / "data" / "folds_paper_id_v3.json"))
    parser.add_argument("--validation-fold", type=int, default=None)
    parser.add_argument("--method-filter", default="")
    parser.add_argument("--attack-profile-filter", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(args.data_file)
    if args.method_filter:
        records = [r for r in records if str(r.get("method", "")) == args.method_filter]
    if args.attack_profile_filter:
        records = [r for r in records if str(r.get("attack_profile", "")) == args.attack_profile_filter]

    examples = prepare_examples(records)
    train_examples, val_examples = split_examples(
        examples,
        Path(args.folds_file) if args.folds_file else None,
        args.validation_fold,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = PromptOnlyLossDataset(train_examples, tokenizer, args.max_length)
    val_dataset = PromptOnlyLossDataset(val_examples, tokenizer, args.max_length)
    collator = Collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
        bf16=args.bf16,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    final_model_dir = out_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    threshold, metrics, predictions = evaluate_model(trainer.model, tokenizer, train_examples, val_examples, rho=args.rho)
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "validation_fold": args.validation_fold,
        "method_filter": args.method_filter or None,
        "attack_profile_filter": args.attack_profile_filter or None,
        "rho": args.rho,
        "threshold": round(threshold, 6),
        "metrics": metrics,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "predictions.json", predictions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
