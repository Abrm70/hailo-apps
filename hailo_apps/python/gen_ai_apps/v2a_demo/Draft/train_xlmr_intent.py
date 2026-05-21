"""
Fine-tune XLM-RoBERTa (or any HF encoder) for intent classification (ID/EN/JA) offline-capable.

This script expects labeled text data and saves a local checkpoint with
id2label/label2id in config.json for fully offline inference.

Data formats supported (auto-detected):
- JSONL: each line is a JSON object with fields: text + label
- JSON (list): a JSON array of objects with fields: text + label

If your data uses different field names (e.g., teks for text), use --text-field/--label-field.

Examples:
- Base model from hub (online once), train, and save:
  python Draft/train_xlmr_intent.py \
    --model_name_or_path xlm-roberta-base \
    --train_file data/train.jsonl --eval_file data/dev.jsonl \
    --output_dir natural_language_processing/xlmroberta_intent_final

- Fully offline using a local base folder (downloaded beforehand):
  python Draft/train_xlmr_intent.py \
    --model_name_or_path natural_language_processing/xlmroberta_base \
    --train_file data/train.json --eval_file data/dev.json \
    --output_dir natural_language_processing/xlmroberta_intent_final \
    --local_only
"""

import argparse
import json
import os
import random
import inspect
from typing import Dict, List, Tuple

import numpy as np


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_from_file(path: str, text_field: str, label_field: str):
    """Load a dataset from JSON/JSONL into a Hugging Face Dataset.

    Supports:
    - JSONL (one JSON object per line)
    - JSON (list of objects)
    """
    from datasets import Dataset

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    items: List[Dict] = []
    # Try JSON lines first
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_chunk = f.read(4096)
            # Heuristic: JSONL usually has many newlines and multiple JSON objects
            is_jsonl = "\n" in first_chunk and first_chunk.strip().startswith("{")
    except Exception:
        is_jsonl = False

    try:
        if is_jsonl:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    items.append(obj)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                else:
                    # Allow {"data": [...]} structure
                    items = data.get("data", []) if isinstance(data, dict) else []
    except Exception as e:
        raise RuntimeError(f"Failed to parse dataset file '{path}': {e}")

    # Normalize fields (e.g., teks -> text)
    normalized: List[Dict] = []
    for obj in items:
        # Prefer explicit fields; fallback to common alternatives
        t = obj.get(text_field)
        if t is None:
            t = obj.get("text")
        if t is None:
            t = obj.get("teks")
        lab = obj.get(label_field)
        if lab is None:
            lab = obj.get("label")
        if t is None or lab is None:
            # Skip malformed entries
            continue
        normalized.append({"text": str(t), "label": str(lab)})

    if not normalized:
        raise RuntimeError(
            f"No valid samples found in '{path}'. Check --text-field/--label-field and file format."
        )

    return Dataset.from_list(normalized)


def build_label_maps(labels: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    # Deterministic ordering for reproducibility
    uniq = sorted(set(labels))
    id2label = {i: lab for i, lab in enumerate(uniq)}
    label2id = {lab: i for i, lab in id2label.items()}
    return id2label, label2id


def compute_metrics_builder():
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R for intent classification (offline-capable)")
    parser.add_argument("--model_name_or_path", type=str, default="xlm-roberta-base",
                        help="HF model id or local dir for base encoder.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train JSON/JSONL file.")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to eval/validation JSON/JSONL file.")
    parser.add_argument("--output_dir", type=str, default="natural_language_processing/xlmroberta_intent_final",
                        help="Directory to save the fine-tuned model.")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Field name for utterance text (supports 'teks' automatically).")
    parser.add_argument("--label-field", type=str, default="label",
                        help="Field name for intent label.")

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 if GPU supports it.")
    parser.add_argument("--local_only", action="store_true", help="Force local files only for model/tokenizer.")
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load datasets
    ds_train = load_dataset_from_file(args.train_file, args.text_field, args.label_field)
    ds_eval = load_dataset_from_file(args.eval_file, args.text_field, args.label_field)

    # Build labels
    labels_all = list(ds_train["label"]) + list(ds_eval["label"])
    id2label, label2id = build_label_maps(labels_all)

    # Tokenizer & model
    from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    # EarlyStoppingCallback may not exist in older versions
    try:
        from transformers.trainer_callback import EarlyStoppingCallback
    except Exception:
        EarlyStoppingCallback = None

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_only)

    def preprocess(batch):
        enc = tok(batch["text"], truncation=True, max_length=args.max_length)
        enc["labels"] = [label2id[l] for l in batch["label"]]
        return enc

    ds_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
    ds_eval = ds_eval.map(preprocess, batched=True, remove_columns=ds_eval.column_names)

    cfg = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        local_files_only=args.local_only,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=cfg,
        local_files_only=args.local_only,
    )

    data_collator = DataCollatorWithPadding(tok)

    # Build TrainingArguments adaptively; prefer epoch-based eval/save
    ta_sig = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    base_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": 50,
        "fp16": args.fp16,
    }
    # Optional common extras
    if "save_total_limit" in ta_sig:
        base_kwargs["save_total_limit"] = args.save_total_limit
    if "metric_for_best_model" in ta_sig:
        base_kwargs["metric_for_best_model"] = "f1_micro"
    if "greater_is_better" in ta_sig:
        base_kwargs["greater_is_better"] = True
    if "report_to" in ta_sig:
        base_kwargs["report_to"] = []

    # Try epoch strategies first
    epoch_kwargs = dict(base_kwargs)
    if "evaluation_strategy" in ta_sig:
        epoch_kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in ta_sig:
        epoch_kwargs["save_strategy"] = "epoch"
    if "load_best_model_at_end" in ta_sig:
        epoch_kwargs["load_best_model_at_end"] = True

    try:
        training_args = TrainingArguments(**epoch_kwargs)
    except Exception:
        # Fallback for very old versions: use steps + evaluate_during_training
        steps_kwargs = dict(base_kwargs)
        if "evaluate_during_training" in ta_sig:
            steps_kwargs["evaluate_during_training"] = True
        if "eval_steps" in ta_sig:
            steps_kwargs["eval_steps"] = 500
        if "save_steps" in ta_sig:
            steps_kwargs["save_steps"] = 500
        # Avoid load_best_model_at_end if strategies can't be aligned
        if "load_best_model_at_end" in ta_sig:
            steps_kwargs.pop("load_best_model_at_end", None)
        training_args = TrainingArguments(**steps_kwargs)

    # Attach EarlyStopping only if evaluation is actually enabled
    def _has_active_eval(tr_args) -> bool:
        try:
            val = getattr(tr_args, "evaluation_strategy", None)
            if val is not None and str(val).lower() not in ("no", "none", "intervalstrategy.no"):
                return True
        except Exception:
            pass
        try:
            val = getattr(tr_args, "eval_strategy", None)
            if val is not None and str(val).lower() not in ("no", "none", "intervalstrategy.no"):
                return True
        except Exception:
            pass
        try:
            # Very old versions
            val = getattr(tr_args, "evaluate_during_training", None)
            if bool(val):
                return True
        except Exception:
            pass
        return False

    callbacks = []
    if EarlyStoppingCallback is not None and _has_active_eval(training_args):
        try:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        except Exception:
            pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    # Save best model and tokenizer with id2label/label2id in config
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
