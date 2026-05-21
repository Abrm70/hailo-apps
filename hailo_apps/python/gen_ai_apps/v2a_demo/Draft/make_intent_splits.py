"""
Create train/dev splits for intent classification from one or more JSON/JSONL files.

Each input file should contain objects with a text field ("text" or "teks") and a
"label" field. The script merges all inputs, optionally deduplicates by normalized
text+label, shuffles, and writes train/dev JSON arrays.

Examples:
  # Use defaults if files exist in natural_language_processing/
  python Draft/make_intent_splits.py \
    --output-train natural_language_processing/train_split.json \
    --output-dev natural_language_processing/dev_split.json

  # Explicit inputs
  python Draft/make_intent_splits.py \
    --inputs natural_language_processing/final_train_data.json \
             natural_language_processing/additional_en.json \
             natural_language_processing/additional_id.json \
             natural_language_processing/additional_ja.json \
    --dev-ratio 0.1 --seed 42 \
    --output-train natural_language_processing/train_split.json \
    --output-dev natural_language_processing/dev_split.json
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple


def _load_items(path: str, text_keys=("text", "teks")) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.read(4096)
            is_jsonl = "\n" in first and first.strip().startswith("{")
    except Exception:
        is_jsonl = False

    items: List[Dict[str, str]] = []
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
                elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    items = data["data"]
                else:
                    raise RuntimeError("Unsupported JSON structure; expected list or {data: [...]}.")
    except Exception as e:
        raise RuntimeError(f"Failed to parse {path}: {e}")

    normed: List[Dict[str, str]] = []
    for obj in items:
        t = None
        for k in text_keys:
            if k in obj and obj[k] is not None:
                t = obj[k]
                break
        lab = obj.get("label")
        if t is None or lab is None:
            continue
        normed.append({"text": str(t), "label": str(lab)})
    return normed


def _normalize_key(text: str) -> str:
    import re
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge intent data and create train/dev splits.")
    ap.add_argument("--inputs", nargs="*", default=None, help="Input JSON/JSONL files.")
    ap.add_argument("--dev-ratio", type=float, default=0.1, help="Fraction for dev split (0..1).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by normalized text+label.")
    ap.add_argument("--output-train", type=str, required=True, help="Output train JSON path.")
    ap.add_argument("--output-dev", type=str, required=True, help="Output dev JSON path.")
    args = ap.parse_args()

    default_candidates = [
        os.path.join("natural_language_processing", "final_train_data.json"),
        os.path.join("natural_language_processing", "additional_en.json"),
        os.path.join("natural_language_processing", "additional_id.json"),
        os.path.join("natural_language_processing", "additional_ja.json"),
    ]
    inputs = args.inputs if args.inputs else [p for p in default_candidates if os.path.isfile(p)]
    if not inputs:
        print("ERROR: No input files found. Provide --inputs or place dataset files under natural_language_processing/.")
        sys.exit(1)

    # Load and merge
    all_items: List[Dict[str, str]] = []
    for p in inputs:
        try:
            loaded = _load_items(p)
            all_items.extend(loaded)
            print(f"Loaded {len(loaded)} items from {p}")
        except Exception as e:
            print(f"WARNING: Skipping {p}: {e}")

    if not all_items:
        print("ERROR: No valid items after loading inputs.")
        sys.exit(1)

    # Optional dedup by normalized text+label
    if args.dedup:
        seen = set()
        deduped = []
        for it in all_items:
            key = (_normalize_key(it["text"]), it["label"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)
        print(f"Deduplicated: {len(all_items)} -> {len(deduped)}")
        all_items = deduped

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_items)
    n_total = len(all_items)
    n_dev = max(1, int(args.dev_ratio * n_total))
    dev_items = all_items[:n_dev]
    train_items = all_items[n_dev:]

    # Ensure output dirs
    os.makedirs(os.path.dirname(os.path.abspath(args.output_train)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_dev)) or ".", exist_ok=True)

    # Save as JSON arrays
    with open(args.output_train, "w", encoding="utf-8") as f:
        json.dump(train_items, f, ensure_ascii=False, indent=2)
    with open(args.output_dev, "w", encoding="utf-8") as f:
        json.dump(dev_items, f, ensure_ascii=False, indent=2)

    print(f"Wrote train: {len(train_items)} -> {args.output_train}")
    print(f"Wrote dev:   {len(dev_items)} -> {args.output_dev}")


if __name__ == "__main__":
    main()

