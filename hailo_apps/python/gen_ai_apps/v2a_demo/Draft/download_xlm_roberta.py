"""
Utility to download a Hugging Face XLM‑RoBERTa model (or any repo_id)
into a local folder for fully offline use with Transformers.

Examples:
  - Download base XLM‑RoBERTa into project folder (no symlinks, portable):
      python Draft/download_xlm_roberta.py \
        --repo-id xlm-roberta-base \
        --local-dir natural_language_processing/xlmroberta_base \
        --copy

  - Download your fine‑tuned classifier checkpoint by repo id:
      python Draft/download_xlm_roberta.py \
        --repo-id your-username/your-xlmroberta-intent \
        --local-dir natural_language_processing/xlmroberta_intent_final \
        --copy

After downloading, load fully offline with:
  AutoTokenizer.from_pretrained(<local_dir>, local_files_only=True)
  AutoModelForSequenceClassification.from_pretrained(<local_dir>, local_files_only=True)
"""

import argparse
import os
import sys
from typing import List, Optional


def _snapshot_download(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    use_symlinks: bool = False,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(
            "ERROR: huggingface_hub is not installed. Please install it first:\n"
            "  pip install huggingface_hub\n"
            f"Details: {e}"
        )
        sys.exit(1)

    os.makedirs(local_dir, exist_ok=True)

    kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "local_dir_use_symlinks": use_symlinks,
        "resume_download": True,
    }
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    print(
        f"Downloading '{repo_id}' to '{local_dir}'"
        f" (revision={revision or 'main'}, symlinks={'on' if use_symlinks else 'off'})..."
    )
    path = snapshot_download(**kwargs)
    print(f"Done. Local path: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a HF repo (e.g., xlm-roberta-base) for offline use.")
    parser.add_argument("--repo-id", type=str, default="xlm-roberta-base", help="HF repo id to download.")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=os.path.join("natural_language_processing", "xlmroberta_base"),
        help="Destination directory for local files.",
    )
    parser.add_argument("--revision", type=str, default=None, help="Optional git revision/tag.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--copy", action="store_true", help="Copy files (no symlinks). Recommended for portability.")
    group.add_argument("--use-symlinks", action="store_true", help="Use symlinks to HF cache (faster, less portable).")

    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Attempt to load tokenizer/model locally to verify offline readiness.",
    )

    args = parser.parse_args()

    use_symlinks = args.use_symlinks and not args.copy
    local_path = _snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        use_symlinks=use_symlinks,
    )

    if args.verify_load:
        try:
            from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        except Exception as e:
            print(
                "WARNING: transformers is not installed; skipping verification.\n"
                "Install with: pip install transformers"
            )
            return

        print("Verifying offline load (tokenizer)…")
        tok = None
        try:
            tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            print("  Tokenizer OK")
        except Exception as e:
            print(f"  Tokenizer load failed: {e}")

        print("Verifying offline load (model)…")
        try:
            # Try classification first; fall back to base model if not classification head
            try:
                _ = AutoModelForSequenceClassification.from_pretrained(local_path, local_files_only=True)
                print("  Model (classification) OK")
            except Exception:
                _ = AutoModel.from_pretrained(local_path, local_files_only=True)
                print("  Model (base) OK")
        except Exception as e:
            print(f"  Model load failed: {e}")


if __name__ == "__main__":
    main()

