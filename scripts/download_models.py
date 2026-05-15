"""Batch-download every model used by the paper from Hugging Face.

Usage:
    # Optional: a token is only needed for gated/private repos. If you do not need one, omit it.
    export HF_TOKEN=hf_xxx
    # Optional: enable the China mirror (used originally) — comment out if you do not need it.
    export HF_ENDPOINT=https://hf-mirror.com

    # Default destination is $MODELS_ROOT (or ./models) at the repo root.
    python scripts/download_models.py
    python scripts/download_models.py --tasks medical thai          # subset
    python scripts/download_models.py --models-root /data/my/models # custom dest

This reproduces the exact HF repo IDs the paper used. Per-task fused-model
output names (e.g. `medllama_fused_alpha01_fortrain_1b`) and α/lr live in
MODELS.md; the end-to-end command is in REPRODUCE.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# (task, hf_repo_id, local_dirname)
HF_REPO_MAP: list[tuple[str, str, str]] = [
    # Six tasks reported in the paper (run_train_final.sh)
    ("medical",    "PathFinderKR/Llama-3-1B-Medical-Instruct",        "llama3-1b-med"),
    ("thai",       "scb10x/llama3.2-typhoon2-1b-instruct",            "llama3.2-typhoon2-1b-instruct"),
    ("finance",    "unsloth/Llama-3.2-1B-Instruct",                   "Llama-3.2-1B-Instruct"),
    ("cantonese",  "FlagAlpha/Llama3-Chinese-8B-Instruct",            "Llama3-Chinese-8B-Instruct"),
    ("indonesian", "digo-prayudha/Llama-3.2-1B-Indonesian-lora",      "Llama-3.2-1B-Indonesian-QLora"),
    ("malay",      "mesolitica/Malaysian-Llama-3.2-1B-Instruct",      "Malaysian-Llama-3.2-1B-Instruct-v0.1"),
    # The shared 8B donor for every task above
    ("base_8b",    "unsloth/Llama-3.1-8B-Instruct",                   "Llama-3.1-8B-Instruct"),
    # Optional secondary experiments (math/reasoning ablation in trans_train_reasoning.sh)
    ("math_1b",    "masani/SFT_math_Llama-3.2-1B_epoch_1_global_step_29", "SFT_math_Llama-3.2-1B"),
]


def hf_endpoint_for(repo_id: str, default_endpoint: str | None) -> str | None:
    if repo_id.startswith("meta-llama/"):
        return None
    return default_endpoint


def download(task: str, repo_id: str, local_dir: Path, token: str | None) -> bool:
    from huggingface_hub import snapshot_download

    print(f"\n=== {task}: {repo_id} -> {local_dir} ===", flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
            max_workers=8,
        )
        print(f"[OK] {task}", flush=True)
        return True
    except Exception as exc:
        print(f"[FAIL] {task}: {exc}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        default=os.environ.get("MODELS_ROOT", str(Path(__file__).resolve().parent.parent / "models")),
        help="Destination directory (default: $MODELS_ROOT or ./models at repo root).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Subset of {[t for t, *_ in HF_REPO_MAP]} (default: all).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    default_endpoint = os.environ.get("HF_ENDPOINT")

    models_root = Path(args.models_root).resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {models_root}", flush=True)

    selected = HF_REPO_MAP if not args.tasks else [
        row for row in HF_REPO_MAP if row[0] in args.tasks
    ]
    if not selected:
        print(f"[ERROR] No tasks matched {args.tasks}.", file=sys.stderr)
        return 2

    failures: list[str] = []
    for task, repo_id, dirname in selected:
        ep = hf_endpoint_for(repo_id, default_endpoint)
        if ep is not None:
            os.environ["HF_ENDPOINT"] = ep
        else:
            os.environ.pop("HF_ENDPOINT", None)
        if not download(task, repo_id, models_root / dirname, token):
            failures.append(task)

    if failures:
        print(f"\n[SUMMARY] {len(failures)} task(s) failed: {failures}", file=sys.stderr)
        return 1
    print(f"\n[SUMMARY] All {len(selected)} downloads OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
