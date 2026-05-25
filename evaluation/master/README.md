# Master Evaluation Scripts

Top-level orchestration: a single command runs every per-domain evaluation
that the paper reports.

## Files

- `eval_all_tasks.sh` — Drives the six paper tasks in one shot
  (medical, thai, finance, indonesian, math, cantonese, malay) by calling
  `lm-evaluation-harness`, `Yue-Benchmark`, and `MalayMMLU`. Skips a task
  whose output directory already contains `results*.json`. Appends a
  human-readable summary block per task to `eval_results_summary_base.log`.
- `run_source8b_eval.sh` — Evaluates the source LLaMA-3.1-8B-Instruct model
  on every paper benchmark (used to produce the source-8B baseline numbers).
- `test_lm_eval_env.sh` — Minimal sanity check (imports + a 1-batch boolq
  run) that the `lm-eval` conda environment can launch. Run this once after
  `pip install -e ".[dev]"` on `lm-evaluation-harness`.

## Required environment

Hardcoded paths inside the scripts assume the original lab layout
(`/data/chenhang/...`, `/home/chenhang/...`); they are easy to override:

- `LM_EVAL_REPO`, `MALAY_REPO`, `YUE_BENCHMARK_ROOT` — repo locations.
- `HF_DATASETS_CACHE`, `HF_ENDPOINT` — HF cache + mirror.
- `MODEL_*_OVERRIDE` (in some scripts) — substitute model paths.

## Conda envs used

- `trans_opt` — for our training-side code.
- `lm-eval` — for evaluation (`pip install -e ".[dev]"` in
  `lm-evaluation-harness`).

The CMMLU and MalayMMLU paths in `eval_all_tasks.sh` switch into `lm-eval`
on the fly via `conda activate`.
