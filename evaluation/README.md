# Evaluation Pipelines

End-to-end evaluation code for every domain reported in the paper.

This directory now ships the **complete** per-domain pipelines used to
generate the paper's main and appendix tables, organised by task. Three
external benchmark repositories are reused as-is — `lm-evaluation-harness`,
`Yue-Benchmark`, `MalayMMLU` — and we provide thin wrappers that the
master driver (`master/eval_all_tasks.sh`) orchestrates.

## Layout

```
evaluation/
├── master/                     # Top-level orchestration
│   ├── eval_all_tasks.sh       # All 6 paper tasks, one shot
│   ├── run_source8b_eval.sh    # Source-8B baseline numbers
│   └── test_lm_eval_env.sh     # Sanity check for lm-eval env
├── medical/                    # MedQA + MMLU medical subsets (lm-eval)
├── thai/                       # XCOPA / XQuAD / XNLI Thai (lm-eval)
├── finance/                    # Finance lm-eval tasks + FinanceQA generative eval
├── indonesian/                 # Indonesian lm-eval + bundled IndoMMLU pipeline
├── math/                       # Arithmetic + Minerva math + MMLU elementary math
├── cantonese/                  # CMMLU / Yue-MMLU prediction + aggregation
├── malay/                      # Bundled MalayMMLU evaluator (code-only)
├── general_ability/            # arc_easy / commonsense_qa / piqa / social_iqa / winogrande
└── utils/                      # Result aggregation, picking, baseline-vs-best comparison
```

Every leaf folder has its own `README.md` with the task list, the exact
`lm_eval` (or domain-specific) command line, environment variables,
and any aggregation step.

## External repos required

| Domain                                    | External repo                                       |
| ----------------------------------------- | --------------------------------------------------- |
| medical / thai / finance / indonesian     | [lm-evaluation-harness] (`pip install -e ".[dev]"`) |
| math / general_ability                    | [lm-evaluation-harness]                             |
| cantonese                                 | [Yue-Benchmark] (only the `data/latest_data/` tree) |
| malay                                     | none — bundled under `malay/MalayMMLU/`             |
| indonesian (subject-level IndoMMLU)       | none — bundled under `indonesian/IndoMMLU/`         |

Set these env vars once, then call `master/eval_all_tasks.sh`:

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export YUE_BENCHMARK_ROOT=/path/to/Yue-Benchmark
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"
export HF_DATASETS_CACHE=/path/to/hf_cache_eval
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ENDPOINT=https://hf-mirror.com   # optional, for users behind a mirror
```

## End-to-end run

After training the fused models with
`scripts/run_train_final.sh` (see top-level `REPRODUCE.md`), run all six
paper tasks for one set of (alpha, lr) checkpoints:

```bash
cd evaluation/master
bash eval_all_tasks.sh
```

Each task is skipped if its output directory already contains
`results*.json`, so the script is safely re-runnable. A combined log file
`eval_results_summary_base.log` accumulates a human-readable summary per
task per session.

## Aggregating sweep results

When running an alpha / lr sweep (the alpha-ablation tables in the
appendix), use the post-processing tools in `utils/`:

```bash
# 1. Collect raw numbers per (alpha, lr, variant)
python utils/extract_eval_results.py \
  --root /path/to/output \
  --output extracted.csv

# 2. Pick best config vs baseline
python utils/auto_eval_baseline.py \
  --root /path/to/output \
  --output best_per_task.csv

# 3. Render paper-ready table
python utils/pick.py \
  --csv extracted.csv \
  --output paper_table.csv
```

For Cantonese specifically, `cantonese/aggregate_cmmlu_results.py` and
`cantonese/show_cmmlu_table.py` produce the CMMLU appendix tables
directly.

## Notes on hardcoded paths

The shell wrappers carried over from the lab still contain absolute
paths (`/data/chenhang/...`, `/home/chenhang/...`). They serve as
reference invocations — every script either accepts an env-var override
(see in-script comments) or is short enough to edit in place when
porting to a new machine.

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
[Yue-Benchmark]: https://github.com/jiangjyjy/Yue-Benchmark
