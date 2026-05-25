# Evaluation Utilities

Aggregation, picking, and reporting tools for many-config eval sweeps.

## Files

- `extract_eval_results.py` (+ `extract_eval_results_README.md`) —
  generic `results_*.json` walker that produces hot / nohot / untrained
  CSVs sorted by primary metric. The companion shell wrappers
  (`extract_existing_eval_results.sh`, `extract_fused_manual.sh`,
  `extract_malay_manual.sh`, `extract_missing_indonesian_results.sh`,
  `extract_additional_alphas.sh`) wire it up to specific output trees.
- `batch_extract_ablation_results.py` — same idea, batched over many
  `alpha*_lr*` runs, used for the alpha-ablation tables.
- `collect_res.py` — recursive `results_*.json` collector that writes a
  flat CSV (one task per row, no alpha/lr semantics — useful for
  one-off baselines).
- `auto_eval_baseline.py` — compares each experimental config against
  the per-task baseline and outputs the best configuration with the
  largest gain.
- `pick.py` — pandas-based picker that parses `alpha0.X_lrYe-Z`
  patterns out of paths, normalises variant names to
  `{hot, nohot, untrained}`, and selects the best alpha/lr combo per
  variant.

## Typical flow

After running `evaluation/master/eval_all_tasks.sh` (or the per-domain
scripts) over a sweep of alpha / lr:

```bash
# 1. Dump raw numbers per task to CSV
python utils/extract_eval_results.py \
  --root output/llama3-eval \
  --output extracted.csv

# 2. Compare against the baseline and report the best config
python utils/auto_eval_baseline.py \
  --root output/llama3-eval \
  --baseline_name baseline \
  --output best_per_task.csv

# 3. Final paper-ready picker (per-task best alpha/lr, hot vs nohot)
python utils/pick.py \
  --csv extracted.csv \
  --output paper_table.csv
```
