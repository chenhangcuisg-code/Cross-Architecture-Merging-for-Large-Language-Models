# General Ability Evaluation

Sanity-check that cross-architecture transport + SFT does **not** harm the
model's general English reasoning ability. Used for the appendix table
"general ability" in the paper.

## Tasks

```
arc_easy
commonsense_qa
piqa
social_iqa
winogrande
```

## Files

- `eval_general_ability.sh` — full lab driver. Supports four variants
  (`hot`, `nohot`, `frozen_base`, `baseline`) and overridable model paths
  via `MODEL_*_OVERRIDE` env vars; appends a result block to
  `eval_general_ability.log`.
- `eval_general_ability_simple.sh` — minimal one-shot lm-eval invocation
  (useful for the source-8B / single-baseline numbers).
- `eval_benchmark_general.sh` — alternate lm-eval setup focused on
  `ifeval` + `mmlu` (used for general capability checks).
- `eval_benchmark_gen.sh` — variant launching all four ablation models
  on the medical task list (cross-domain transfer check).
- `README_zh_legacy.md` — original (Chinese) usage notes shipped with
  `eval_general_ability.sh`.

## Quick run

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval

bash eval_general_ability.sh
```

The script will:

1. Run `winogrande,social_iqa,arc_easy,piqa,commonsense_qa` with
   `lm-evaluation-harness` for each variant.
2. Skip variants whose output directory already has `results*.json`.
3. Append a Markdown-formatted summary block to
   `eval_general_ability.log`.

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
