# Medical Evaluation

Evaluates the fused 8B model (after cross-architecture transport + SFT) on
medical-domain QA via [lm-evaluation-harness].

## Tasks (8 in total)

```
medqa_4options
mmlu_anatomy
medmcqa
mmlu_clinical_knowledge
mmlu_college_biology
mmlu_college_medicine
mmlu_medical_genetics
mmlu_professional_medicine
```

These match the paper's `medical` row.

## Quick run

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval
export HF_DATASETS_TRUST_REMOTE_CODE=1

python -m lm_eval --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=float" \
  --tasks medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine \
  --device cuda:0 \
  --batch_size 1 \
  --output_path output/medical
```

## Files

- `eval_benchmark_medical.sh` — original lab shell wrapper showing the
  exact `lm_eval` invocations used for medical hot/nohot/baseline runs.
- `eval_medical.py` — post-processing utility that scans
  `output/.../results_*.json` from many alpha/lr sweeps, picks the best
  configuration per task, and writes a CSV with hot/nohot/untrained
  side-by-side (also embeds the paper's baseline numbers as a fallback).

## Aggregation

After all runs are done:

```bash
python eval_medical.py \
  --root /data/chenhang/optimal_trans_new/hot_sft_runs_*medical*/ \
  --output medical_summary.csv
```

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
