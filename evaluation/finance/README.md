# Finance Evaluation

Two complementary pipelines:

1. **Multilingual / classification benchmarks** via
   [lm-evaluation-harness] — used for the paper's `finance` row.
2. **Generative QA** on
   [AfterQuery/FinanceQA](https://huggingface.co/datasets/AfterQuery/FinanceQA)
   via direct Hugging Face inference + token-F1 / exact match.

## Tasks (lm-eval, paper row)

```
global_piqa_completions_fin_latn
global_piqa_prompted_fin_latn
belebele_fin_Latn
multiblimp_fin
financial_tweets
```

The source-8B baseline additionally uses MMLU finance subsets
(`global_mmlu_full_en_business_ethics`,
`global_mmlu_full_en_high_school_microeconomics`,
`global_mmlu_full_en_professional_accounting`) — see
`evaluation/master/run_source8b_eval.sh`.

## Quick run (lm-eval)

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"

python -m lm_eval --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=float" \
  --tasks global_piqa_completions_fin_latn,global_piqa_prompted_fin_latn,belebele_fin_Latn,multiblimp_fin,financial_tweets \
  --device cuda:0 \
  --batch_size 1 \
  --output_path output/finance
```

## Files

- `eval_hf_finqa.py` — generative FinanceQA evaluator. Loads
  `AfterQuery/FinanceQA`, generates with `transformers.pipeline`, computes
  EM and token-level F1. Use `--save_predictions out.json` to keep
  generations.

  ```bash
  python eval_hf_finqa.py \
    --model "${MODEL_PATH}" \
    --max_samples 200 \
    --save_predictions output/finance/finqa_pred.json
  ```

- `fin_eval.py` — post-processing tool that scans
  `output_fin/**/pred.json` and computes accuracy with regex-based answer
  extraction (handles `$32,095 (in millions)` style targets).

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
