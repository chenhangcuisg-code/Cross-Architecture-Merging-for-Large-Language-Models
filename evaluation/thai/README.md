# Thai Evaluation

Evaluates the fused model on Thai-language tasks via
[lm-evaluation-harness].

## Tasks

The paper's `thai` row uses:

```
xcopa_th
xquad_th
xnli_th
```

`run_source8b_eval.sh` (in `evaluation/master/`) additionally runs
`mgsm_direct_th` and `mgsm_native_cot_th` for the source-8B baseline.

## Quick run

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval

python -m lm_eval --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=float" \
  --tasks xcopa_th,xquad_th,xnli_th \
  --device cuda:0 \
  --batch_size 1 \
  --output_path output/thai
```

## Files

- `eval_benchmark_thai.sh` — original lab shell wrapper showing the
  hot/nohot/ablation runs for Thai. Includes the
  `mgsm_direct_th,mmlu_prox_lite_th` variant used in the appendix.

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
