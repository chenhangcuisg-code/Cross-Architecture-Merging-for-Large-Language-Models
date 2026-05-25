# Indonesian Evaluation

Two pipelines:

1. **Reasoning / cloze tasks** via [lm-evaluation-harness] — paper row.
2. **IndoMMLU** (subject-level Indonesian knowledge benchmark, ported from
   [yahma/IndoMMLU](https://github.com/yahma/IndoMMLU)).

## Tasks (lm-eval, paper row)

```
belebele_ind_Latn
xcopa_id
arc_id
xstorycloze_id
truthfulqa_id_mc1
truthfulqa_id_mc2
```

## Quick run (lm-eval)

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"

python -m lm_eval --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=float16,load_in_4bit=False,load_in_8bit=False" \
  --tasks belebele_ind_Latn,xcopa_id,arc_id,xstorycloze_id,truthfulqa_id_mc1,truthfulqa_id_mc2 \
  --device cuda:0 \
  --batch_size 1 \
  --output_path output/indonesian
```

`eval_indo.sh` is the original lab wrapper for hot / nohot / merged runs.

## IndoMMLU pipeline

```bash
cd IndoMMLU
pip install -r requirements.txt

# Generate predictions
python evaluate.py \
  --model "${MODEL_PATH}" \
  --output_dir output/IndoMMLU/${MODEL_NAME}

# Aggregate per-subject accuracies
python get_res.py --result_dir output/IndoMMLU/${MODEL_NAME}
```

`evaluate.sh` and `run.sh` show the lab's batched conventions
(per-subject parallelism + LoRA loading).

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
