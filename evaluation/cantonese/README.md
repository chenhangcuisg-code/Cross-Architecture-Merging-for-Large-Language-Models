# Cantonese (CMMLU / Yue-MMLU) Evaluation

Two-stage: prediction generation, then per-subject accuracy aggregation.
Built on top of [Yue-Benchmark] data
(`Yue-Benchmark/data/latest_data/Yue-MMLU`).

## Files

- `generate_cmmlu_predictions.py` — generate one JSON per CMMLU subject
  (0-shot or 5-shot, configurable dtype/device/max_new_tokens).
- `generate_cmmlu_predictions_resume.py` — same as above but resumes from
  partial output (useful for long sweeps).
- `evaluate_cmmlu_yue.py` — aggregate predictions into a summary CSV +
  JSON, plus an error file.
- `eval_cmmlu.sh` — driver script that runs predict + evaluate end-to-end
  and supports a configurable `MODEL_A_PATH` / `MODEL_B_PATH` pair (used
  for paired comparison runs in the paper).
- `aggregate_cmmlu_results.py`, `analyze_cmmlu_results.py`,
  `create_cmmlu_table.py`, `show_cmmlu_table.py` — turn raw predictions
  across many alpha/lr settings into the appendix tables in the paper.

## Quick run

```bash
git clone https://github.com/jiangjyjy/Yue-Benchmark
export YUE_BENCHMARK_ROOT=$(pwd)/Yue-Benchmark

PRED_DIR=output/cmmlu/${MODEL_NAME}/0shot
EVAL_DIR=output/cmmlu/${MODEL_NAME}/eval
mkdir -p "${PRED_DIR}" "${EVAL_DIR}/modelA"

python generate_cmmlu_predictions.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${YUE_BENCHMARK_ROOT}/data/latest_data/Yue-MMLU" \
  --output_dir "${PRED_DIR}" \
  --num_shots 0 \
  --device cuda:0 \
  --dtype float16 \
  --max_new_tokens 512

cp "${PRED_DIR}"/*.json "${EVAL_DIR}/modelA/"

python evaluate_cmmlu_yue.py \
  --predictions_dir "$(dirname ${EVAL_DIR})" \
  --output_dir "${EVAL_DIR}"
```

Outputs:

- `${EVAL_DIR}/cmmlu_yue_results.csv` — per-subject accuracy
- `${EVAL_DIR}/cmmlu_yue_results.json` — full result dict
- `${EVAL_DIR}/cmmlu_yue_errors.json` — wrongly-classified examples for
  qualitative analysis

## Aggregating across alpha / lr sweeps

```bash
python aggregate_cmmlu_results.py --root output/cmmlu --output cmmlu_summary.csv
python show_cmmlu_table.py --csv cmmlu_summary.csv
```

[Yue-Benchmark]: https://github.com/jiangjyjy/Yue-Benchmark
