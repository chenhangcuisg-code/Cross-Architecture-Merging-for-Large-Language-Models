#!/usr/bin/env bash
# Math benchmarks via lm-evaluation-harness
# Default model is the paper's SFT_math-Llama-3.2-1B; override MODEL_PATH env var as needed.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/chenhang/models/SFT_math_Llama-3.2-1B}
OUTPUT_DIR=${OUTPUT_DIR:-output/math}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-cuda:0}
TASKS=${TASKS:-arithmetic_2da,arithmetic_2ds,minerva_math_prealgebra,bigbench_simple_arithmetic_json_generate_until,mmlu_elementary_mathematics}

mkdir -p "$OUTPUT_DIR"
python -m lm_eval --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=float" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_DIR}"
