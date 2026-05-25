#!/usr/bin/env bash
# Evaluate source LLaMA-3.1-8B-Instruct on all paper benchmarks
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval
export PYTHONPATH="/data/chenhang/codes/lm-evaluation-harness:${PYTHONPATH:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lm-eval

MODEL="/data/chenhang/models/Llama-3.1-8B-Instruct"
LM_EVAL_REPO="/data/chenhang/codes/lm-evaluation-harness"
OUT_BASE="${LM_EVAL_REPO}/output_source_8b_eval"
LOG="/home/chenhang/optimal_trans/eval_source_8b.log"

mkdir -p "$OUT_BASE"
echo "=== Source 8B Eval Started: $(date) ===" | tee -a "$LOG"

cd "$LM_EVAL_REPO"

# 1. Indonesian benchmarks
echo "=== Indonesian ===" | tee -a "$LOG"
python -m lm_eval --model hf \
  --model_args pretrained="${MODEL}",dtype=float \
  --tasks arc_id,belebele_ind_Latn,truthfulqa_id_mc1,truthfulqa_id_mc2,xcopa_id \
  --batch_size 8 --output_path "${OUT_BASE}/indonesian" \
  2>&1 | tee -a "$LOG"

# 2. Thai benchmarks
echo "=== Thai ===" | tee -a "$LOG"
python -m lm_eval --model hf \
  --model_args pretrained="${MODEL}",dtype=float \
  --tasks xcopa_th,mgsm_direct_th,mgsm_native_cot_th \
  --batch_size 4 --output_path "${OUT_BASE}/thai" \
  2>&1 | tee -a "$LOG"

# 3. Medical benchmarks (MMLU subsets)
echo "=== Medical ===" | tee -a "$LOG"
python -m lm_eval --model hf \
  --model_args pretrained="${MODEL}",dtype=float \
  --tasks anatomy,medical_genetics,professional_medicine \
  --batch_size 8 --output_path "${OUT_BASE}/medical" \
  2>&1 | tee -a "$LOG"

# 4. Finance benchmarks (MMLU subsets)
echo "=== Finance ===" | tee -a "$LOG"
python -m lm_eval --model hf \
  --model_args pretrained="${MODEL}",dtype=float \
  --tasks "global_mmlu_full_en_business_ethics,global_mmlu_full_en_high_school_microeconomics,global_mmlu_full_en_professional_accounting" \
  --batch_size 8 --output_path "${OUT_BASE}/finance" \
  2>&1 | tee -a "$LOG"

echo "=== Source 8B Eval Done: $(date) ===" | tee -a "$LOG"
