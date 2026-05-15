#!/usr/bin/env bash
# Minimal end-to-end example: fuse a small (Model A) and a large (Model B)
# checkpoint, then SFT the fused model. Defaults below reproduce the paper's
# **malay** task; override any var to switch task. The full 6-task pipeline
# lives in run_train_final.sh.
#
# See MODELS.md for the per-task HF repos and α / lr values.
# See REPRODUCE.md for the end-to-end walk-through.
#
# Usage:
#   bash scripts/run_pipeline.sh                                # malay default
#   MODEL_A=PathFinderKR/Llama-3-1B-Medical-Instruct \
#   MODEL_B=unsloth/Llama-3.1-8B-Instruct DATA_SUBSET=medical \
#   ALPHA_FUSE=0.03 ALPHA_TRAIN=0.005 LR=3e-7 \
#       bash scripts/run_pipeline.sh                            # medical
#
# Either pass a Hugging Face repo id (downloaded on demand) or a local path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

#############################################
# Configuration (defaults = paper "malay" task)
#############################################

MODEL_A="${MODEL_A:-mesolitica/Malaysian-Llama-3.2-1B-Instruct}"   # 1B donor
MODEL_B="${MODEL_B:-unsloth/Llama-3.1-8B-Instruct}"                # 8B base

DATA_SUBSET="${DATA_SUBSET:-malay}"
DATA_SPLIT="${DATA_SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"

TRANSPORT_DIR="${TRANSPORT_DIR:-${REPO_ROOT}/transport_results/hot_results_demo}"
FUSED_MODEL_DIR="${FUSED_MODEL_DIR:-${REPO_ROOT}/models/demo_fused}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/sft_runs_demo}"

ALPHA_FUSE="${ALPHA_FUSE:-0.1}"      # Step 2 fusion strength
ALPHA_TRAIN="${ALPHA_TRAIN:-0.1}"    # Step 3 training residual scale
LR="${LR:-1e-6}"                     # Step 3 learning rate

MODEL_TYPE="${MODEL_TYPE:-llama}"
DATASET_TYPE="${DATASET_TYPE:-malaysian_sft}"

BATCH_SIZE_A=2
BATCH_SIZE_B=2
DEVICE_A="${DEVICE_A:-cuda:0}"
DEVICE_B="${DEVICE_B:-cuda:0}"

#############################################
# Step 1: Extract activations and compute transport plans
#############################################
echo "=================================================="
echo "[Step 1] Activations + transport plans"
echo "  Model A (donor): ${MODEL_A}"
echo "  Model B (base):  ${MODEL_B}"
echo "  Data: ${DATA_SUBSET}/${DATA_SPLIT} max=${MAX_SAMPLES}"
echo "=================================================="

python run_activs_and_hot.py \
    --model-a-path "${MODEL_A}" \
    --model-b-path "${MODEL_B}" \
    --batch-size-a "${BATCH_SIZE_A}" \
    --batch-size-b "${BATCH_SIZE_B}" \
    --device-a "${DEVICE_A}" \
    --device-b "${DEVICE_B}" \
    --data-subset "${DATA_SUBSET}" \
    --data-split "${DATA_SPLIT}" \
    --max-samples "${MAX_SAMPLES}" \
    --hot-chunk-cols 1024 --hot-dtype float32 \
    --out-dir "${TRANSPORT_DIR}"

#############################################
# Step 2: Fuse models using transport plans
#############################################
echo "=================================================="
echo "[Step 2] Fuse Model A with the transport residual (α=${ALPHA_FUSE})"
echo "=================================================="

python generate_hot_residual.py \
    --modelA_id "${MODEL_A}" \
    --modelB_id "${MODEL_B}" \
    --hot_dir "${TRANSPORT_DIR}" \
    --alpha "${ALPHA_FUSE}" \
    --lm_only --verbose \
    --attn_device "${DEVICE_A}" --attn_max_mem_mb 1200 \
    --output_dir "${FUSED_MODEL_DIR}"

#############################################
# Step 3: Train with transport-based residual
#############################################
echo "=================================================="
echo "[Step 3] SFT (HOT, α=${ALPHA_TRAIN}, lr=${LR})"
echo "=================================================="

python train_hot_residual_sft.py \
    --training_scenario hot \
    --save_untrained_folded \
    --model_dir "${FUSED_MODEL_DIR}" \
    --tokenizer_dir "${MODEL_A}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --dataset_type "${DATASET_TYPE}" \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate "${LR}" \
    --num_train_epochs 1 \
    --block_size 2048 \
    --alpha "${ALPHA_TRAIN}" \
    --fp16

echo "[Done] Pipeline completed. Fused model -> ${FUSED_MODEL_DIR}; SFT -> ${OUTPUT_DIR}"
