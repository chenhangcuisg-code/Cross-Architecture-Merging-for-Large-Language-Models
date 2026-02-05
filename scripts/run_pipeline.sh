#!/usr/bin/env bash
# Example pipeline script for hierarchical optimal transport
# This script demonstrates the complete workflow:
#   1) Extract activations and compute transport plans (Q/P)
#   2) Fuse models using transport plans
#   3) Train with transport-based residual

set -euo pipefail

#############################################
# Configuration
#############################################

# Model paths (update these with your model paths)
MODEL_A_PATH="<path_to_source_model>"
MODEL_B_PATH="<path_to_target_model>"

# Data configuration
DATA_SUBSET="eng"
DATA_SPLIT="train"
MAX_SAMPLES=5000

# Transport plan computation parameters
TRANSPORT_DIR="./transport_results"
BATCH_SIZE_A=2
BATCH_SIZE_B=2
DEVICE_A="cuda:0"
DEVICE_B="cuda:0"

# Fusion parameters
ALPHA=0.1
FUSED_MODEL_DIR="./fused_model"

# Training parameters
OUTPUT_DIR="./training_output"
MODEL_TYPE="llama"  # Options: llama, qwen2, qwen2vl, tinyllava

#############################################
# Step 1: Extract activations and compute transport plans
#############################################

echo "=================================================="
echo "[Step 1] Extract activations and compute transport plans"
echo "=================================================="

python run_activs_and_hot.py \
    --model-a-path "${MODEL_A_PATH}" \
    --model-b-path "${MODEL_B_PATH}" \
    --batch-size-a "${BATCH_SIZE_A}" \
    --batch-size-b "${BATCH_SIZE_B}" \
    --device-a "${DEVICE_A}" \
    --device-b "${DEVICE_B}" \
    --data-subset "${DATA_SUBSET}" \
    --data-split "${DATA_SPLIT}" \
    --max-samples "${MAX_SAMPLES}" \
    --out-dir "${TRANSPORT_DIR}"

if [ ! -d "${TRANSPORT_DIR}" ]; then
    echo "[ERROR] TRANSPORT_DIR=${TRANSPORT_DIR} does not exist." >&2
    exit 1
fi

#############################################
# Step 2: Fuse models using transport plans
#############################################

echo "=================================================="
echo "[Step 2] Fuse models using transport plans"
echo "=================================================="

python generate_hot_residual.py \
    --modelA_id "${MODEL_B_PATH}" \
    --modelB_id "${MODEL_A_PATH}" \
    --hot_dir "${TRANSPORT_DIR}" \
    --alpha "${ALPHA}" \
    --output_dir "${FUSED_MODEL_DIR}"

#############################################
# Step 3: Train with transport-based residual
#############################################

echo "=================================================="
echo "[Step 3] Train with transport-based residual"
echo "=================================================="

python train_hot_residual_sft.py \
    --model_dir "${FUSED_MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --training_scenario hot \
    --freeze_strategy frozen_hot

echo "[Done] Pipeline completed successfully."
