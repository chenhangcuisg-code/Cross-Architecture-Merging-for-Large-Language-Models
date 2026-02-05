#!/usr/bin/env bash

# Hugging Face download configuration: increased timeout and retry counts
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_RETRIES=10
export HF_HUB_DOWNLOAD_RETRY_DELAY=5
export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN:-$HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

#############################################################
#                 Distributed Training and Evaluation Guide
#############################################################
# The script supports distributed training and evaluation.
# Training uses torchrun, evaluation uses multi-GPU acceleration.
#
# Usage:
#   1. Single GPU (compatible with old method):
#      CUDA_VISIBLE_DEVICES=3 NUM_GPUS=1 ./trans_train_final_merged.sh
#
#   2. Multi-GPU distributed (recommended):
#      CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 ./trans_train_final_merged.sh
#
#   3. Custom port (to avoid port conflicts):
#      CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 MASTER_PORT=29501 ./trans_train_final_merged.sh
#
#   4. Custom run label (for distinguishing different experiments, recommended for long runs):
#      RUN_LABEL=exp_v1 ./trans_train_final_merged.sh
#      If not set, timestamp will be used automatically as label
#############################################################

#############################################################
#                 Configuration Switches (true/false)
#############################################################

RUN_STEP1=true     # Step1 Activation extraction + Transport plan computation
RUN_STEP2=true     # Step2 Transport plan-based fusion
RUN_STEP3=true      # Step3 Training + Dual evaluation (original benchmark + ablation benchmark)
RUN_STEP4=false     # Step4 Now just a hint, actual eval has been done in Step3

#############################################################
#                 Run Label Configuration
#############################################################

# Run label (for distinguishing different experiment runs)
# Can be set via environment variable: RUN_LABEL=my_experiment bash trans_train_final_merged.sh
# If not set, final_nolora will be used as default value
RUN_LABEL=${RUN_LABEL:-final_paper_mala}

#############################################################
#                 Evaluation Mode Switches
#############################################################

# Dual evaluation mode:
#   ENABLE_ORIGINAL_EVAL=true  - Run original benchmark evaluation (each task uses its own evaluation method)
#   ENABLE_ABLATION_EVAL=true  - Run ablation benchmark evaluation (unified general benchmark)
# Both can be enabled simultaneously without conflict
ENABLE_ORIGINAL_EVAL=true
ENABLE_ABLATION_EVAL=true

#############################################################
#                 Common Configuration
#############################################################

ACTIVS_SCRIPT="run_activs_and_hot.py"
FUSE_SCRIPT="generate_hot_residual.py"
TRAIN_SCRIPT="train_hot_residual_sft.py"

# Common evaluation configuration
CLEANUP_AFTER_EVAL=true
FORCE_CLEANUP_AFTER_EVAL=false
IGNORE_EVAL_LOG_CHECK=${IGNORE_EVAL_LOG_CHECK:-true}
TRAIN_LAUNCH=${TRAIN_LAUNCH:-torchrun}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_RESULTS_LOG="${SCRIPT_DIR}/eval_results_summary_merged_final.log"

#############################################################
#                 Workspace Root Paths Configuration
#############################################################
# These paths can be set via environment variables or will use defaults
# Default WORKSPACE_ROOT is the parent directory of the scripts directory
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-${WORKSPACE_ROOT}/models}"
HOT_RESULTS_ROOT="${HOT_RESULTS_ROOT:-${WORKSPACE_ROOT}/transport_results}"

# Ablation evaluation configuration (unified general benchmark)
ABLATION_EVAL_TASKS="winogrande,social_iqa,arc_easy,piqa,commonsense_qa"
# ABLATION_EVAL_OUT_ROOT will be dynamically set in STEP 3 based on timestamp
ABLATION_EVAL_BATCH_SIZE=8

#############################################################
#                 Evaluation Library Configuration
#############################################################
# This script uses three evaluation libraries:
#
# 1. lm-evaluation-harness (for most tasks)
#    Repository: https://github.com/EleutherAI/lm-evaluation-harness
#    Used for: Medical, Thai, Finance, Indonesian tasks
#    Installation: Follow the repository README
#
# 2. MalayMMLU (for Malay language evaluation)
#    Repository: https://github.com/UMxYTL-AI-Labs/MalayMMLU
#    Used for: Malay task evaluation
#    Installation: Clone the repository and follow its setup instructions
#
# 3. Yue-Benchmark (for Cantonese evaluation)
#    Repository: https://github.com/jiangjyjy/Yue-Benchmark
#    Used for: Cantonese (CMMLU) task evaluation
#    Installation: Clone the repository and follow its setup instructions
#
# Set these paths via environment variables or modify the defaults below:
#############################################################

# lm-evaluation-harness repository path
# Default: assumes repository is cloned at WORKSPACE_ROOT/lm-evaluation-harness
# Override with: LM_EVAL_REPO=/path/to/lm-evaluation-harness
LM_EVAL_REPO="${LM_EVAL_REPO:-${WORKSPACE_ROOT}/lm-evaluation-harness}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_REPO}/output_nolora_${RUN_LABEL}"

# MalayMMLU repository path
# Default: assumes repository is cloned at WORKSPACE_ROOT/MalayMMLU
# Override with: MALAY_REPO=/path/to/MalayMMLU
MALAY_REPO="${MALAY_REPO:-${WORKSPACE_ROOT}/MalayMMLU}"
MALAY_OUTPUT_DIR="${MALAY_REPO}/output"

# Yue-Benchmark repository path (for Cantonese evaluation)
# Default: assumes repository is cloned at WORKSPACE_ROOT/Yue-Benchmark
# Override with: YUE_BENCHMARK_ROOT=/path/to/Yue-Benchmark
YUE_BENCHMARK_ROOT="${YUE_BENCHMARK_ROOT:-${WORKSPACE_ROOT}/Yue-Benchmark}"

# Training params
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRAD_ACC_STEPS=8
NUM_EPOCHS=1
BLOCK_SIZE=2048

# FineWeb Thai cache (path must be ASCII)
FINEWEB_THAI_DATASET_ID="ChavyvAkvar/fineweb-2-1M-Sample-Thai"
FINEWEB_THAI_CACHE_DIR="${FINEWEB_THAI_CACHE_DIR:-${WORKSPACE_ROOT}/hf_cache_eval/fineweb_thai_cli}"

# Distributed
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  VISIBLE_GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
  NUM_GPUS=${NUM_GPUS:-${VISIBLE_GPU_COUNT}}
else
  if command -v nvidia-smi &> /dev/null; then
    SYSTEM_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    NUM_GPUS=${NUM_GPUS:-${SYSTEM_GPU_COUNT}}
  else
    NUM_GPUS=${NUM_GPUS:-4}
  fi
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-${NUM_GPUS}}

# Auto-detect available port
find_available_port() {
  local start_port=${1:-29500}
  local port=$start_port
  local max_attempts=10
  
  for i in $(seq 0 $max_attempts); do
    port=$((start_port + i))
    if command -v nc >/dev/null 2>&1; then
      if ! nc -z localhost $port 2>/dev/null; then
        echo $port
        return 0
      fi
    elif command -v timeout >/dev/null 2>&1; then
      if ! timeout 0.1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
        echo $port
        return 0
      fi
    else
      if ! (ss -tln 2>/dev/null | grep -q ":$port ") && ! (netstat -tln 2>/dev/null | grep -q ":$port "); then
        echo $port
        return 0
      fi
    fi
  done
  
  echo $((29500 + RANDOM % 100))
}

MASTER_PORT=${MASTER_PORT:-$(find_available_port 29500)}

echo ">>> [Distributed Config] NUM_GPUS=${NUM_GPUS}, NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER_PORT=${MASTER_PORT}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo ">>> [Distributed Config] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Common fusion parameters
BATCH_SIZE_A=2
BATCH_SIZE_B=2
DEVICE_A="cuda:0"
DEVICE_B="cuda:0"
HOT_CHUNK_COLS=1024
HOT_DTYPE="float32"
LM_ONLY=true
ATTN_DEVICE="cuda:0"
ATTN_MAX_MEM_MB=1200
ATTN_AUTocast=true

#############################################################
#                 Training Type Switches (global control)
#############################################################

ENABLE_HOT=true           # Transport-based (frozen_transport)
ENABLE_NOHOT=true         # No transport (ablation)
ENABLE_FROZEN_BASE=false  # frozen_base
ENABLE_NONE=false         # none

#############################################################
#                 Task Configuration Functions (each task has its own config)
#############################################################

# Configure task: medical
config_medical() {
  TASK_NAME="medical"
  MODEL_A="${MODELS_ROOT}/llama3-1b-med"
  MODEL_B="${MODELS_ROOT}/Llama-3.1-8B-Instruct"
  HOT_DIR="${HOT_RESULTS_ROOT}/hot_results_medllama_llama_norm"
  FUSED_DIR="${MODELS_ROOT}/medllama_fused_alpha01_fortrain_1b"
  DATA_SUBSET="medical"
  TOP_NEURON_DIR_A="${WORKSPACE_ROOT}/top_neurons_medllama"
  HOT_NEURON_DIR="${WORKSPACE_ROOT}/top_neurons_medllama"
  MODEL_TYPE="llama"
  DATASET_TYPE="medical_llama3"
  GEO_SPLIT="train"
  OUTPUT_ROOT="${WORKSPACE_ROOT}/hot_sft_runs_medical_llama3_1b"
  ALPHA_SEARCH_LIST=(0.005)
  LR_SEARCH_LIST=(3e-7)
  MAX_SAMPLES=2000
  FUSE_ALPHA=0.03
  # Original evaluation configuration
  EVAL_TYPE="lm_eval"
  EVAL_TASKS_ORI="medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine"
  EVAL_OUT_ROOT_ORI="${LM_EVAL_OUTPUT_DIR}/llama3-eval/${MODEL_TYPE}_${DATASET_TYPE}_${GEO_SPLIT}_fixed"
  EVAL_BATCH_SIZE_ORI=8
}

# Configure task: thai
config_thai() {
  TASK_NAME="thai"
  MODEL_A="${MODELS_ROOT}/llama3.2-typhoon2-1b-instruct"
  MODEL_B="${MODELS_ROOT}/Llama-3.1-8B-Instruct"
  HOT_DIR="${HOT_RESULTS_ROOT}/hot_results_llamathai_llama_norm"
  FUSED_DIR="${MODELS_ROOT}/llamathai_fused_alpha01_fortrain_1b_thai_instruction_sft"
  DATA_SUBSET="fineweb_thai"
  TOP_NEURON_DIR_A="${WORKSPACE_ROOT}/top_neurons_thai"
  HOT_NEURON_DIR="${WORKSPACE_ROOT}/top_neurons_thai"
  MODEL_TYPE="llama"
  DATASET_TYPE="fineweb_thai"
  GEO_SPLIT="train"
  OUTPUT_ROOT="${WORKSPACE_ROOT}/hot_sft_runs_llamathai_1b_fineweb_thai_instruction_sft"
  ALPHA_SEARCH_LIST=(0.01)
  LR_SEARCH_LIST=(1e-7)
  MAX_SAMPLES=8000
  FUSE_ALPHA=0.01
  EVAL_TYPE="lm_eval"
  EVAL_TASKS_ORI="belebele_tha_Thai,mgsm_direct_th,mgsm_native_cot_th,xcopa_th,mmlu_prox_lite_th_other,xquad_th,xnli_th"
  EVAL_OUT_ROOT_ORI="${LM_EVAL_OUTPUT_DIR}/llama3.2-1bth-instruct-fixed"
  EVAL_BATCH_SIZE_ORI=8
  LOCAL_DATASET_PATH="${FINEWEB_THAI_CACHE_DIR}"
}

# Configure task: finance
config_finance() {
  TASK_NAME="finance"
  MODEL_A="${MODELS_ROOT}/Llama-3.2-1B-Instruct"
  MODEL_B="${MODELS_ROOT}/Llama-3.1-8B-Instruct"
  HOT_DIR="${HOT_RESULTS_ROOT}/hot_results_finllama_llama"
  FUSED_DIR="${MODELS_ROOT}/finllama_fused_alpha01_fortrain_1b"
  DATA_SUBSET="finance"
  TOP_NEURON_DIR_A="${WORKSPACE_ROOT}/top_neurons_finllama"
  HOT_NEURON_DIR="${WORKSPACE_ROOT}/top_neurons_finllama"
  MODEL_TYPE="llama"
  DATASET_TYPE="finance"
  GEO_SPLIT="train"
  OUTPUT_ROOT="${WORKSPACE_ROOT}/hot_sft_runs_finical_llama3_1b"
  ALPHA_SEARCH_LIST=(0.03)
  LR_SEARCH_LIST=(1e-6)
  MAX_SAMPLES=2000
  FUSE_ALPHA=0.03
  EVAL_TYPE="lm_eval"
  EVAL_TASKS_ORI="financial_tweets,belebele_fin_Latn,multiblimp_fin,global_piqa_completions_fin_latn,global_piqa_prompted_fin_latn"
  EVAL_OUT_ROOT_ORI="${LM_EVAL_OUTPUT_DIR}/llama3-eval/${MODEL_TYPE}_${DATASET_TYPE}_${GEO_SPLIT}_fixed"
  EVAL_BATCH_SIZE_ORI=8
}

# Configure task: cantonese
config_cantonese() {
  TASK_NAME="cantonese"
  MODEL_A="${MODELS_ROOT}/Llama-3.2-1B-Instruct"
  MODEL_B="${MODELS_ROOT}/Llama3-Chinese-8B-Instruct"
  HOT_DIR="${HOT_RESULTS_ROOT}/hot_results_llamacan_llama"
  FUSED_DIR="${MODELS_ROOT}/llamacan_fused_alpha01_fortrain_1b"
  DATA_SUBSET="cantonese"
  TOP_NEURON_DIR_A="${WORKSPACE_ROOT}/top_neurons_can"
  HOT_NEURON_DIR="${WORKSPACE_ROOT}/top_neurons_can"
  MODEL_TYPE="llama"
  DATASET_TYPE="cantonese"
  GEO_SPLIT="train"
  OUTPUT_ROOT="${WORKSPACE_ROOT}/hot_sft_runs_llamacan_1b"
  ALPHA_SEARCH_LIST=(0.1)
  LR_SEARCH_LIST=(1e-8)
  MAX_SAMPLES=2000
  FUSE_ALPHA=0.01
  EVAL_TYPE="cmmlu"
  EVAL_OUT_ROOT_ORI="${YUE_BENCHMARK_ROOT}/output_llama3.2-1bcan-instruct-conversation"
  CMMLU_DATA_DIR="${YUE_BENCHMARK_ROOT}/data/latest_data/Yue-MMLU"
  CMMLU_PREDICT_SCRIPT="${WORKSPACE_ROOT}/evaluation/generate_cmmlu_predictions.py"
  CMMLU_EVAL_SCRIPT_LOCAL="${WORKSPACE_ROOT}/evaluation/evaluate_cmmlu_yue.py"
}

# Configure task: indonesian
config_indonesian() {
  TASK_NAME="indonesian"
  MODEL_A="${MODELS_ROOT}/Llama-3.2-1B-Indonesian-QLora"
  MODEL_B="${MODELS_ROOT}/Llama-3.1-8B-Instruct"
  HOT_DIR="${HOT_RESULTS_ROOT}/hot_results_llamaindo_llama_norm"
  FUSED_DIR="${WORKSPACE_ROOT}/indo_llama_fused_alpha01_fortrain_1b"
  DATA_SUBSET="indonesian"
  TOP_NEURON_DIR_A="${WORKSPACE_ROOT}/top_neurons_llama3_indo_new"
  HOT_NEURON_DIR="${WORKSPACE_ROOT}/top_neurons_llama3_indo_new"
  MODEL_TYPE="llama"
  DATASET_TYPE="indonesian_conversation"
  GEO_SPLIT="train"
  OUTPUT_ROOT="${WORKSPACE_ROOT}/hot_sft_runs_llamaindo_1b"
  ALPHA_SEARCH_LIST=(0.1)
  LR_SEARCH_LIST=(1e-6)
  MAX_SAMPLES=2000
  FUSE_ALPHA=0.01
  EVAL_TYPE="lm_eval"
  EVAL_TASKS_ORI="belebele_ind_Latn,xcopa_id,arc_id,xstorycloze_id,truthfulqa_id_mc1,truthfulqa_id_mc2,hellaswag_id"
  EVAL_OUT_ROOT_ORI="${LM_EVAL_OUTPUT_DIR}/llama3.2-1bid-instruct-fixed"
  EVAL_BATCH_SIZE_ORI=8
}

# Configure task: malay
config_malay() {
  TASK_NAME="malay"
  MODEL_A="${MODELS_ROOT}/Malaysian-Llama-3.2-1B-Instruct-v0.1"
  MODEL_B="${MODELS_ROOT}/Llama-3.1-8B-Instruct"
  HOT_DIR="${HOT_RESULTS_ROOT}/hot_results_llamamala_llama_norm"
  FUSED_DIR="${WORKSPACE_ROOT}/maly_llama_fused_alpha01_fortrain_1b_select"
  DATA_SUBSET="malay"
  DATASET_TYPE="malaysian_sft"
  TOP_NEURON_DIR_A="${WORKSPACE_ROOT}/top_neurons_llama3_mala_new"
  HOT_NEURON_DIR="${WORKSPACE_ROOT}/top_neurons_llama3_mala_new"
  MODEL_TYPE="llama"
  GEO_SPLIT="train"
  OUTPUT_ROOT="${WORKSPACE_ROOT}/hot_sft_runs_llamamala_1b"
  ALPHA_SEARCH_LIST=(0.1)
  LR_SEARCH_LIST=(1e-6)
  MAX_SAMPLES=2000
  FUSE_ALPHA=0.1
  EVAL_TYPE="malay_mmlu"
  EVAL_OUT_ROOT_ORI="${MALAY_OUTPUT_DIR}/llama3.2-1bma-instruct-fixed-nolora"
  MALAY_TOKEN="${MALAY_HF_TOKEN:-}"
  MALAY_SHOT=0
}

#############################################################
#                 Task List (tasks to be executed)
#############################################################

# Six tasks to reproduce the paper: medical, thai, finance, cantonese, indonesian, malay
TASK_CONFIGS=(
  "config_medical"
  "config_thai"
  "config_finance"
  "config_cantonese"
  "config_indonesian"
  "config_malay"
)

#############################################
#                 Check Required Scripts
#############################################

echo ">>> Checking if required scripts exist..."
for f in "${ACTIVS_SCRIPT}" "${FUSE_SCRIPT}" "${TRAIN_SCRIPT}"; do
  [ -f "$f" ] || { echo "[ERROR] Script not found: $f"; exit 1; }
done

# Initialize evaluation results log file
if [ "$RUN_STEP3" = true ]; then
  if [ ! -f "$EVAL_RESULTS_LOG" ]; then
    {
      echo "=========================================="
      echo "Evaluation Results Summary (Merged)"
      echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo ""
    } > "$EVAL_RESULTS_LOG"
  else
    {
      echo ""
      echo "=========================================="
      echo "New Session Started at: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo ""
    } >> "$EVAL_RESULTS_LOG"
  fi
fi

#############################################
#          Resume Helper Functions
#############################################

dir_has_files () {
  local d="$1"
  if [ -d "$d" ]; then
    if find "$d" -maxdepth 1 -type f 2>/dev/null | grep -q .; then
      return 0
    fi
  fi
  return 1
}

ensure_fineweb_thai_cache () {
  if [ "${DATASET_TYPE}" != "fineweb_thai" ]; then
    return
  fi

  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[WARN] huggingface-cli not installed, cannot pre-download FineWeb Thai"
    return
  fi

  local target_dir="${FINEWEB_THAI_CACHE_DIR}"
  mkdir -p "$target_dir"

  local shard_count
  shard_count=$(find "${target_dir}" -type f -name "train-*-of-00018.parquet" 2>/dev/null | wc -l | tr -d ' ')

  if [ "${shard_count}" -ge 18 ]; then
    echo ">>> [FineWeb-Thai] Local cache hit (${shard_count}/18) at ${target_dir}"
  else
    echo ">>> [FineWeb-Thai] Pre-downloading 18 parquet files using huggingface-cli download"
    huggingface-cli download \
      --repo-type dataset "${FINEWEB_THAI_DATASET_ID}" \
      --local-dir "${target_dir}" \
      --local-dir-use-symlinks False \
      --resume-download
    shard_count=$(find "${target_dir}" -type f -name "train-*-of-00018.parquet" 2>/dev/null | wc -l | tr -d ' ')
    echo ">>> [FineWeb-Thai] Download complete, current shard count: ${shard_count}/18"
  fi

  export FINEWEB_THAI_LOCAL_DATASET="${target_dir}"
}

train_done () {
  local out_dir=$1
  if [ -f "${out_dir}/_TRAIN_DONE" ]; then
    return 0
  fi
  if [ -d "${out_dir}" ]; then
    if find "${out_dir}" -maxdepth 1 -type f \( -name "pytorch_model*.bin" -o -name "model.safetensors" \) 2>/dev/null | grep -q .; then
      return 0
    fi
  fi
  return 1
}

lm_eval_done () {
  local d="$1"
  find "$d" -maxdepth 5 -type f -name "results*.json" 2>/dev/null | grep -q .
}

eval_logged () {
  local task=$1 run_name=$2 model_variant=$3 eval_type=$4
  local key="Task: ${task} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
  grep -F "$key" "$EVAL_RESULTS_LOG" >/dev/null 2>&1
}

# Original evaluation completed
ori_eval_done () {
  local run_name=$1 model_variant=$2
  local d="${EVAL_OUT_ROOT_ORI}/${run_name}/${model_variant}"
  case "$EVAL_TYPE" in
    lm_eval)
      lm_eval_done "$d"
      ;;
    cmmlu)
      dir_has_files "${d}/cmmlu_eval"
      ;;
    malay_mmlu)
      dir_has_files "$d"
      ;;
    *)
      return 1
      ;;
  esac
}

# Ablation evaluation completed
ablation_eval_done () {
  local run_name=$1 model_variant=$2
  local d="${ABLATION_EVAL_OUT_ROOT}/${run_name}/${model_variant}"
  lm_eval_done "$d"
}

cleanup_model_dir () {
  local model_path=$1
  if [ "${CLEANUP_AFTER_EVAL}" = true ] && [ -d "${model_path}" ]; then
    echo ">>> Deleting model directory: ${model_path}"
    rm -rf "${model_path}"
  fi
}

#############################################
#                 Result Extraction and Logging Functions
#############################################

# Extract lm_eval results table
extract_lm_eval_results() {
  local output_file=$1
  local task_name=$2
  local run_name=$3
  local model_variant=$4
  local model_path=$5
  local out_dir=$6
  local eval_type=$7  # "original" or "ablation"

  [ -f "$output_file" ] || return

  local results
  results=$(awk '
    BEGIN { in_table=0; in_mmlu_groups=0 }
    /Saving results aggregated/ { 
      in_table=1; 
      print; 
      next 
    }
    in_table { 
      print; 
      if (/^$/) { 
        in_table=0;
        next
      }
      next
    }
    /^\|.*Groups.*\|/ { 
      in_mmlu_groups=1; 
      print; 
      next 
    }
    in_mmlu_groups { 
      print; 
      if (/^$/) { 
        in_mmlu_groups=0; 
        exit 
      }
    }
  ' "$output_file" 2>/dev/null || true)

  if [ -n "$results" ]; then
    local key="Task: ${task_name} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
    if ! eval_logged "$task_name" "$run_name" "$model_variant" "$eval_type"; then
      {
        echo "=========================================="
        echo "${key}"
        echo "Model: ${model_path}"
        echo "Output: ${out_dir}"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo "$results"
        echo ""
      } >> "$EVAL_RESULTS_LOG"
    fi
  fi
}

# Extract CMMLU evaluation results (from evaluate_cmmlu_yue.py output and JSON file)
extract_cmmlu_results() {
  local eval_dir=$1
  local task_name=$2
  local run_name=$3
  local model_variant=$4
  local model_path=$5
  local eval_type=$6
  local temp_output_file=$7  # output from evaluate_cmmlu_yue.py

  local key="Task: ${task_name} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
  if eval_logged "$task_name" "$run_name" "$model_variant" "$eval_type"; then
    echo ">>> [SKIP] CMMLU results already logged: ${key}"
    return
  fi

  local results_file="${eval_dir}/cmmlu_yue_results.json"
  local summary=""

  echo ">>> [CMMLU] Extracting evaluation results: ${eval_dir}"

  # First try to extract from evaluate_cmmlu_yue.py print output (if temp output file provided)
  if [ -n "$temp_output_file" ] && [ -f "$temp_output_file" ]; then
    echo ">>> [CMMLU] Trying to extract from temp output file: ${temp_output_file}"
    # Extract evaluation summary from console output (only process 0-shot)
    summary=$(awk '
      /Evaluation Results Summary/ { in_summary=1; next }
      in_summary && /^Model:/ { 
        model=$0
        getline
        if (/0shot:/) {
          shot="0shot"
          getline
          while ($0 !~ /^Model:/ && $0 !~ /^=/) {
            if (/Average Accuracy:/) {
              match($0, /([0-9.]+)/, arr)
              if (arr[1]) print shot " Average Accuracy: " arr[1] "%"
            }
            if (/Total Samples:/) {
              match($0, /([0-9]+)/, arr)
              if (arr[1]) print "Total Samples: " arr[1]
            }
            if (/Total Correct:/) {
              match($0, /([0-9]+)/, arr)
              if (arr[1]) print "Total Correct: " arr[1]
            }
            if (/Total Subjects:/) {
              match($0, /([0-9]+)/, arr)
              if (arr[1]) print "Total Subjects: " arr[1]
              break
            }
            getline
          }
        }
      }
      /^=+$/ && in_summary { exit }
    ' "$temp_output_file" 2>/dev/null || echo "")
    
    if [ -n "$summary" ]; then
      echo ">>> [CMMLU] Successfully extracted results from temp output file"
    else
      echo ">>> [CMMLU] Failed to extract from temp output file, trying from JSON file"
    fi
  fi

  # If extraction from output failed, try from JSON file
  if [ -z "$summary" ] && [ -f "$results_file" ]; then
    echo ">>> [CMMLU] Extracting from JSON file: ${results_file}"
    # Extract model name from model_path (for matching JSON key)
    local model_name=$(basename "$model_path")
    summary=$(python3 <<EOF
import json
import sys
import os

results_file = '${results_file}'
model_name = '${model_name}'
model_variant = '${model_variant}'

if not os.path.exists(results_file):
    print(f"ERROR: JSON file not found: {results_file}", file=sys.stderr)
    sys.exit(1)

try:
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print(f"ERROR: JSON file is empty", file=sys.stderr)
        sys.exit(1)
    
    # Try to match model name
    # 1. Use model_variant directly
    # 2. Use model_name (basename)
    # 3. Use the first available model
    matched_model = None
    
    # Priority: try model_variant first
    if model_variant in data:
        matched_model = model_variant
    # Try model_name
    elif model_name in data:
        matched_model = model_name
    # Try partial match (model_variant contained in some key)
    elif data:
        for key in data.keys():
            if model_variant in key or key in model_variant:
                matched_model = key
                break
        # If still not found, use first model
        if not matched_model:
            matched_model = list(data.keys())[0]
            print(f"WARN: Using first available model: {matched_model}", file=sys.stderr)
    
    if not matched_model or matched_model not in data:
        print(f"ERROR: Cannot match model name", file=sys.stderr)
        print(f"DEBUG: model_variant to match: {model_variant}", file=sys.stderr)
        print(f"DEBUG: model_name to match: {model_name}", file=sys.stderr)
        print(f"DEBUG: Model keys in JSON: {list(data.keys())}", file=sys.stderr)
        sys.exit(1)
    
    # Only process 0-shot (evaluate_cmmlu_yue.py only outputs 0-shot)
    if '0shot' in data[matched_model] and 'summary' in data[matched_model]['0shot']:
        s = data[matched_model]['0shot']['summary']
        print(f'0shot Average Accuracy: {s["average_accuracy"]:.2f}%')
        print(f'Total Samples: {s["total_samples"]}')
        print(f'Total Correct: {s["total_correct"]}')
        print(f'Total Subjects: {s["total_subjects"]}')
    else:
        print(f"ERROR: 0shot summary data not found", file=sys.stderr)
        print(f"DEBUG: Keys for model {matched_model}: {list(data[matched_model].keys())}", file=sys.stderr)
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: Error processing JSON: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
EOF
    )
    
    local python_exit_code=$?
    if [ $python_exit_code -eq 0 ] && [ -n "$summary" ]; then
      echo ">>> [CMMLU] Successfully extracted results from JSON file"
    else
      echo ">>> [CMMLU] Failed to extract from JSON file, Python exit code: ${python_exit_code}"
    fi
  elif [ ! -f "$results_file" ]; then
    echo ">>> [CMMLU] JSON file not found: ${results_file}"
  fi

  if [ -n "$summary" ]; then
    {
      echo "=========================================="
      echo "${key}"
      echo "Model: ${model_path}"
      echo "Output: ${eval_dir}"
      echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo "$summary"
      echo ""
    } >> "$EVAL_RESULTS_LOG"
    echo ">>> [CMMLU] Results written to log"
  else
    echo ">>> [WARN] Failed to extract information from CMMLU evaluation results"
    echo ">>> [WARN] Temp output file: ${temp_output_file:-not found}"
    echo ">>> [WARN] JSON results file: ${results_file:-not found}"
  fi
}

# Extract MalayMMLU evaluation results (using eval_batch.py)
extract_malay_results() {
  local out_dir=$1
  local task_name=$2
  local run_name=$3
  local model_variant=$4
  local model_path=$5
  local eval_type=$6

  local key="Task: ${task_name} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
  if eval_logged "$task_name" "$run_name" "$model_variant" "$eval_type"; then
    return
  fi

  # Find CSV result file
  local csv_file=$(find "$out_dir" -name "MalayMMLU_result_*.csv" -type f | head -1)
  [ -f "$csv_file" ] || {
    echo ">>> [WARN] MalayMMLU CSV result file not found: ${out_dir}"
    return
  }

  echo ">>> [MalayMMLU] Extracting full results using eval_batch.py: ${csv_file}"
  
  # Switch to MalayMMLU directory and run eval_batch.py
  local saved_dir=$(pwd)
  cd "${MALAY_REPO}"
  local temp_output=$(mktemp)
  local temp_result_dir=$(mktemp -d)
  
  # Extract shot count from CSV filename (e.g.: MalayMMLU_result_model_True_0shot.csv -> 0)
  local shot=$(echo "$csv_file" | grep -oP '\d+shot' | grep -oP '\d+' || echo "0")
  
  # Run eval_batch.py to extract full results
  python eval_batch.py \
    --pred_files "$csv_file" \
    --shot "${shot}" \
    --output_dir "${temp_result_dir}" 2>&1 | tee "$temp_output" || {
    echo "[WARN] eval_batch.py execution failed"
    rm -f "$temp_output"
    rm -rf "$temp_result_dir"
    cd "$saved_dir"
    return
  }

  # Extract results from output (including accuracy for each category)
  local results=$(awk '
    /average accuracy/ { 
      match($0, /([0-9.]+)/, arr)
      if (arr[1]) print "Average Accuracy: " arr[1] "%"
    }
    /accuracy for STEM/ {
      match($0, /([0-9.]+)/, arr)
      if (arr[1]) print "STEM Accuracy: " arr[1] "%"
    }
    /accuracy for Language/ {
      match($0, /([0-9.]+)/, arr)
      if (arr[1]) print "Language Accuracy: " arr[1] "%"
    }
    /accuracy for Social science/ {
      match($0, /([0-9.]+)/, arr)
      if (arr[1]) print "Social Science Accuracy: " arr[1] "%"
    }
    /accuracy for Others/ {
      match($0, /([0-9.]+)/, arr)
      if (arr[1]) print "Others Accuracy: " arr[1] "%"
    }
    /accuracy for Humanities/ {
      match($0, /([0-9.]+)/, arr)
      if (arr[1]) print "Humanities Accuracy: " arr[1] "%"
    }
  ' "$temp_output")
  
  rm -f "$temp_output"
  rm -rf "$temp_result_dir"
  cd "$saved_dir"

  if [ -n "$results" ]; then
    {
      echo "=========================================="
      echo "${key}"
      echo "Model: ${model_path}"
      echo "Output: ${out_dir}"
      echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo "$results"
      echo ""
    } >> "$EVAL_RESULTS_LOG"
  else
    echo ">>> [WARN] Failed to extract results from eval_batch.py output"
  fi
}

#############################################
#                 Evaluation Functions
#############################################

# lm_eval evaluation function
run_lm_eval () {
  local model_path=$1 task_name=$2 run_name=$3 model_variant=$4
  local eval_tasks=$5 eval_out_root=$6 eval_batch_size=$7 eval_type=$8

  [ -d "$model_path" ] || { echo "[ERROR] Model dir not found: $model_path"; return 1; }

  local OUTDIR="${eval_out_root}/${run_name}/${model_variant}"
  mkdir -p "$OUTDIR"

  echo ">>> [LM_EVAL ${eval_type}] ${task_name} | ${run_name} | ${model_variant}"
  echo ">>> Model path: ${model_path}"
  echo ">>> Output dir: ${OUTDIR}"

  source ~/miniconda3/etc/profile.d/conda.sh
  cd "${LM_EVAL_REPO}"
  conda activate lm-eval

  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_DATASETS_OFFLINE=0
  export HF_DATASETS_TRUST_REMOTE_CODE=1
  export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/hf_cache_eval}
  mkdir -p "$HF_DATASETS_CACHE"

  local tmp_out=$(mktemp)

  if [ "${NUM_GPUS:-1}" -gt 1 ]; then
    echo ">>> [LM_EVAL] Using multi-GPU mode (${NUM_GPUS} GPUs)"
    accelerate launch -m lm_eval \
      --model hf \
      --model_args pretrained="${model_path}",dtype="float" \
      --tasks "${eval_tasks}" \
      --batch_size "${eval_batch_size}" \
      --output_path "${OUTDIR}" 2>&1 | tee "$tmp_out"
  else
    echo ">>> [LM_EVAL] Using single-GPU mode"
    lm_eval --model hf \
      --model_args pretrained="${model_path}",dtype="float" \
      --tasks "${eval_tasks}" \
      --device cuda:0 \
      --batch_size "${eval_batch_size}" \
      --output_path "${OUTDIR}" 2>&1 | tee "$tmp_out"
  fi

  extract_lm_eval_results "$tmp_out" "$task_name" "$run_name" "$model_variant" "$model_path" "$OUTDIR" "$eval_type"
  rm -f "$tmp_out"

  cd "${WORKSPACE_ROOT}"
  conda activate trans_opt
}

# CMMLU evaluation function
run_cmmlu_eval () {
  local model_path=$1 task_name=$2 run_name=$3 model_variant=$4
  local eval_out_root=$5 data_dir=$6 pred_script=$7 eval_script=$8 eval_type=$9

  [ -d "$model_path" ] || { echo "[ERROR] Model directory not found: $model_path"; return 1; }

  local PRED_ROOT="${eval_out_root}/${run_name}/${model_variant}/cmmlu_predictions"
  local EVAL_ROOT="${eval_out_root}/${run_name}/${model_variant}/cmmlu_eval"
  mkdir -p "$PRED_ROOT" "$EVAL_ROOT"

  echo ">>> [CMMLU ${eval_type}] ${task_name} | ${run_name} | ${model_variant}"
  echo ">>> Model path: ${model_path}"

  source ~/miniconda3/etc/profile.d/conda.sh
  cd "${YUE_BENCHMARK_ROOT}"
  conda activate lm-eval

  # Generate 0-shot predictions
  echo ">>> [CMMLU] Generating predictions"
  python "${pred_script}" \
    --model_path "${model_path}" \
    --data_dir "${data_dir}" \
    --output_dir "${PRED_ROOT}/0shot" \
    --num_shots 0 \
    --device cuda:0 \
    --dtype float16 \
    --max_new_tokens 512 || {
    echo "[WARN] CMMLU prediction failed"
    cd "${WORKSPACE_ROOT}"
    conda activate trans_opt
    return
  }

  # Combine prediction files
  local COMBINED="${PRED_ROOT}/combined/${model_variant}"
  mkdir -p "${COMBINED}"
  cp "${PRED_ROOT}/0shot"/*.json "${COMBINED}/" 2>/dev/null || true

  # Run evaluation (this script will print results to console)
  local tmp_out=$(mktemp)
  python "${eval_script}" \
    --predictions_dir "${PRED_ROOT}/combined" \
    --output_dir "${EVAL_ROOT}" 2>&1 | tee "$tmp_out" || {
    echo "[WARN] CMMLU evaluation failed"
    rm -f "$tmp_out"
    cd "${WORKSPACE_ROOT}"
    conda activate trans_opt
    return
  }

  # Extract results (pass temp output file to extract console output)
  extract_cmmlu_results "$EVAL_ROOT" "$task_name" "$run_name" "$model_variant" "$model_path" "$eval_type" "$tmp_out"

  rm -f "$tmp_out"

  cd "${WORKSPACE_ROOT}"
  conda activate trans_opt
}

# MalayMMLU evaluation function
run_malay_eval () {
  local model_path=$1 task_name=$2 run_name=$3 model_variant=$4
  local eval_out_root=$5 token=$6 shot=$7 eval_type=$8

  [ -d "$model_path" ] || { echo "[ERROR] Model dir not found: $model_path"; return 1; }

  local OUTDIR="${eval_out_root}/${run_name}/${model_variant}"
  mkdir -p "$OUTDIR"

  echo ">>> [MalayMMLU ${eval_type}] ${task_name} | ${run_name} | ${model_variant}"
  echo ">>> Model path: ${model_path}"
  echo ">>> Output dir: ${OUTDIR}"

  cd "${MALAY_REPO}"

  # Run evaluation
  python src/evaluate.py \
    --by_letter \
    --shot "${shot}" \
    --task=MalayMMLU \
    --base_model="${model_path}" \
    --output_folder="${OUTDIR}" \
    --token "${token}"

  # Extract results (using eval_batch.py)
  extract_malay_results "$OUTDIR" "$task_name" "$run_name" "$model_variant" "$model_path" "$eval_type"

  cd "${WORKSPACE_ROOT}"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate trans_opt
}

# Original evaluation (task-specific benchmark)
run_original_eval () {
  local model_path=$1 task_name=$2 run_name=$3 model_variant=$4

  case "$EVAL_TYPE" in
    lm_eval)
      run_lm_eval "$model_path" "$task_name" "$run_name" "$model_variant" \
        "$EVAL_TASKS_ORI" "$EVAL_OUT_ROOT_ORI" "$EVAL_BATCH_SIZE_ORI" "original"
      ;;
    cmmlu)
      run_cmmlu_eval "$model_path" "$task_name" "$run_name" "$model_variant" \
        "$EVAL_OUT_ROOT_ORI" "$CMMLU_DATA_DIR" "$CMMLU_PREDICT_SCRIPT" "$CMMLU_EVAL_SCRIPT_LOCAL" "original"
      ;;
    malay_mmlu)
      run_malay_eval "$model_path" "$task_name" "$run_name" "$model_variant" \
        "$EVAL_OUT_ROOT_ORI" "$MALAY_TOKEN" "$MALAY_SHOT" "original"
      ;;
    *)
      echo "[ERROR] Unknown EVAL_TYPE: $EVAL_TYPE"
      return 1
      ;;
  esac
}

# Ablation evaluation (unified general benchmark)
run_ablation_eval () {
  local model_path=$1 task_name=$2 run_name=$3 model_variant=$4
  run_lm_eval "$model_path" "$task_name" "$run_name" "$model_variant" \
    "$ABLATION_EVAL_TASKS" "$ABLATION_EVAL_OUT_ROOT" "$ABLATION_EVAL_BATCH_SIZE" "ablation"
}

#############################################################
#                     STEP 1: Activation Extraction + Transport Plan Computation
#############################################################

if [ "$RUN_STEP1" = true ]; then
  echo "=========== [STEP 1] Extract Activations + Transport Plan Computation ==========="
  
  for config_func in "${TASK_CONFIGS[@]}"; do
    $config_func
    echo ">>> Processing task: ${TASK_NAME}"

    if [ "$DATASET_TYPE" = "fineweb_thai" ]; then
      ensure_fineweb_thai_cache
    fi

    if dir_has_files "$HOT_DIR"; then
      echo ">>> [SKIP STEP1] ${TASK_NAME} TRANSPORT_DIR already has content: ${HOT_DIR}"
      continue
    fi
    
    CMD=(python "$ACTIVS_SCRIPT"
      --model-a-path "$MODEL_A"
      --model-b-path "$MODEL_B"
      --batch-size-a "$BATCH_SIZE_A"
      --batch-size-b "$BATCH_SIZE_B"
      --device-a "$DEVICE_A"
      --device-b "$DEVICE_B"
      --data-subset "$DATA_SUBSET"
      --data-split "train"
      --max-samples "$MAX_SAMPLES"
      --hot-chunk-cols "$HOT_CHUNK_COLS"
      --hot-dtype "$HOT_DTYPE"
      --out-dir "$HOT_DIR"
    )
    
    [ -n "$TOP_NEURON_DIR_A" ] && CMD+=(--top-neuron-dir-a "$TOP_NEURON_DIR_A")
    
    echo ">>> Executing command: ${CMD[*]}"
    "${CMD[@]}"
  done
else
  echo ">>> Skipping Step 1"
fi

#############################################################
#                     STEP 2: Transport Plan-Based Fusion
#############################################################

if [ "$RUN_STEP2" = true ]; then
  echo "=========== [STEP 2] Transport Plan-Based Fusion Model ==========="
  
  for config_func in "${TASK_CONFIGS[@]}"; do
    $config_func
    echo ">>> Processing task: ${TASK_NAME}"

    if dir_has_files "$FUSED_DIR"; then
      echo ">>> [SKIP STEP2] ${TASK_NAME} FUSED_DIR already has content: ${FUSED_DIR}"
      continue
    fi
    
    CMD=(python "$FUSE_SCRIPT"
      --modelA_id "$MODEL_A"
      --modelB_id "$MODEL_B"
      --hot_dir "$HOT_DIR"
      --alpha "$FUSE_ALPHA"
      --lm_only
      --verbose
      --output_dir "$FUSED_DIR"
      --attn_device "$ATTN_DEVICE"
      --attn_max_mem_mb "$ATTN_MAX_MEM_MB"
    )
    
    if [ -n "$TOP_NEURON_DIR_A" ]; then
      CMD+=(--hot_neuron_dir "$TOP_NEURON_DIR_A")
    elif [ -n "$HOT_NEURON_DIR" ]; then
      CMD+=(--hot_neuron_dir "$HOT_NEURON_DIR")
    fi
    
    [ "$ATTN_AUTocast" = false ] && CMD+=(--attn_no_autocast)
    
    echo ">>> Executing command: ${CMD[*]}"
    "${CMD[@]}"
  done
else
  echo ">>> Skipping Step 2"
fi

#############################################################
#          STEP 3: Hyperparameter Tuning Training + Dual Evaluation (supports resume)
#############################################################

if [ "$RUN_STEP3" = true ]; then
  echo "=================================================="
  echo "[STEP 3] Iterate alpha/lr combinations + Dual evaluation (original + ablation)"
  echo "=================================================="
  
  # Determine run label: prioritize user-set RUN_LABEL, otherwise use timestamp
  if [ -z "$RUN_LABEL" ]; then
    RUN_LABEL=$(date '+%Y%m%d_%H%M%S')
    echo ">>> [INFO] RUN_LABEL not set, using timestamp: ${RUN_LABEL}"
  else
    echo ">>> [INFO] Using user-set RUN_LABEL: ${RUN_LABEL}"
  fi
  
  # Set ablation evaluation output root (with label)
  # Use absolute path, based on LM_EVAL_REPO
  ABLATION_EVAL_OUT_ROOT="${LM_EVAL_REPO}/output/unified_eval_${RUN_LABEL}"
  
  for config_func in "${TASK_CONFIGS[@]}"; do
    $config_func
    echo "=================================================="
    echo ">>> Processing task: ${TASK_NAME}"
    echo ">>> Alpha search list: ${ALPHA_SEARCH_LIST[@]}"
    echo ">>> LR search list: ${LR_SEARCH_LIST[@]}"
    echo ">>> Run label: ${RUN_LABEL}"
    echo "=================================================="
    
    # Add label to output directory to ensure fresh run each time
    SEARCH_ROOT="${OUTPUT_ROOT}/${EXPERIMENT_TAG}_alpha_lr_search_${RUN_LABEL}"
    mkdir -p "$SEARCH_ROOT"
    
    # Iterate through all alpha and lr combinations
    for ALPHA_FIXED in "${ALPHA_SEARCH_LIST[@]}"; do
      for LR_FIXED in "${LR_SEARCH_LIST[@]}"; do
        echo "=================================================="
        echo ">>> Alpha: ${ALPHA_FIXED}, LR: ${LR_FIXED}"
        echo "=================================================="
        
        RUN_NAME="alpha${ALPHA_FIXED}_lr${LR_FIXED}"
        RUN_DIR="${SEARCH_ROOT}/${RUN_NAME}"

    HOT_OUT="${RUN_DIR}/hot"
    NOHOT_OUT="${RUN_DIR}/nohot"
    FROZEN_BASE_OUT="${RUN_DIR}/frozen_base"
    NONE_OUT="${RUN_DIR}/none"
    mkdir -p "$HOT_OUT" "$NOHOT_OUT" "$FROZEN_BASE_OUT" "$NONE_OUT"

    COMMON_ARGS=(
      --model_type "$MODEL_TYPE"
      --dataset_type "$DATASET_TYPE"
      --model_dir "$FUSED_DIR"
      --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
      --gradient_accumulation_steps "$GRAD_ACC_STEPS"
      --learning_rate "$LR_FIXED"
      --num_train_epochs "$NUM_EPOCHS"
      --block_size "$BLOCK_SIZE"
      --alpha "$ALPHA_FIXED"
      --hot_neuron_dir "$HOT_NEURON_DIR"
      --fp16
    )
    
    case "$DATASET_TYPE" in
      finance)
        COMMON_ARGS+=(--finance_split "$GEO_SPLIT")
        COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
        ;;
      fineweb_thai)
        COMMON_ARGS+=(--geo3k_split "$GEO_SPLIT")
        if [ -n "${LOCAL_DATASET_PATH:-}" ] && [ -d "$LOCAL_DATASET_PATH" ]; then
          COMMON_ARGS+=(--local_dataset_path "$LOCAL_DATASET_PATH")
        fi
        ;;
      *)
        COMMON_ARGS+=(--geo3k_split "$GEO_SPLIT")
        COMMON_ARGS+=(--max_samples_per_subset "$MAX_SAMPLES")
        ;;
    esac
    
    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/hf_cache_eval}

    ###########################################################
    # HOT training (with ablation_untrained_hot_fused)
    ###########################################################
    if [ "$ENABLE_HOT" = true ]; then
      if train_done "$HOT_OUT" && [ -d "${HOT_OUT}/ablation_untrained_hot_fused" ]; then
        echo ">>> [SKIP TRAIN] HOT training completed: $HOT_OUT"
      else
        echo ">>> [TRAIN] HOT (frozen_hot)"
        if [ "$TRAIN_LAUNCH" = "torchrun" ]; then
          ${TRAIN_LAUNCH} --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} "$TRAIN_SCRIPT" \
            --training_scenario hot \
            --freeze_strategy frozen_hot \
            --save_untrained_folded \
            --output_dir "$HOT_OUT" \
            "${COMMON_ARGS[@]}"
        else
          ${TRAIN_LAUNCH} "$TRAIN_SCRIPT" \
            --training_scenario hot \
            --freeze_strategy frozen_hot \
            --save_untrained_folded \
            --output_dir "$HOT_OUT" \
            "${COMMON_ARGS[@]}"
        fi
        touch "${HOT_OUT}/_TRAIN_DONE"
      fi

      ABLATION_MODEL="${HOT_OUT}/ablation_untrained_hot_fused"

    # Original evaluation (unified check eval root)
      if [ "$ENABLE_ORIGINAL_EVAL" = true ]; then
        if ! ori_eval_done "$RUN_NAME" "ablation"; then
          run_original_eval "$ABLATION_MODEL" "$TASK_NAME" "$RUN_NAME" "ablation"
        else
          echo ">>> [SKIP ORI EVAL] ablation already has original evaluation results"
        fi

        if ! ori_eval_done "$RUN_NAME" "hot"; then
          run_original_eval "$HOT_OUT" "$TASK_NAME" "$RUN_NAME" "hot"
        else
          echo ">>> [SKIP ORI EVAL] hot already has original evaluation results"
        fi
      fi

      # Ablation evaluation (unified check eval root)
      if [ "$ENABLE_ABLATION_EVAL" = true ]; then
        if ! ablation_eval_done "$RUN_NAME" "ablation"; then
          run_ablation_eval "$ABLATION_MODEL" "$TASK_NAME" "$RUN_NAME" "ablation"
        else
          echo ">>> [SKIP ABLATION EVAL] ablation already has ablation evaluation results"
        fi

        if ! ablation_eval_done "$RUN_NAME" "hot"; then
          run_ablation_eval "$HOT_OUT" "$TASK_NAME" "$RUN_NAME" "hot"
        else
          echo ">>> [SKIP ABLATION EVAL] hot already has ablation evaluation results"
        fi
      fi
    fi

    ###########################################################
    # noHOT
    ###########################################################
    if [ "$ENABLE_NOHOT" = true ]; then
      if train_done "$NOHOT_OUT"; then
        echo ">>> [SKIP TRAIN] noHOT training completed: $NOHOT_OUT"
      else
        echo ">>> [TRAIN] noHOT"
        if [ "$TRAIN_LAUNCH" = "torchrun" ]; then
          ${TRAIN_LAUNCH} --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} "$TRAIN_SCRIPT" \
            --training_scenario no_hot \
            --output_dir "$NOHOT_OUT" \
            "${COMMON_ARGS[@]}"
        else
          ${TRAIN_LAUNCH} "$TRAIN_SCRIPT" \
            --training_scenario no_hot \
            --output_dir "$NOHOT_OUT" \
            "${COMMON_ARGS[@]}"
        fi
        touch "${NOHOT_OUT}/_TRAIN_DONE"
      fi

      # Original evaluation (unified check eval root)
      if [ "$ENABLE_ORIGINAL_EVAL" = true ]; then
        if ! ori_eval_done "$RUN_NAME" "nohot"; then
          run_original_eval "$NOHOT_OUT" "$TASK_NAME" "$RUN_NAME" "nohot"
        else
          echo ">>> [SKIP ORI EVAL] nohot already has original evaluation results"
        fi
      fi

      # Ablation evaluation (unified check eval root)
      if [ "$ENABLE_ABLATION_EVAL" = true ]; then
        if ! ablation_eval_done "$RUN_NAME" "nohot"; then
          run_ablation_eval "$NOHOT_OUT" "$TASK_NAME" "$RUN_NAME" "nohot"
        else
          echo ">>> [SKIP ABLATION EVAL] nohot already has ablation evaluation results"
        fi
      fi
    fi

    ###########################################################
    # frozen_base
    ###########################################################
    if [ "$ENABLE_FROZEN_BASE" = true ]; then
      if train_done "$FROZEN_BASE_OUT"; then
        echo ">>> [SKIP TRAIN] frozen_base training completed: $FROZEN_BASE_OUT"
      else
        echo ">>> [TRAIN] frozen_base"
        if [ "$TRAIN_LAUNCH" = "torchrun" ]; then
          ${TRAIN_LAUNCH} --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} "$TRAIN_SCRIPT" \
            --training_scenario hot \
            --freeze_strategy frozen_base \
            --output_dir "$FROZEN_BASE_OUT" \
            "${COMMON_ARGS[@]}"
        else
          ${TRAIN_LAUNCH} "$TRAIN_SCRIPT" \
            --training_scenario hot \
            --freeze_strategy frozen_base \
            --output_dir "$FROZEN_BASE_OUT" \
            "${COMMON_ARGS[@]}"
        fi
        touch "${FROZEN_BASE_OUT}/_TRAIN_DONE"
      fi

      # Original evaluation (unified check eval root)
      if [ "$ENABLE_ORIGINAL_EVAL" = true ]; then
        if ! ori_eval_done "$RUN_NAME" "frozen_base"; then
          run_original_eval "$FROZEN_BASE_OUT" "$TASK_NAME" "$RUN_NAME" "frozen_base"
        else
          echo ">>> [SKIP ORI EVAL] frozen_base already has original evaluation results"
        fi
      fi

      # Ablation evaluation (unified check eval root)
      if [ "$ENABLE_ABLATION_EVAL" = true ]; then
        if ! ablation_eval_done "$RUN_NAME" "frozen_base"; then
          run_ablation_eval "$FROZEN_BASE_OUT" "$TASK_NAME" "$RUN_NAME" "frozen_base"
        else
          echo ">>> [SKIP ABLATION EVAL] frozen_base already has ablation evaluation results"
        fi
      fi
    fi

    # Cleanup (if all evaluations completed)
    if [ "${CLEANUP_AFTER_EVAL}" = true ]; then
      if [ "$ENABLE_HOT" = true ]; then
        hot_done=true
        ablation_done=true
        
        if [ "$ENABLE_ORIGINAL_EVAL" = true ]; then
          ori_eval_done "$RUN_NAME" "hot" || hot_done=false
          ori_eval_done "$RUN_NAME" "ablation" || ablation_done=false
        fi
        if [ "$ENABLE_ABLATION_EVAL" = true ]; then
          ablation_eval_done "$RUN_NAME" "hot" || hot_done=false
          ablation_eval_done "$RUN_NAME" "ablation" || ablation_done=false
        fi
        
        if [ "$hot_done" = true ] && [ "$ablation_done" = true ]; then
          cleanup_model_dir "$ABLATION_MODEL"
          cleanup_model_dir "$HOT_OUT"
        fi
      fi
      
      if [ "$ENABLE_NOHOT" = true ]; then
        nohot_done=true
        if [ "$ENABLE_ORIGINAL_EVAL" = true ]; then
          ori_eval_done "$RUN_NAME" "nohot" || nohot_done=false
        fi
        if [ "$ENABLE_ABLATION_EVAL" = true ]; then
          ablation_eval_done "$RUN_NAME" "nohot" || nohot_done=false
        fi
        [ "$nohot_done" = true ] && cleanup_model_dir "$NOHOT_OUT"
      fi
    fi
    
        echo ">>> [STEP 3] Alpha=${ALPHA_FIXED}, LR=${LR_FIXED} combination completed"
      done  # LR_FIXED loop end
    done  # ALPHA_FIXED loop end

    export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/hf_cache}
    
    ###########################################################
    # Evaluate source model modelA (baseline) - executed once per task
    ###########################################################
    if [ -d "$MODEL_A" ]; then
      echo ">>> [EVAL] ${TASK_NAME} - modelA (source model baseline)"

      # Original evaluation (unified check eval root)
      if [ "$ENABLE_ORIGINAL_EVAL" = true ]; then
        if ! ori_eval_done "baseline" "modelA"; then
          run_original_eval "$MODEL_A" "$TASK_NAME" "baseline" "modelA"
        else
          echo ">>> [SKIP ORI EVAL] baseline modelA already has original evaluation results"
        fi
      fi

      # Ablation evaluation (unified check eval root)
      if [ "$ENABLE_ABLATION_EVAL" = true ]; then
        if ! ablation_eval_done "baseline" "modelA"; then
          run_ablation_eval "$MODEL_A" "$TASK_NAME" "baseline" "modelA"
        else
          echo ">>> [SKIP ABLATION EVAL] baseline modelA already has ablation evaluation results"
        fi
      fi
    fi
    
    echo ">>> [STEP 3] Task ${TASK_NAME} completed"
  done  # config_func loop end
  
  echo "=================================================="
  echo "[STEP 3] All tasks training and evaluation completed"
  echo "=================================================="
else
  echo ">>> Skipping Step 3"
fi

#############################################################
#                     STEP 4: Hint
#############################################################

if [ "$RUN_STEP4" = true ]; then
  echo "=================================================="
  echo "[STEP 4] (Hint)"
  echo "Evaluation has been completed within STEP3, and models have been deleted as needed."
  echo "Step4 does not perform any evaluation operations."
  echo "=================================================="
else
  echo ">>> Skipping Step 4"
fi

