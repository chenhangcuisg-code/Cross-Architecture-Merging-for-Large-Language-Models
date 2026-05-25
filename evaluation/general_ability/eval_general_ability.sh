#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# General Ability 评估脚本
# 评估任务: arc_easy, commonsense_qa, piqa, social_iqa, winogrande
# 变体: hot, nohot, frozen_base, baseline
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 评估结果日志
EVAL_RESULTS_LOG="${SCRIPT_DIR}/eval_general_ability.log"

# lm-eval harness 配置
export PYTHONPATH="/data/chenhang/codes/lm-evaluation-harness:${PYTHONPATH:-}"
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
mkdir -p "$HF_DATASETS_CACHE"

LM_EVAL_REPO="/data/chenhang/codes/lm-evaluation-harness"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_REPO}/output_general_ability"

# 评估任务
EVAL_TASKS="winogrande,social_iqa,arc_easy,piqa,commonsense_qa"
EVAL_BATCH_SIZE=8

# ============================================================
# 模型路径配置
# 请根据实际情况修改以下路径
# ============================================================

# 基础路径（alpha0.005_lr6e-7对应的run）
BASE_RUN_PATH="/data/chenhang/optimal_trans_new/hot_sft_runs_finical_llama3_1b_qwen25/llama_finance_train_alpha_lr_search_final_nolora_same_param_finllamaqwen/alpha0.005_lr6e-7"

# 各变体的模型路径
# 注意：如果路径不存在，请通过环境变量覆盖或修改以下路径
MODEL_HOT="${BASE_RUN_PATH}/hot"
MODEL_NOHOT="${BASE_RUN_PATH}/nohot"
# frozen_base可能是ablation_untrained_hot_fused，也可能直接在frozen_base目录
MODEL_FROZEN_BASE="${BASE_RUN_PATH}/frozen_base"
# 如果上面的路径不存在，尝试这个：
[ ! -d "$MODEL_FROZEN_BASE" ] && MODEL_FROZEN_BASE="${BASE_RUN_PATH}/hot/ablation_untrained_hot_fused"
MODEL_BASELINE="/data/chenhang/models/Llama-3.2-1B-TEL-A-finance"

# 如果模型路径不存在，尝试其他可能的路径
# 可以通过环境变量覆盖
MODEL_HOT=${MODEL_HOT_OVERRIDE:-${MODEL_HOT}}
MODEL_NOHOT=${MODEL_NOHOT_OVERRIDE:-${MODEL_NOHOT}}
MODEL_FROZEN_BASE=${MODEL_FROZEN_BASE_OVERRIDE:-${MODEL_FROZEN_BASE}}
MODEL_BASELINE=${MODEL_BASELINE_OVERRIDE:-${MODEL_BASELINE}}

# ============================================================
# 辅助函数
# ============================================================

# 初始化日志
init_log() {
  if [ ! -f "$EVAL_RESULTS_LOG" ]; then
    {
      echo "=========================================="
      echo "General Ability Evaluation Results"
      echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo ""
    } >"$EVAL_RESULTS_LOG"
  else
    {
      echo ""
      echo "=========================================="
      echo "New Session Started at: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo ""
    } >>"$EVAL_RESULTS_LOG"
  fi
}

# 检查lm_eval结果是否已存在
lm_eval_done() {
  local out_dir="$1"
  find "$out_dir" -maxdepth 1 -type f -name "results*.json" 2>/dev/null | grep -q .
}

# 提取lm_eval结果并写入日志
extract_eval_results() {
  local output_file=$1
  local task_name=$2
  local model_path=$3
  local out_dir=$4
  local variant=$5

  [ -f "$output_file" ] || return

  local results
  results=$(awk '
    BEGIN { in_table=0 }
    /Saving results aggregated/ {
      in_table=1; print; next
    }
    in_table {
      print
      if (/^$/) exit
    }
  ' "$output_file" 2>/dev/null || true)

  if [ -n "$results" ]; then
    {
      echo "=========================================="
      echo "Task: general_ability"
      echo "Variant: ${variant}"
      echo "Model: ${model_path}"
      echo "Output: ${out_dir}"
      echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo "$results"
      echo ""
    } >>"$EVAL_RESULTS_LOG"
  fi
}

# 运行lm_eval评估
run_lm_eval() {
  local model_path=$1
  local variant=$2
  local out_dir=$3

  # 检查模型路径
  if [ ! -d "$model_path" ]; then
    echo "[ERROR] 模型路径不存在: ${model_path}"
    echo "[ERROR] 请检查路径或设置环境变量覆盖（如 MODEL_HOT_OVERRIDE）"
    return 1
  fi

  mkdir -p "$out_dir"

  # 检查是否已完成
  if lm_eval_done "$out_dir"; then
    echo "[SKIP] ${variant} 已有 results*.json，跳过评测。"
    return 0
  fi

  echo ">>> [LM_EVAL] General Ability - ${variant}"
  echo ">>> 模型路径: ${model_path}"
  echo ">>> 输出目录: ${out_dir}"

  local tmp_out
  tmp_out=$(mktemp)

  # 激活conda环境并运行评估
  source ~/miniconda3/etc/profile.d/conda.sh
  cd "${LM_EVAL_REPO}"
  conda activate lm-eval

  python -m lm_eval --model hf \
    --model_args "pretrained=${model_path},dtype=float" \
    --tasks "${EVAL_TASKS}" \
    --device cuda:0 \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --output_path "${out_dir}" 2>&1 | tee "${tmp_out}"

  extract_eval_results "${tmp_out}" "general_ability" "${model_path}" "${out_dir}" "${variant}"
  rm -f "${tmp_out}"

  cd "$SCRIPT_DIR"
  conda activate trans_opt
}

# ============================================================
# 主执行流程
# ============================================================

main() {
  init_log

  echo "=========================================="
  echo "开始 General Ability 评估"
  echo "评估任务: ${EVAL_TASKS}"
  echo "=========================================="
  echo ""

  # 创建输出目录
  mkdir -p "${LM_EVAL_OUTPUT_DIR}"

  # 评估各个变体
  echo ">>> 评估 hot 变体..."
  run_lm_eval "${MODEL_HOT}" "hot" "${LM_EVAL_OUTPUT_DIR}/hot"

  echo ""
  echo ">>> 评估 nohot 变体..."
  run_lm_eval "${MODEL_NOHOT}" "nohot" "${LM_EVAL_OUTPUT_DIR}/nohot"

  echo ""
  echo ">>> 评估 frozen_base 变体..."
  run_lm_eval "${MODEL_FROZEN_BASE}" "frozen_base" "${LM_EVAL_OUTPUT_DIR}/frozen_base"

  echo ""
  echo ">>> 评估 baseline 变体..."
  run_lm_eval "${MODEL_BASELINE}" "baseline" "${LM_EVAL_OUTPUT_DIR}/baseline"

  echo ""
  echo "=========================================="
  echo "评估完成！"
  echo "结果日志: ${EVAL_RESULTS_LOG}"
  echo "输出目录: ${LM_EVAL_OUTPUT_DIR}"
  echo "=========================================="
}

# 运行主函数
main "$@"
