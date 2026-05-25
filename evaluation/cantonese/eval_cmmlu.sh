#!/usr/bin/env bash
set -euo pipefail

#############################################
# 基础配置（修改这里）
#############################################

MODEL_A_PATH="/data/chenhang/models/Llama-3.2-1B-Instruct"
MODEL_B_PATH="/data/chenhang/models/Llama-3.2-1B-Instruct"

# 控制是否执行 CMMLU 预测与评测
ENABLE_CMMLU_PREDICT=true
ENABLE_CMMLU_EVAL=true

# 评测任务
EVAL_TASKS="ifeval,mmlu"
EVAL_OUT_ROOT="output/base_model_eval"

# ------- CMMLU 粤语配置 -------
CMMLU_DATA_DIR="/data/chenhang/codes/Yue-Benchmark/data/latest_data/Yue-MMLU"
CMMLU_PREDICT_SCRIPT="evaluation/generate_cmmlu_predictions.py"
CMMLU_EVAL_SCRIPT_LOCAL="eval_batch.py"

#############################################
# 函数：CMMLU 评测（仅 0-shot）
#############################################
run_cmmlu_eval () {
  local model_path=$1
  local model_name=$2

  echo ">>> [CMMLU] Evaluating ${model_name}"

  local BASE_OUT="${EVAL_OUT_ROOT}/${model_name}"
  local PRED_DIR="${BASE_OUT}/cmmlu_predictions"
  local EVAL_DIR="${BASE_OUT}/cmmlu_eval"

  mkdir -p "$PRED_DIR" "$EVAL_DIR"

  #############################################
  #            CMMLU 0-shot 预测
  #############################################
  if [ "${ENABLE_CMMLU_PREDICT}" = true ]; then
    echo ">>> [CMMLU] 生成 0-shot 预测..."

    python "${CMMLU_PREDICT_SCRIPT}" \
      --model_path "${model_path}" \
      --data_dir "${CMMLU_DATA_DIR}" \
      --output_dir "${PRED_DIR}/0shot" \
      --num_shots 0 \
      --device cuda:0 \
      --dtype float16 \
      --max_new_tokens 512 || echo "[WARN] CMMLU 0-shot 失败"
  else
    echo ">>> [CMMLU] 已关闭预测（ENABLE_CMMLU_PREDICT=false）"
  fi


  #############################################
  #            合并预测文件结构
  #############################################
  local COMBINED="${PRED_DIR}/combined/${model_name}"
  mkdir -p "${COMBINED}"

  cp "${PRED_DIR}/0shot"/*.json "${COMBINED}/" 2>/dev/null || true


  #############################################
  #            CMMLU 评测（只 0-shot）
  #############################################
  if [ "${ENABLE_CMMLU_EVAL}" = true ]; then
    echo ">>> [CMMLU] Running evaluation..."

    python "${CMMLU_EVAL_SCRIPT_LOCAL}" \
      --predictions_dir "${PRED_DIR}/combined" \
      --output_dir "${EVAL_DIR}" || echo "[WARN] CMMLU eval 失败"
  else
    echo ">>> [CMMLU] 已关闭评测（ENABLE_CMMLU_EVAL=false）"
  fi

  echo ">>> [CMMLU] Done for ${model_name}"
}


#############################################
# 主流程
#############################################

mkdir -p "${EVAL_OUT_ROOT}"

#run_cmmlu_eval "${MODEL_A_PATH}" "modelA"
run_cmmlu_eval "${MODEL_B_PATH}" "modelB"

echo "==============================================="
echo "🎉 Base models evaluation 完成"
echo "输出目录: ${EVAL_OUT_ROOT}"
echo "==============================================="
