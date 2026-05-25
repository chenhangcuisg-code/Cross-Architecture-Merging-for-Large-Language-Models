#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_RESULTS_LOG="${SCRIPT_DIR}/eval_results_summary_base.log"

export PYTHONPATH="/data/chenhang/codes/lm-evaluation-harness:${PYTHONPATH:-}"
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TRUST_REMOTE_CODE=1
mkdir -p "$HF_DATASETS_CACHE"

# 日志头：仅文件不存在时创建；存在则追加新 session 分隔
if [ ! -f "$EVAL_RESULTS_LOG" ]; then
  {
    echo "=========================================="
    echo "Evaluation Results Summary"
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

dir_has_files() {
  local d="$1"
  if [ -d "$d" ]; then
    find "$d" -maxdepth 1 -type f 2>/dev/null | grep -q .
  else
    return 1
  fi
}

# lm_eval 结果是否已存在（results*.json 即视为完成）；兼容扁平/任务目录
lm_eval_done() {
  local out_dir="$1"
  # 常规：输出目录下 results*.json
  if find "$out_dir" -maxdepth 1 -type f -name "results*.json" 2>/dev/null | grep -q .; then
    return 0
  fi
  # 兼容 baseline 早期扁平/任务名目录（如 output/llama3-eval/medical/.../results_*.json）
  find "$out_dir" -maxdepth 5 -type f -name "results*.json" 2>/dev/null | grep -q .
}

# 提取 lm_eval 表格并写入日志
extract_eval_results() {
  local output_file=$1
  local task_name=$2
  local model_path=$3
  local out_dir=$4

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
      echo "Task: ${task_name}"
      echo "Model: ${model_path}"
      echo "Output: ${out_dir}"
      echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo "$results"
      echo ""
    } >>"$EVAL_RESULTS_LOG"
  fi
}

run_lm_eval() {
  local model_path=$1
  local tasks=$2
  local out_dir=$3
  local task_name=$4
  local batch_size=$5

  mkdir -p "$out_dir"

  if lm_eval_done "$out_dir"; then
    echo "[SKIP] ${task_name} 已有 results*.json，跳过评测。"
    return
  fi

  echo ">>> [LM_EVAL] ${task_name}"
  local tmp_out
  tmp_out=$(mktemp)
  python -m lm_eval --model hf \
    --model_args "pretrained=${model_path},dtype=float" \
    --tasks "${tasks}" \
    --device cuda:0 \
    --batch_size "${batch_size}" \
    --output_path "${out_dir}" 2>&1 | tee "${tmp_out}"

  extract_eval_results "${tmp_out}" "${task_name}" "${model_path}" "${out_dir}"
  rm -f "${tmp_out}"
}

run_cmmlu_eval() {
  local model_path=$1
  local pred_dir=$2
  local eval_dir=$3

  mkdir -p "${pred_dir}/0shot" "${pred_dir}/combined/modelA" "${eval_dir}"

  # 注释掉跳过逻辑，允许重新运行
  if dir_has_files "${eval_dir}"; then
    echo "[SKIP] CMMLU 已有评测结果，跳过。"
    return
  fi

  source ~/miniconda3/etc/profile.d/conda.sh
  cd /data/chenhang/codes/Yue-Benchmark
  conda activate lm-eval

  local tmp_out
  tmp_out=$(mktemp)

  python evaluation/generate_cmmlu_predictions.py \
    --model_path "${model_path}" \
    --data_dir /data/chenhang/codes/Yue-Benchmark/data/latest_data/Yue-MMLU \
    --output_dir "${pred_dir}/0shot" \
    --num_shots 0 \
    --device cuda:0 \
    --dtype float16 \
    --max_new_tokens 512 2>&1 | tee -a "${tmp_out}"

  cp "${pred_dir}/0shot"/*.json "${pred_dir}/combined/modelA/" 2>/dev/null || true

  python evaluation/evaluate_cmmlu_yue.py \
    --predictions_dir "${pred_dir}/combined" \
    --output_dir "${eval_dir}" 2>&1 | tee -a "${tmp_out}"

  # 提取并记录 CMMLU 结果
  {
    echo "=========================================="
    echo "Task: cantonese (CMMLU)"
    echo "Model: ${model_path}"
    echo "Output: ${eval_dir}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    # 尝试从结果文件中提取信息
    if [ -f "${eval_dir}/cmmlu_yue_results.json" ]; then
      echo "CMMLU Results:"
      cat "${eval_dir}/cmmlu_yue_results.json" 2>/dev/null || echo "结果文件读取失败"
    elif [ -f "${eval_dir}/results.json" ]; then
      echo "CMMLU Results:"
      cat "${eval_dir}/results.json" 2>/dev/null || echo "结果文件读取失败"
    else
      echo "评测完成，结果保存在: ${eval_dir}"
      find "${eval_dir}" -type f -name "*.json" -o -name "*.txt" 2>/dev/null | head -5
    fi
    echo ""
  } >>"$EVAL_RESULTS_LOG"

  rm -f "${tmp_out}"
  cd /home/chenhang/optimal_trans
  conda activate trans_opt
}

run_malay_eval() {
  local model_path=$1
  local out_dir=$2

  mkdir -p "${out_dir}"
  # 注释掉跳过逻辑，允许重新运行
  if dir_has_files "${out_dir}"; then
    echo "[SKIP] MalayMMLU 已有评测结果，跳过。"
    return
  fi

  local tmp_out
  tmp_out=$(mktemp)

  cd /home/chenhang/MalayMMLU
  python src/evaluate.py \
    --by_letter \
    --shot 0 \
    --task=MalayMMLU \
    --base_model="${model_path}" \
    --output_folder="${out_dir}" \
    --token "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}" 2>&1 | tee "${tmp_out}"

  # 提取并记录 MalayMMLU 结果
  {
    echo "=========================================="
    echo "Task: malay (MalayMMLU)"
    echo "Model: ${model_path}"
    echo "Output: ${out_dir}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    # 尝试从输出中提取结果信息
    if grep -q "accuracy\|score\|result" "${tmp_out}" 2>/dev/null; then
      grep -E "accuracy|score|result|Overall" "${tmp_out}" 2>/dev/null | head -20 || echo "评测完成，结果保存在: ${out_dir}"
    else
      echo "评测完成，结果保存在: ${out_dir}"
      find "${out_dir}" -type f -name "*.json" -o -name "*.txt" 2>/dev/null | head -5
    fi
    echo ""
  } >>"$EVAL_RESULTS_LOG"

  rm -f "${tmp_out}"
  cd /home/chenhang/optimal_trans
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate trans_opt
}

############################
# 任务列表（顺序可调整）
############################

run_lm_eval "/data/chenhang/models/llama3-1b-med" \
  "medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine" \
  "output/llama3-eval/medical" \
  "medical" \
  1

run_lm_eval "/data/chenhang/models/llama3.2-typhoon2-1b-instruct" \
  "xcopa_th,xquad_th,xnli_th" \
  "output/llama3.2-1bth-instruct/thai" \
  "thai" \
  1

run_lm_eval "/data/chenhang/models/Llama-3.2-1B-Instruct" \
  "global_piqa_completions_fin_latn,global_piqa_prompted_fin_latn,belebele_fin_Latn,multiblimp_fin,financial_tweets" \
  "output/unified_eval/finance" \
  "finance" \
  1

run_lm_eval "/data/chenhang/models/Llama-3.2-1B-Indonesian-QLora" \
  "belebele_ind_Latn,xcopa_id,arc_id,xstorycloze_id,truthfulqa_id_mc1,truthfulqa_id_mc2" \
  "output/llama3.2-1bid-instruct-reasoning/indonesian" \
  "indonesian" \
  1

run_lm_eval "/data/chenhang/models/SFT_math_Llama-3.2-1B" \
  "arithmetic_2da,arithmetic_2ds,minerva_math_prealgebra,bigbench_simple_arithmetic_json_generate_until,mmlu_elementary_mathematics" \
  "output/llama3-eval/math" \
  "math" \
  1

# CMMLU 粤语 baseline
run_cmmlu_eval "/data/chenhang/models/Llama3-Chinese-8B-Instruct" \
  "/data/chenhang/codes/Yue-Benchmark/output_llama3.2-1bcan-instruct-conversation/baseline/modelA/cmmlu_predictions" \
  "/data/chenhang/codes/Yue-Benchmark/output_llama3.2-1bcan-instruct-conversation/baseline/modelA/cmmlu_eval"

# MalayMMLU baseline
run_malay_eval "/data/chenhang/models/Malaysian-Llama-3.2-1B-Instruct-v0.1" \
  "/home/chenhang/MalayMMLU/output/llama3.2-1bma-instruct/baseline/modelA"

