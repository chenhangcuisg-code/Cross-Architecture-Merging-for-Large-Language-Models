#!/bin/bash

# 手动提取fused模型的MalayMMLU结果

eval_logged() {
  local task=$1 run_name=$2 model_variant=$3 eval_type=$4
  local key="Task: ${task} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
  grep -F "$key" "/home/chenhang/optimal_trans/eval_results_summary_merged_malaqwen.log" >/dev/null 2>&1
}

extract_malay_results() {
  local out_dir=$1
  local task_name=$2
  local run_name=$3
  local model_variant=$4
  local model_path=$5
  local eval_type=$6

  local key="Task: ${task_name} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
  if eval_logged "$task_name" "$run_name" "$model_variant" "$eval_type"; then
    echo ">>> [SKIP] 结果已存在于日志中"
    return
  fi

  # 查找所有CSV结果文件
  local csv_files=($(find "$out_dir" -name "MalayMMLU_result_*.csv" -type f))
  if [ ${#csv_files[@]} -eq 0 ]; then
    echo ">>> [WARN] 未找到MalayMMLU CSV结果文件: ${out_dir}"
    return
  fi

  echo ">>> [MalayMMLU] 使用 eval_batch.py 提取完整结果 (找到 ${#csv_files[@]} 个CSV文件)"
  
  # 切换到MalayMMLU目录并运行eval_batch.py
  local saved_dir=$(pwd)
  cd "/home/chenhang/MalayMMLU"
  local temp_result_dir=$(mktemp -d)
  
  # 从第一个CSV文件名提取shot数
  local shot=$(echo "${csv_files[0]}" | grep -oP '\d+shot' | grep -oP '\d+' || echo "0")
  
  echo ">>> [MalayMMLU] 运行 eval_batch.py，shot=${shot}，CSV文件数=${#csv_files[@]}"
  
  # 运行eval_batch.py提取完整结果
  if ! python eval_batch.py \
    --pred_files "${csv_files[@]}" \
    --shot "${shot}" \
    --output_dir "${temp_result_dir}" 2>&1; then
    echo "[WARN] eval_batch.py 执行失败"
    rm -rf "$temp_result_dir"
    cd "$saved_dir"
    return
  fi

  # 从生成的CSV文件中读取结果
  local results_csv="${temp_result_dir}/accuracy_results_sorted.csv"
  if [ ! -f "$results_csv" ]; then
    echo ">>> [WARN] 未找到eval_batch.py生成的CSV文件: ${results_csv}"
    rm -rf "$temp_result_dir"
    cd "$saved_dir"
    return
  fi

  echo ">>> [MalayMMLU] 从CSV文件读取结果: ${results_csv}"

  # 从run_name中提取alpha和lr
  local alpha=$(echo "$run_name" | sed -n 's/.*alpha\([^_]*\)_lr.*/\1/p' || echo "")
  local lr=$(echo "$run_name" | sed -n 's/.*_lr\(.*\)/\1/p' || echo "")
  
  echo ">>> [MalayMMLU] 匹配参数: alpha=${alpha}, lr=${lr}, eval_variant=${model_variant}"
  
  # 使用Python从CSV中提取匹配的结果
  local results=$(python3 <<PYTHON_EOF
import csv
import sys

results_csv = "${results_csv}"
alpha = "${alpha}"
lr = "${lr}"
model_variant = "${model_variant}"

try:
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 查找匹配的行
    matching_rows = []
    for row in rows:
        row_alpha = row.get("alpha", "").strip()
        row_lr = row.get("lr", "").strip()
        row_variant = row.get("eval_variant", "").strip()
        
        if row_alpha == alpha and row_lr == lr and row_variant == model_variant:
            matching_rows.append(row)
    
    if not matching_rows:
        print(f"No matching rows found", file=sys.stderr)
        sys.exit(1)
    
    # 格式化结果
    result_lines = []
    for row in matching_rows:
        result_lines.append(f"Alpha: {row.get('alpha', 'N/A')}, LR: {row.get('lr', 'N/A')}, Variant: {row.get('eval_variant', 'N/A')}")
        result_lines.append(f"Overall Accuracy: N/A")
        for col in ["accuracy_Humanities", "accuracy_Language", "accuracy_Others", "accuracy_STEM", "accuracy_Social science"]:
            if col in row and row[col]:
                subject_name = col.replace("accuracy_", "")
                result_lines.append(f"  {subject_name}: {row[col]}")
    
    print("\\n".join(result_lines))

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
)

  local python_exit_code=$?
  if [ $python_exit_code -eq 0 ] && [ -n "$results" ]; then
    {
      echo "=========================================="
      echo "${key}"
      echo "Model: ${model_path}"
      echo "Output: ${out_dir}"
      echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "=========================================="
      echo "$results"
      echo ""
    } >> "/home/chenhang/optimal_trans/eval_results_summary_merged_malaqwen.log"
    echo ">>> [MalayMMLU] 结果已写入日志"
  else
    echo ">>> [WARN] 未能从eval_batch.py生成的CSV中提取结果 (alpha=${alpha}, lr=${lr}, eval_variant=${model_variant})"
    echo ">>> [WARN] Python退出码: ${python_exit_code}"
  fi
  
  rm -rf "$temp_result_dir"
  cd "$saved_dir"
}

# 主执行部分
echo "开始手动提取fused模型的MalayMMLU结果..."

# 提取fused结果 (对应ablation目录中的结果，但eval_variant是fused)
echo "提取fused模型结果..."
extract_malay_results \
  "/home/chenhang/MalayMMLU/output/llama3.2-1bma-instruct-fixed-nolora-sameparam_final_nolora_same_param_final_nolora_same_param_qwen/alpha0.2_lr1e-6/ablation" \
  "malay" \
  "alpha0.2_lr1e-6" \
  "fused" \
  "/data/chenhang/optimal_trans_new/hot_sft_runs_llamamala_1b/llama_malaysian_sft_train_alpha_lr_search_final_nolora_same_param_qwen/alpha0.2_lr1e-6/hot/ablation_untrained_hot_fused" \
  "original"

echo "fused模型结果提取完成。"
