#!/bin/bash
# 从已有的评测结果JSON文件中提取结果并写入汇总日志

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_RESULTS_LOG="${SCRIPT_DIR}/eval_results_summary_merged_indoqwen_oridata.log"

# 评测结果根目录
EVAL_OUT_ROOT_ORI="/data/chenhang/codes/lm-evaluation-harness/output_nolora_final_nolora_same_param_debugindo2/llama3.2-1bid-instruct-fixed_final_nolora_same_param_debugindo2"

# 任务配置
TASK_NAME="indonesian_qwen"
RUN_NAME_PREFIX="alpha"

# 从JSON文件提取结果并格式化
extract_from_json() {
    local json_file=$1
    local task_name=$2
    local run_name=$3
    local model_variant=$4
    local model_path=$5
    local eval_type=$6
    
    [ -f "$json_file" ] || return 1
    
    # 使用Python提取结果
    python3 << PYTHON_EOF
import json
import sys

try:
    with open("$json_file", 'r') as f:
        data = json.load(f)
    
    if 'results' not in data:
        sys.exit(1)
    
    results = data['results']
    
    # 打印表头
    print("Saving results aggregated")
    print("| Task | Metric | Value |")
    print("|------|--------|-------|")
    
    # 打印每个任务的结果
    for task, result in sorted(results.items()):
        # 优先使用acc_norm，否则使用acc
        if 'acc_norm,none' in result:
            metric = 'acc_norm'
            value = result['acc_norm,none']
            stderr = result.get('acc_norm_stderr,none', 0)
            print(f"| {task} | {metric} | {value:.6f} ± {stderr:.6f} |")
        elif 'acc,none' in result:
            metric = 'acc'
            value = result['acc,none']
            stderr = result.get('acc_stderr,none', 0)
            print(f"| {task} | {metric} | {value:.6f} ± {stderr:.6f} |")
    
    print("")
    
except Exception as e:
    sys.exit(1)
PYTHON_EOF
}

# 检查结果是否已记录
eval_logged() {
    local task=$1 run_name=$2 model_variant=$3 eval_type=$4
    local key="Task: ${task} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
    grep -F "$key" "$EVAL_RESULTS_LOG" >/dev/null 2>&1
}

# 提取并写入结果
process_eval_result() {
    local run_name=$1
    local model_variant=$2
    local eval_type="original"
    
    local result_dir="${EVAL_OUT_ROOT_ORI}/${run_name}/${model_variant}"
    
    # 查找最新的results JSON文件
    local json_file=$(find "$result_dir" -name "results*.json" -type f 2>/dev/null | sort | tail -1)
    
    if [ -z "$json_file" ]; then
        echo ">>> [WARN] 未找到结果文件: ${result_dir}"
        return
    fi
    
    # 检查是否已记录
    if eval_logged "$TASK_NAME" "$run_name" "$model_variant" "$eval_type"; then
        echo ">>> [SKIP] 结果已记录: ${run_name}/${model_variant}"
        return
    fi
    
    # 从模型路径推断（简化处理）
    local model_path="N/A"
    
    # 提取结果
    local results=$(extract_from_json "$json_file" "$TASK_NAME" "$run_name" "$model_variant" "$model_path" "$eval_type")
    
    if [ -n "$results" ]; then
        local key="Task: ${TASK_NAME} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
        {
            echo "=========================================="
            echo "${key}"
            echo "Model: ${model_path}"
            echo "Output: ${result_dir}"
            echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
            echo "$results"
            echo ""
        } >> "$EVAL_RESULTS_LOG"
        echo ">>> [OK] 已提取结果: ${run_name}/${model_variant}"
    else
        echo ">>> [WARN] 未能提取结果: ${run_name}/${model_variant}"
    fi
}

# 主流程
echo "=========================================="
echo "从已有评测结果中提取并写入汇总日志"
echo "=========================================="
echo "结果目录: ${EVAL_OUT_ROOT_ORI}"
echo "汇总日志: ${EVAL_RESULTS_LOG}"
echo ""

# 获取所有参数组合
for alpha_lr_dir in "${EVAL_OUT_ROOT_ORI}"/alpha*; do
    if [ -d "$alpha_lr_dir" ]; then
        run_name=$(basename "$alpha_lr_dir")
        echo ">>> 处理参数组合: ${run_name}"
        
        # 处理每个变体
        for variant in hot nohot ablation; do
            if [ -d "${alpha_lr_dir}/${variant}" ]; then
                process_eval_result "$run_name" "$variant"
            fi
        done
        
    fi
done

# 处理baseline（在根目录下）
if [ -d "${EVAL_OUT_ROOT_ORI}/baseline" ]; then
    echo ">>> 处理baseline"
    # baseline目录下可能有子目录
    for baseline_subdir in "${EVAL_OUT_ROOT_ORI}"/baseline/*; do
        if [ -d "$baseline_subdir" ]; then
            variant_name=$(basename "$baseline_subdir")
            # 查找results文件
            json_file=$(find "$baseline_subdir" -name "results*.json" -type f 2>/dev/null | sort | tail -1)
            if [ -n "$json_file" ]; then
                result_dir="$baseline_subdir"
                run_name="baseline"
                model_variant="$variant_name"
                eval_type="original"
                
                if ! eval_logged "$TASK_NAME" "$run_name" "$model_variant" "$eval_type"; then
                    results=$(extract_from_json "$json_file" "$TASK_NAME" "$run_name" "$model_variant" "N/A" "$eval_type")
                    if [ -n "$results" ]; then
                        key="Task: ${TASK_NAME} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${eval_type}"
                        {
                            echo "=========================================="
                            echo "${key}"
                            echo "Model: N/A"
                            echo "Output: ${result_dir}"
                            echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
                            echo "=========================================="
                            echo "$results"
                            echo ""
                        } >> "$EVAL_RESULTS_LOG"
                        echo ">>> [OK] 已提取结果: baseline/${model_variant}"
                    fi
                else
                    echo ">>> [SKIP] 结果已记录: baseline/${model_variant}"
                fi
            fi
        fi
    done
fi

echo ""
echo "=========================================="
echo "提取完成！"
echo "汇总日志: ${EVAL_RESULTS_LOG}"
echo "=========================================="

