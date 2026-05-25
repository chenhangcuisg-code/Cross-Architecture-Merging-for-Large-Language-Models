#!/bin/bash
# 从已有的评测结果JSON文件中提取缺失的结果并写入汇总日志

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_RESULTS_LOG="${SCRIPT_DIR}/eval_results_summary_merged_indonesian_oridata.log"

# 评测结果根目录
EVAL_OUT_ROOT_ORI="/data/chenhang/codes/lm-evaluation-harness/output_nolora_final_nolora_same_param_debugindo2/llama3.2-1bid-instruct-fixed_final_nolora_same_param_debugindo2"

# 任务配置
TASK_NAME="indonesian"
EVAL_TYPE="original"

# 从JSON文件提取结果并格式化
extract_from_json() {
    local json_file=$1
    
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
    
    # 打印表头（匹配lm_eval的输出格式）
    print("2026-01-15:00:00:00 INFO     [loggers.evaluation_tracker:209] Saving results aggregated")
    print("hf (pretrained=N/A,dtype=float), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 8")
    print("|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|")
    print("|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|")
    
    # 打印每个任务的结果
    for task, result in sorted(results.items()):
        # 处理arc_id等有多个metric的任务
        if 'acc,none' in result:
            acc = result['acc,none']
            acc_stderr = result.get('acc_stderr,none', 0)
            print(f"|{task:<17}|{result.get('version', 'N/A'):>6}|none  |{result.get('n-shot', 0):>5}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")
        
        if 'acc_norm,none' in result:
            acc_norm = result['acc_norm,none']
            acc_norm_stderr = result.get('acc_norm_stderr,none', 0)
            print(f"|{'':<17}|{'':>6}|none  |{result.get('n-shot', 0):>5}|acc_norm|↑  |{acc_norm:.4f}|±  |{acc_norm_stderr:.4f}|")
        elif 'acc,none' in result and 'acc_norm,none' not in result:
            # 如果没有acc_norm，只打印acc（已经在上面打印了）
            pass
    
    print("")
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
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
    
    local result_dir="${EVAL_OUT_ROOT_ORI}/${run_name}/${model_variant}"
    
    # 查找最新的results JSON文件
    local json_file=$(find "$result_dir" -name "results*.json" -type f 2>/dev/null | sort | tail -1)
    
    if [ -z "$json_file" ]; then
        echo ">>> [WARN] 未找到结果文件: ${result_dir}"
        return
    fi
    
    # 检查是否已记录
    if eval_logged "$TASK_NAME" "$run_name" "$model_variant" "$EVAL_TYPE"; then
        echo ">>> [SKIP] 结果已记录: ${run_name}/${model_variant}"
        return
    fi
    
    # 从模型路径推断（简化处理）
    local model_path="N/A"
    if [ -d "${result_dir}/../../hot" ] || [ -d "${result_dir}/../../nohot" ]; then
        # 尝试从路径推断
        local parent_dir=$(dirname "$result_dir")
        local grandparent_dir=$(dirname "$parent_dir")
        model_path="${grandparent_dir}/${model_variant}"
        [ ! -d "$model_path" ] && model_path="N/A"
    fi
    
    # 提取结果
    local results=$(extract_from_json "$json_file")
    
    if [ -n "$results" ]; then
        local key="Task: ${TASK_NAME} | Run: ${run_name} | Variant: ${model_variant} | EvalType: ${EVAL_TYPE}"
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
echo "从已有评测结果中提取缺失的结果"
echo "=========================================="
echo "结果目录: ${EVAL_OUT_ROOT_ORI}"
echo "汇总日志: ${EVAL_RESULTS_LOG}"
echo ""

# 获取所有参数组合
for alpha_lr_dir in "${EVAL_OUT_ROOT_ORI}"/alpha* "${EVAL_OUT_ROOT_ORI}"/baseline; do
    if [ -d "$alpha_lr_dir" ]; then
        run_name=$(basename "$alpha_lr_dir")
        echo ">>> 处理参数组合: ${run_name}"
        
        # 处理每个变体
        for variant in hot nohot ablation; do
            if [ -d "${alpha_lr_dir}/${variant}" ]; then
                process_eval_result "$run_name" "$variant"
            fi
        done
        
        # 处理baseline（如果存在）
        if [ "$run_name" = "baseline" ]; then
            for baseline_subdir in "${alpha_lr_dir}"/*; do
                if [ -d "$baseline_subdir" ]; then
                    variant_name=$(basename "$baseline_subdir")
                    # 查找results文件
                    json_file=$(find "$baseline_subdir" -name "results*.json" -type f 2>/dev/null | sort | tail -1)
                    if [ -n "$json_file" ]; then
                        result_dir="$baseline_subdir"
                        if ! eval_logged "$TASK_NAME" "baseline" "$variant_name" "$EVAL_TYPE"; then
                            results=$(extract_from_json "$json_file")
                            if [ -n "$results" ]; then
                                key="Task: ${TASK_NAME} | Run: baseline | Variant: ${variant_name} | EvalType: ${EVAL_TYPE}"
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
                                echo ">>> [OK] 已提取结果: baseline/${variant_name}"
                            fi
                        else
                            echo ">>> [SKIP] 结果已记录: baseline/${variant_name}"
                        fi
                    fi
                fi
            done
        fi
    fi
done

echo ""
echo "=========================================="
echo "提取完成！"
echo "汇总日志: ${EVAL_RESULTS_LOG}"
echo "=========================================="

































