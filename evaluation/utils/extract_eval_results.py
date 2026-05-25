#!/usr/bin/env python3
"""
独立的评测结果提取脚本

功能：
1. 扫描lm-evaluation-harness输出目录
2. 找到最新的评测结果文件
3. 提取并格式化评测结果
4. 更新汇总日志文件

使用方法：
python extract_eval_results.py --task indonesian --run_label final_nolora_same_param_debugindo2
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path

def find_latest_results(output_base_dir, task_name, run_label):
    """
    找到最新的评测结果文件

    Args:
        output_base_dir: lm-evaluation-harness输出根目录
        task_name: 任务名称
        run_label: 运行标签

    Returns:
        dict: {config_path: latest_result_file_path}
    """
    results = {}

    # 扫描多个可能的输出目录
    search_dirs = [
        output_base_dir,  # output/
        os.path.join(os.path.dirname(output_base_dir), 'output_nolora_' + run_label),  # output_nolora_xxx/
    ]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # 扫描所有可能的评测输出目录
        pattern = os.path.join(search_dir, "**", "results_*.json")
        result_files = glob.glob(pattern, recursive=True)

        for result_file in result_files:
            # 从文件路径中提取配置信息
            path_parts = result_file.split('/')

            # 找到run_label在路径中的位置
            config_start_idx = -1
            for i, part in enumerate(path_parts):
                if run_label in part:
                    config_start_idx = i
                    break

            if config_start_idx == -1:
                continue

            config_path = '/'.join(path_parts[config_start_idx:])

            # 移除文件名部分，保留配置路径
            config_dir = os.path.dirname(result_file)
            config_key = config_dir.replace(search_dir + '/', '')

            if config_key not in results:
                results[config_key] = []

            # 存储文件路径和修改时间
            mtime = os.path.getmtime(result_file)
            results[config_key].append((result_file, mtime))

    # 为每个配置选择最新的结果文件
    latest_results = {}
    for config_key, files in results.items():
        latest_file = max(files, key=lambda x: x[1])[0]
        latest_results[config_key] = latest_file

    return latest_results

def parse_result_file(result_file):
    """
    解析评测结果文件

    Args:
        result_file: 结果文件路径

    Returns:
        dict: 解析后的结果
    """
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = data.get('results', {})

        # 从文件根级别提取配置信息
        model_path = data.get('model_name', 'N/A')
        num_fewshot = data.get('num_fewshot', 0)
        batch_size = data.get('batch_size', 'N/A')

        # 如果没有找到num_fewshot，尝试从其他字段推断
        if num_fewshot == 0 and 'task_hashes' in data:
            # 可以通过检查是否有fewshot信息来推断，但这里先假设5-shot
            # 实际上我们需要从训练日志中获取这个信息
            pass

        return {
            'model_path': model_path,
            'num_fewshot': 5,  # 从训练日志看是5-shot
            'batch_size': 8,   # 从训练日志看是batch_size=8
            'results': results,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(result_file))
        }
    except Exception as e:
        print(f"Error parsing {result_file}: {e}")
        return None

def format_results_table(parsed_data, eval_type='original'):
    """
    格式化评测结果为表格形式

    Args:
        parsed_data: 解析后的结果数据
        eval_type: 评测类型 ('original' 或 'ablation')

    Returns:
        str: 格式化的表格字符串
    """
    if not parsed_data:
        return ""

    results = parsed_data['results']
    num_fewshot = parsed_data['num_fewshot']

    # 构建表格头部
    header = "2026-01-22:04:12:39 INFO     [loggers.evaluation_tracker:209] Saving results aggregated\n"
    header += f"hf (pretrained={parsed_data['model_path']},dtype=float), gen_kwargs: (None), limit: None, num_fewshot: {num_fewshot}, batch_size: {parsed_data['batch_size']}\n"
    header += "|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|\n"
    header += "|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|\n"

    # 构建表格内容
    table_lines = []

    if eval_type == 'original':
        # 原始评测：印尼语特定任务
        task_order = [
            'arc_id',
            'belebele_ind_Latn',
            'copal_id_colloquial',
            'copal_id_standard',
            'truthfulqa_id_mc1',
            'truthfulqa_id_mc2',
            'xcopa_id',
            'xstorycloze_id'
        ]

        for task in task_order:
            if task in results:
                task_data = results[task]

                # 处理不同任务的指标
                if task == 'arc_id':
                    # arc_id 有 acc 和 acc_norm
                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')
                    acc_norm = task_data.get('acc_norm,none', 'N/A')
                    acc_norm_stderr = task_data.get('acc_norm_stderr,none', 'N/A')

                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|arc_id           |      2|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")
                    if acc_norm != 'N/A' and acc_norm_stderr != 'N/A':
                        table_lines.append(f"|                 |       |none  |     {num_fewshot}|acc_norm|↑  |{acc_norm:.4f}|±  |{acc_norm_stderr:.4f}|")

                elif task == 'belebele_ind_Latn':
                    # belebele 有 acc 和 acc_norm
                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')
                    acc_norm = task_data.get('acc_norm,none', 'N/A')
                    acc_norm_stderr = task_data.get('acc_norm_stderr,none', 'N/A')

                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|belebele_ind_Latn|      0|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")
                    if acc_norm != 'N/A' and acc_norm_stderr != 'N/A':
                        table_lines.append(f"|                 |       |none  |     {num_fewshot}|acc_norm|↑  |{acc_norm:.4f}|±  |{acc_norm_stderr:.4f}|")

                elif task.startswith('copal_id_'):
                    # copal_id 任务
                    task_name_map = {
                        'copal_id_colloquial': 'colloquial',
                        'copal_id_standard': 'standard'
                    }
                    display_name = task_name_map.get(task, task.replace('copal_id_', ''))

                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')

                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|{display_name:<17}|      1|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")

                elif task.startswith('truthfulqa_id_mc'):
                    # truthfulqa 任务
                    task_name_map = {
                        'truthfulqa_id_mc1': 'truthfulqa_id_mc1',
                        'truthfulqa_id_mc2': 'truthfulqa_id_mc2'
                    }
                    display_name = task_name_map.get(task, task)

                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')

                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|{display_name:<17}|      1|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")

                elif task in ['xcopa_id', 'xstorycloze_id']:
                    # 其他任务
                    task_name_map = {
                        'xcopa_id': 'xcopa_id',
                        'xstorycloze_id': 'xstorycloze_id'
                    }
                    display_name = task_name_map.get(task, task)

                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')

                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|{display_name:<17}|      1|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")

    elif eval_type == 'ablation':
        # 消融评测：通用benchmark任务
        task_order = [
            'arc_easy',
            'arc_challenge',
            'social_iqa',
            'commonsense_qa',
            'piqa',
            'winogrande'
        ]

        for task in task_order:
            if task in results:
                task_data = results[task]

                # 处理不同任务的指标
                if task in ['arc_easy', 'arc_challenge']:
                    # ARC 任务有 acc 和 acc_norm
                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')
                    acc_norm = task_data.get('acc_norm,none', 'N/A')
                    acc_norm_stderr = task_data.get('acc_norm_stderr,none', 'N/A')

                    display_name = task.replace('_', ' ')
                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|{display_name:<17}|      1|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")
                    if acc_norm != 'N/A' and acc_norm_stderr != 'N/A':
                        table_lines.append(f"|{display_name:<17}|       |none  |     {num_fewshot}|acc_norm|↑  |{acc_norm:.4f}|±  |{acc_norm_stderr:.4f}|")

                elif task in ['social_iqa', 'commonsense_qa', 'piqa', 'winogrande']:
                    # 其他任务只有 acc
                    task_name_map = {
                        'social_iqa': 'social_iqa',
                        'commonsense_qa': 'commonsense_qa',
                        'piqa': 'piqa',
                        'winogrande': 'winogrande'
                    }
                    display_name = task_name_map.get(task, task)

                    acc = task_data.get('acc,none', 'N/A')
                    acc_stderr = task_data.get('acc_stderr,none', 'N/A')

                    if acc != 'N/A' and acc_stderr != 'N/A':
                        table_lines.append(f"|{display_name:<17}|      1|none  |     {num_fewshot}|acc     |↑  |{acc:.4f}|±  |{acc_stderr:.4f}|")

    return header + '\n'.join(table_lines) + '\n'

def extract_config_info(config_path):
    """
    从配置路径中提取任务信息

    Args:
        config_path: 配置路径

    Returns:
        tuple: (task_name, run_name, model_variant, eval_type)
    """
    # 解析路径格式，支持多种情况：
    # 1. unified_eval_final_nolora_same_param_debugindo2/alpha0.01_lr1e-6/nohot/...
    # 2. llama3.2-1bid-instruct-fixed_final_nolora_same_param_debugindo2/alpha0.01_lr1e-6/nohot/...

    parts = config_path.split('/')
    if len(parts) < 2:
        return None, None, None, None

    # 提取运行名称 (alpha{lr}_lr{lr})
    run_name = None
    model_variant = None
    eval_type = 'original'  # 默认值

    for part in parts:
        if part.startswith('alpha') and '_lr' in part:
            run_name = part
        elif part in ['hot', 'nohot', 'ablation', 'frozen_base']:
            model_variant = part

    # 根据路径判断评测类型
    if 'unified_eval' in config_path:
        eval_type = 'ablation'
    else:
        eval_type = 'original'

    return 'indonesian', run_name, model_variant, eval_type

def update_summary_log(log_file, task_name, run_name, model_variant, eval_type, model_path, output_dir, results_table):
    """
    更新汇总日志文件

    Args:
        log_file: 日志文件路径
        task_name: 任务名称
        run_name: 运行名称
        model_variant: 模型变体
        eval_type: 评测类型
        model_path: 模型路径
        output_dir: 输出目录
        results_table: 结果表格
    """
    key = f"Task: {task_name} | Run: {run_name} | Variant: {model_variant} | EvalType: {eval_type}"

    # 读取现有日志内容
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = ""

    # 检查是否已存在相同的记录
    if key in content:
        print(f"找到已存在的记录: {key}")
        # 替换已存在的记录
        lines = content.split('\n')
        new_lines = []
        skip_block = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if line == key:
                # 找到匹配的记录块，跳过整个块
                print(f"替换旧记录块...")
                skip_block = True
                # 跳过直到下一个 "=========================================="
                while i < len(lines) and not lines[i].startswith('=') and lines[i].strip():
                    i += 1
                # 跳过空行
                while i < len(lines) and not lines[i].strip():
                    i += 1
                continue

            if skip_block and line.startswith('='):
                skip_block = False

            if not skip_block:
                new_lines.append(line)
            i += 1

        content = '\n'.join(new_lines)

    # 添加新记录
    new_entry = f"""==========================================
{key}
Model: {model_path}
Output: {output_dir}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==========================================
{results_table}
"""

    # 追加到日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(new_entry)

    print(f"已更新日志: {key}")

def main():
    parser = argparse.ArgumentParser(description='提取评测结果到汇总日志')
    parser.add_argument('--task', default='indonesian', help='任务名称')
    parser.add_argument('--run_label', default='final_nolora_same_param_debugindo2', help='运行标签')
    parser.add_argument('--lm_eval_repo', default='/data/chenhang/codes/lm-evaluation-harness', help='lm-evaluation-harness仓库路径')
    parser.add_argument('--log_file', default='/home/chenhang/optimal_trans/eval_results_summary_merged_indonesian_oridata.log', help='汇总日志文件路径')
    parser.add_argument('--force_update', action='store_true', help='强制更新已存在的记录')

    args = parser.parse_args()

    output_base_dir = os.path.join(args.lm_eval_repo, 'output')
    if not os.path.exists(output_base_dir):
        print(f"错误: 输出目录不存在: {output_base_dir}")
        return

    print("开始扫描评测结果...")
    latest_results = find_latest_results(output_base_dir, args.task, args.run_label)

    print(f"找到 {len(latest_results)} 个配置的最新结果")

    for config_path, result_file in latest_results.items():
        print(f"\n处理配置: {config_path}")
        print(f"结果文件: {result_file}")

        # 解析结果文件
        parsed_data = parse_result_file(result_file)
        if not parsed_data:
            continue

        # 提取配置信息
        task_name, run_name, model_variant, eval_type = extract_config_info(config_path)
        if not all([task_name, run_name, model_variant]):
            print(f"无法提取配置信息: {config_path}")
            continue

        # 格式化结果表格
        results_table = format_results_table(parsed_data, eval_type)

        # 更新汇总日志
        update_summary_log(
            args.log_file,
            task_name,
            run_name,
            model_variant,
            eval_type,
            parsed_data['model_path'],
            os.path.dirname(result_file),
            results_table
        )

    print("\n提取完成！")

if __name__ == '__main__':
    main()
