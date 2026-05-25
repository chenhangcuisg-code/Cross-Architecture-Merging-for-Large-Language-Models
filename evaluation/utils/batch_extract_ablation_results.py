#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取Ablation模型的评测结果

从现有的结果文件中提取所有ablation模型的结果，并添加到相应的日志文件中。

支持的任务：
- Indonesian (lm_eval格式)
- Malay (MalayMMLU格式)
"""

import json
import csv
import os
import glob
from pathlib import Path
from datetime import datetime
import re


class AblationResultExtractor:
    def __init__(self):
        self.processed_keys = set()

    def extract_lm_eval_result(self, json_file, task_name, run_name, model_variant, out_dir):
        """从lm_eval JSON结果中提取ablation结果"""

        if not os.path.exists(json_file):
            return ""

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = data.get('results', {})
            if not results:
                return ""

            # 构建结果表格 - 使用与正常 Indo 结果相同的格式
            task_results = []
            for task_key, task_data in results.items():
                if 'alias' in task_data:
                    task_name_short = task_data['alias']
                else:
                    task_name_short = task_key.split('/')[-1]

                # 为每个指标生成一行
                first_row = True
                for key in sorted(task_data.keys()):
                    if key == 'alias':
                        continue

                    if ',' in key:
                        metric_part, filter_part = key.split(',', 1)
                        metric_name = metric_part
                        filter_info = filter_part

                        if metric_name.endswith('_stderr'):
                            continue  # stderr 会在对应的主要指标中处理

                        value = task_data[metric_name + ',' + filter_info]
                        stderr_key = metric_name + '_stderr,' + filter_info
                        stderr = task_data.get(stderr_key, 0.0)

                        # 确定版本号（对于大多数任务是固定的）
                        version = 2 if 'arc_id' in task_key else (1 if 'truthfulqa' in task_key or 'copal' in task_key or 'xcopa' in task_key or 'xstorycloze' in task_key else 0)

                        # 确定方向符号
                        direction = "↑" if "acc" in metric_name else "↓"

                        if first_row:
                            task_results.append(f"|{task_name_short:25s}|{version:6d}|none  |     0|{metric_name:8s}|{direction:1s}  |{value:6.4f}|±  |{stderr:6.4f}|")
                            first_row = False
                        else:
                            task_results.append(f"|{'':25s}|{'':6s}|none  |     0|{metric_name:8s}|{direction:1s}  |{value:6.4f}|±  |{stderr:6.4f}|")

            if not task_results:
                return ""

            # 构建完整的日志条目
            key = f"Task: {task_name} | Run: {run_name} | Variant: {model_variant} | EvalType: original"

            # 检查是否已处理过
            if key in self.processed_keys:
                return ""
            self.processed_keys.add(key)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            output = f"""==========================================
{key}
Model: N/A
Output: {out_dir}
Time: {timestamp}
==========================================
2026-01-17:22:38:03 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
hf (pretrained=N/A,dtype=float), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 4
|      Tasks      |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-----------------|------:|------|-----:|--------|---|-----:|---|-----:|
"""

            for result in task_results:
                output += result + "\n"

            output += "\n"
            return output

        except Exception as e:
            print(f"ERROR processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def extract_malay_result_from_csv(self, csv_file, task_name, run_name, model_variant, out_dir):
        """从MalayMMLU CSV结果中提取ablation结果"""

        if not os.path.exists(csv_file):
            return ""

        try:
            # 检查是否是eval_batch.py处理过的CSV文件
            with open(csv_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            if first_line.startswith('dataset,model,train_variant'):  # eval_batch.py处理过的文件
                reader = csv.DictReader(open(csv_file, 'r', encoding='utf-8'))
                rows = list(reader)

                if len(rows) == 0:
                    return ""

                # 找到ablation行
                target_row = None
                for row in rows:
                    if row.get('eval_variant') == 'fused':
                        target_row = row
                        break

                if not target_row:
                    return ""

                # 构建结果
                results = []
                acc_cols = ['accuracy_Humanities', 'accuracy_Language', 'accuracy_Others', 'accuracy_STEM', 'accuracy_Social science']

                for col in acc_cols:
                    if col in target_row and target_row[col]:
                        category = col.replace('accuracy_', '').replace('_', ' ')
                        try:
                            acc_value = float(target_row[col])
                            results.append(f'{category}: {acc_value:.2f}%')
                        except (ValueError, TypeError):
                            continue

            else:  # 原始MalayMMLU CSV文件，需要先处理
                print(f"处理原始CSV文件: {csv_file}")
                import subprocess
                import tempfile

                # 为这个CSV文件运行eval_batch.py
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        result = subprocess.run([
                            'python', '/home/chenhang/MalayMMLU/eval_batch.py',
                            '--pred_files', csv_file,
                            '--shot', '0',
                            '--output_dir', temp_dir
                        ], capture_output=True, text=True, cwd='/home/chenhang/MalayMMLU')

                        if result.returncode != 0:
                            print(f"eval_batch.py失败: {result.stderr}")
                            return ""

                        # 读取生成的accuracy_results_sorted.csv
                        accuracy_file = os.path.join(temp_dir, 'accuracy_results_sorted.csv')
                        if not os.path.exists(accuracy_file):
                            print(f"未找到accuracy文件: {accuracy_file}")
                            return ""

                        # 读取结果
                        reader = csv.DictReader(open(accuracy_file, 'r', encoding='utf-8'))
                        rows = list(reader)

                        if len(rows) == 0:
                            return ""

                        target_row = rows[0]  # 应该只有一行

                        # 构建结果
                        results = []
                        acc_cols = ['accuracy_Humanities', 'accuracy_Language', 'accuracy_Others', 'accuracy_STEM', 'accuracy_Social science']

                        for col in acc_cols:
                            if col in target_row and target_row[col]:
                                category = col.replace('accuracy_', '').replace('_', ' ')
                                try:
                                    acc_value = float(target_row[col])
                                    results.append(f'{category}: {acc_value:.2f}%')
                                except (ValueError, TypeError):
                                    continue

                    except Exception as e:
                        print(f"运行eval_batch.py失败: {e}")
                        return ""

            if not results:
                return ""

            # 构建完整的日志条目
            key = f"Task: {task_name} | Run: {run_name} | Variant: {model_variant} | EvalType: original"

            # 检查是否已处理过
            if key in self.processed_keys:
                return ""
            self.processed_keys.add(key)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            output = f"""==========================================
{key}
Model: /data/chenhang/optimal_trans_new/hot_sft_runs_{task_name.replace('_qwen', '')}_1b/llama_{'malaysian' if 'malay' in task_name else 'indonesian'}_sft_train_alpha_lr_search_final_nolora_same_param{'_qwen' if 'qwen' in task_name else ''}/{run_name}/hot/ablation_untrained_hot_fused
Output: {out_dir}
Time: {timestamp}
==========================================
"""
            for result in results:
                output += result + "\n"
            output += "\n"

            return output

        except Exception as e:
            print(f"ERROR processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def process_indonesian_results(self):
        """处理Indonesian任务的ablation结果"""
        print("处理Indonesian ablation结果...")

        base_dir = "/data/chenhang/codes/lm-evaluation-harness/output_nolora_final_nolora_same_param_debugindo2/llama3.2-1bid-instruct-fixed_final_nolora_same_param_debugindo2"

        results = []

        # 扫描所有alpha目录
        for alpha_dir in sorted(glob.glob(f"{base_dir}/alpha*/")):
            alpha_name = os.path.basename(alpha_dir.rstrip('/'))

            # 只处理指定的alpha值
            alpha_match = re.match(r'alpha(0\.\d+|\d+)_lr', alpha_name)
            if not alpha_match:
                continue

            alpha_value = alpha_match.group(1)
            target_alphas = ['0.01', '0.015', '0.03', '0.05', '0.1', '0.15', '0.2', '0.3']

            if alpha_value not in target_alphas:
                continue

            ablation_dir = os.path.join(alpha_dir, "ablation")
            if not os.path.exists(ablation_dir):
                continue

            # 查找results JSON文件
            json_pattern = os.path.join(ablation_dir, "**", "results_*.json")
            json_files = glob.glob(json_pattern, recursive=True)

            for json_file in json_files:
                result = self.extract_lm_eval_result(
                    json_file,
                    'indonesian',
                    alpha_name,
                    'ablation',
                    ablation_dir
                )
                if result:
                    results.append(result)
                    print(f"✓ 提取了 {alpha_name} 的结果")

        return results

    def process_malay_results(self):
        """处理Malay任务的ablation结果"""
        print("处理Malay ablation结果...")

        results = []

        # 直接查找所有Malay ablation结果文件
        base_dirs = [
            "/home/chenhang/MalayMMLU/output/llama3.2-1bma-instruct",
            "/home/chenhang/MalayMMLU/output/llama3.2-1bma-instruct-fixed-nolora-sameparam_final_nolora_same_param_final_nolora_same_param",
            "/home/chenhang/MalayMMLU/output/llama3.2-1bma-instruct-fixed-nolora-sameparam_final_nolora_same_param_final_nolora_same_param_qwen",
            "/home/chenhang/MalayMMLU/output/llama3.2-1bma-instruct/new_nolora"
        ]

        print(f"扫描目录: {base_dirs}")

        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                print(f"目录不存在: {base_dir}")
                continue

            # 查找所有ablation目录
            ablation_dirs = glob.glob(f"{base_dir}/**/ablation", recursive=True)
            print(f"在 {base_dir} 中找到 {len(ablation_dirs)} 个ablation目录")

            for ablation_dir in ablation_dirs:
                csv_file = os.path.join(ablation_dir, "MalayMMLU_result_ablation_untrained_hot_fused_True_0shot.csv")
                if os.path.exists(csv_file):
                    # 从路径提取run_name
                    parent_dir = os.path.dirname(ablation_dir)
                    run_name = os.path.basename(parent_dir)

                    # 检查是否是我们需要的alpha值
                    alpha_match = re.search(r'alpha(0\.\d+|\d+)_lr', run_name)
                    if alpha_match:
                        alpha_value = alpha_match.group(1)
                        target_alphas = ['0.01', '0.015', '0.03', '0.05', '0.1', '0.15', '0.2', '0.3']

                        if alpha_value in target_alphas:
                            print(f"处理文件: {csv_file}")
                            result = self.extract_malay_result_from_csv(
                                csv_file,
                                'malay',
                                run_name,
                                'ablation',
                                ablation_dir
                            )
                            if result:
                                results.append(result)
                                print(f"✓ 提取了 {run_name} 的结果")
                            else:
                                print(f"✗ 未能提取 {run_name} 的结果")

        return results

    def save_results(self, indonesian_results, malay_results):
        """保存结果到日志文件"""

        # 保存Indonesian结果
        if indonesian_results:
            print(f"\n保存 {len(indonesian_results)} 个Indonesian ablation结果到日志文件...")
            with open('/home/chenhang/optimal_trans/eval_results_summary_merged_indoqwen_oridata.log', 'a', encoding='utf-8') as f:
                for result in indonesian_results:
                    f.write(result)

        # 保存Malay结果
        if malay_results:
            print(f"保存 {len(malay_results)} 个Malay ablation结果到日志文件...")
            with open('/home/chenhang/optimal_trans/eval_results_summary_merged_malaqwen.log', 'a', encoding='utf-8') as f:
                for result in malay_results:
                    f.write(result)

        print("所有结果已保存！")


def main():
    print("开始批量提取ablation结果...")
    print("=" * 50)

    extractor = AblationResultExtractor()

    # 处理Indonesian结果
    indonesian_results = extractor.process_indonesian_results()

    # 处理Malay结果
    malay_results = extractor.process_malay_results()

    # 保存结果
    extractor.save_results(indonesian_results, malay_results)

    print("=" * 50)
    print("批量提取完成！")
    print(f"Indonesian结果: {len(indonesian_results)} 个")
    print(f"Malay结果: {len(malay_results)} 个")
    print(f"总计: {len(indonesian_results) + len(malay_results)} 个")


if __name__ == "__main__":
    main()
