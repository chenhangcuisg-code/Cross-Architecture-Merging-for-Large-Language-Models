#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建CMMLU评估结果的详细对比表格
"""
import json
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path('/data/chenhang/codes/Yue-Benchmark/output_llama3.2-1bcanwithllama8b-instruct-dialougedata_final_nolora_same_param_withllama8b/alpha0.1_lr1e-8')
    ablation_path = base_dir / 'ablation/cmmlu_eval/cmmlu_yue_results.json'
    hot_path = base_dir / 'hot/cmmlu_eval/cmmlu_yue_results.json'
    base_path = Path('/data/chenhang/codes/Yue-Benchmark/modelA/cmmlu_eval/cmmlu_yue_results.json')
    
    with open(ablation_path) as f:
        ablation_data = json.load(f)
    with open(hot_path) as f:
        hot_data = json.load(f)
    with open(base_path) as f:
        base_data = json.load(f)
    
    ablation_subjects = ablation_data['ablation']['0shot']['subjects']
    hot_subjects = hot_data['hot']['0shot']['subjects']
    base_subjects = base_data['modelA']['0shot']['subjects']
    
    rows = []
    for subject in sorted(ablation_subjects.keys()):
        base = base_subjects[subject]
        abl = ablation_subjects[subject]
        hot = hot_subjects[subject]
        diff_hot_abl = hot['accuracy'] - abl['accuracy']
        diff_hot_base = hot['accuracy'] - base['accuracy']
        
        rows.append({
            'Subject': subject.replace('_', ' ').title(),
            'Base Acc (%)': f'{base["accuracy"]:.2f}',
            'Base Correct/Total': f'{base["n_correct"]}/{base["n_samples"]}',
            'Ablation Acc (%)': f'{abl["accuracy"]:.2f}',
            'Ablation Correct/Total': f'{abl["n_correct"]}/{abl["n_samples"]}',
            'Hot Acc (%)': f'{hot["accuracy"]:.2f}',
            'Hot Correct/Total': f'{hot["n_correct"]}/{hot["n_samples"]}',
            'Hot-Ablation Diff': f'{diff_hot_abl:+.2f}',
            'Hot-Base Diff': f'{diff_hot_base:+.2f}'
        })
    
    # 添加汇总行
    base_summary = base_data['modelA']['0shot']['summary']
    abl_summary = ablation_data['ablation']['0shot']['summary']
    hot_summary = hot_data['hot']['0shot']['summary']
    diff_hot_abl_avg = hot_summary['average_accuracy'] - abl_summary['average_accuracy']
    diff_hot_base_avg = hot_summary['average_accuracy'] - base_summary['average_accuracy']
    
    rows.append({
        'Subject': '=== AVERAGE ===',
        'Base Acc (%)': f'{base_summary["average_accuracy"]:.2f}',
        'Base Correct/Total': f'{base_summary["total_correct"]}/{base_summary["total_samples"]}',
        'Ablation Acc (%)': f'{abl_summary["average_accuracy"]:.2f}',
        'Ablation Correct/Total': f'{abl_summary["total_correct"]}/{abl_summary["total_samples"]}',
        'Hot Acc (%)': f'{hot_summary["average_accuracy"]:.2f}',
        'Hot Correct/Total': f'{hot_summary["total_correct"]}/{hot_summary["total_samples"]}',
        'Hot-Ablation Diff': f'{diff_hot_abl_avg:+.2f}',
        'Hot-Base Diff': f'{diff_hot_base_avg:+.2f}'
    })
    
    df = pd.DataFrame(rows)
    
    # 打印表格
    print("\n" + "="*150)
    print("CMMLU评估结果详细对比表 - alpha0.1_lr1e-8 (包含Base Model)")
    print("="*150)
    print(df.to_string(index=False))
    print("="*150)
    
    # 保存到CSV
    output_csv = base_dir / 'cmmlu_detailed_comparison.csv'
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_csv}")
    except Exception as e:
        print(f"\n无法保存CSV文件: {e}")
    
    # 保存到Markdown
    output_md = base_dir / 'cmmlu_detailed_comparison.md'
    try:
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write("# CMMLU评估结果详细对比表\n\n")
            f.write("**配置**: alpha0.1_lr1e-8\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Markdown格式已保存到: {output_md}")
    except Exception as e:
        print(f"\n无法保存Markdown文件: {e}")
    
    # 统计信息
    print("\n" + "="*150)
    print("统计摘要")
    print("="*150)
    print(f"总子类别数: {len(df) - 1}")
    print(f"Base Model平均准确率: {base_summary['average_accuracy']:.2f}%")
    print(f"Ablation平均准确率: {abl_summary['average_accuracy']:.2f}% (vs Base: {abl_summary['average_accuracy'] - base_summary['average_accuracy']:+.2f}%)")
    print(f"Hot平均准确率: {hot_summary['average_accuracy']:.2f}% (vs Base: {diff_hot_base_avg:+.2f}%, vs Ablation: {diff_hot_abl_avg:+.2f}%)")
    
    # 改进分析
    improvements = []
    for subject in sorted(ablation_subjects.keys()):
        base = base_subjects[subject]
        abl = ablation_subjects[subject]
        hot = hot_subjects[subject]
        diff_hot_abl = hot['accuracy'] - abl['accuracy']
        diff_hot_base = hot['accuracy'] - base['accuracy']
        improvements.append((subject, diff_hot_abl, diff_hot_base, base['accuracy'], abl['accuracy'], hot['accuracy']))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print("\n相对于Ablation改进最大的5个子类别:")
    for i, (subject, diff_abl, diff_base, base_acc, abl_acc, hot_acc) in enumerate(improvements[:5], 1):
        print(f"  {i}. {subject.replace('_', ' ').title()}: {abl_acc:.2f}% → {hot_acc:.2f}% ({diff_abl:+.2f}%)")
    
    print("\n相对于Ablation下降最大的5个子类别:")
    for i, (subject, diff_abl, diff_base, base_acc, abl_acc, hot_acc) in enumerate(improvements[-5:], 1):
        print(f"  {i}. {subject.replace('_', ' ').title()}: {abl_acc:.2f}% → {hot_acc:.2f}% ({diff_abl:+.2f}%)")
    
    print("\n相对于Base Model改进最大的5个子类别:")
    improvements_base = sorted(improvements, key=lambda x: x[2], reverse=True)
    for i, (subject, diff_abl, diff_base, base_acc, abl_acc, hot_acc) in enumerate(improvements_base[:5], 1):
        print(f"  {i}. {subject.replace('_', ' ').title()}: {base_acc:.2f}% → {hot_acc:.2f}% ({diff_base:+.2f}%)")
    
    # 统计改进/下降/不变的数量
    improved_abl = sum(1 for _, diff_abl, _, _, _, _ in improvements if diff_abl > 0)
    declined_abl = sum(1 for _, diff_abl, _, _, _, _ in improvements if diff_abl < 0)
    unchanged_abl = sum(1 for _, diff_abl, _, _, _, _ in improvements if diff_abl == 0)
    
    improved_base = sum(1 for _, _, diff_base, _, _, _ in improvements if diff_base > 0)
    declined_base = sum(1 for _, _, diff_base, _, _, _ in improvements if diff_base < 0)
    
    print(f"\n相对于Ablation: 改进={improved_abl}, 下降={declined_abl}, 不变={unchanged_abl}")
    print(f"相对于Base Model: 改进={improved_base}, 下降={declined_base}")

if __name__ == '__main__':
    main()
