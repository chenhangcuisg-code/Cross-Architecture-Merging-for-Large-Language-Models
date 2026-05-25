#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析CMMLU评估结果，生成包含所有子类别的详细对比表格
"""
import json
import pandas as pd
from pathlib import Path

def load_results(json_path):
    """加载JSON结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_comparison_table(ablation_path, hot_path, output_path=None):
    """创建对比表格"""
    # 加载数据
    ablation_data = load_results(ablation_path)
    hot_data = load_results(hot_path)
    
    # 提取子类别数据
    ablation_subjects = ablation_data['ablation']['0shot']['subjects']
    hot_subjects = hot_data['hot']['0shot']['subjects']
    
    # 创建表格数据
    rows = []
    for subject in sorted(ablation_subjects.keys()):
        abl = ablation_subjects[subject]
        hot = hot_subjects[subject]
        
        # 计算差异
        diff = hot['accuracy'] - abl['accuracy']
        diff_pct = (diff / abl['accuracy'] * 100) if abl['accuracy'] > 0 else 0
        
        rows.append({
            'Subject': subject.replace('_', ' ').title(),
            'Ablation Accuracy (%)': f"{abl['accuracy']:.2f}",
            'Ablation Correct': abl['n_correct'],
            'Ablation Total': abl['n_samples'],
            'Hot Accuracy (%)': f"{hot['accuracy']:.2f}",
            'Hot Correct': hot['n_correct'],
            'Hot Total': hot['n_samples'],
            'Difference': f"{diff:+.2f}",
            'Diff %': f"{diff_pct:+.2f}%"
        })
    
    # 添加汇总行
    abl_summary = ablation_data['ablation']['0shot']['summary']
    hot_summary = hot_data['hot']['0shot']['summary']
    diff_avg = hot_summary['average_accuracy'] - abl_summary['average_accuracy']
    diff_avg_pct = (diff_avg / abl_summary['average_accuracy'] * 100) if abl_summary['average_accuracy'] > 0 else 0
    
    rows.append({
        'Subject': '=== AVERAGE ===',
        'Ablation Accuracy (%)': f"{abl_summary['average_accuracy']:.2f}",
        'Ablation Correct': abl_summary['total_correct'],
        'Ablation Total': abl_summary['total_samples'],
        'Hot Accuracy (%)': f"{hot_summary['average_accuracy']:.2f}",
        'Hot Correct': hot_summary['total_correct'],
        'Hot Total': hot_summary['total_samples'],
        'Difference': f"{diff_avg:+.2f}",
        'Diff %': f"{diff_avg_pct:+.2f}%"
    })
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 打印表格
    print("\n" + "="*120)
    print("CMMLU评估结果详细对比表")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)
    
    # 保存到CSV
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_path}")
    
    # 保存到Markdown格式
    if output_path:
        md_path = output_path.replace('.csv', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# CMMLU评估结果详细对比表\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Markdown格式已保存到: {md_path}")
    
    return df

if __name__ == '__main__':
    # 设置路径
    base_dir = Path('/data/chenhang/codes/Yue-Benchmark/output_llama3.2-1bcanwithllama8b-instruct-dialougedata_final_nolora_same_param_withllama8b/alpha0.1_lr1e-8')
    
    ablation_path = base_dir / 'ablation/cmmlu_eval/cmmlu_yue_results.json'
    hot_path = base_dir / 'hot/cmmlu_eval/cmmlu_yue_results.json'
    output_path = base_dir / 'cmmlu_detailed_comparison.csv'
    
    # 加载数据用于统计
    ablation_data = load_results(ablation_path)
    hot_data = load_results(hot_path)
    ablation_subjects = ablation_data['ablation']['0shot']['subjects']
    hot_subjects = hot_data['hot']['0shot']['subjects']
    
    # 创建对比表格
    df = create_comparison_table(ablation_path, hot_path, output_path)
    
    # 打印统计信息
    print("\n" + "="*120)
    print("统计摘要")
    print("="*120)
    print(f"总子类别数: {len(df) - 1}")
    abl_avg = ablation_data['ablation']['0shot']['summary']['average_accuracy']
    hot_avg = hot_data['hot']['0shot']['summary']['average_accuracy']
    print(f"Ablation平均准确率: {abl_avg:.2f}%")
    print(f"Hot平均准确率: {hot_avg:.2f}%")
    print(f"平均差异: {hot_avg - abl_avg:+.2f}%")
    
    # 找出改进最大的子类别
    improvements = []
    for subject in sorted(ablation_subjects.keys()):
        abl = ablation_subjects[subject]
        hot = hot_subjects[subject]
        diff = hot['accuracy'] - abl['accuracy']
        improvements.append((subject, diff))
    
    improvements.sort(key=lambda x: x[1], reverse=True)
    print("\n改进最大的5个子类别:")
    for i, (subject, diff) in enumerate(improvements[:5], 1):
        print(f"  {i}. {subject.replace('_', ' ').title()}: {diff:+.2f}%")
    
    print("\n下降最大的5个子类别:")
    for i, (subject, diff) in enumerate(improvements[-5:], 1):
        print(f"  {i}. {subject.replace('_', ' ').title()}: {diff:+.2f}%")
