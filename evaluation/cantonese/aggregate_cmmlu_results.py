#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将CMMLU的22个子类别合并为几个代表性类别，并生成最终结果
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# 定义类别映射
CATEGORY_MAPPING = {
    # Humanities (人文学科)
    'humanities': [
        'arts', 'chinese_literature', 'philosophy', 'world_history', 'world_religions'
    ],
    # Social Science (社会科学)
    'social_science': [
        'economics', 'education', 'ethnology', 'high_school_geography', 'journalism',
        'management', 'marketing', 'marxist_theory', 'professional_psychology',
        'security_study', 'sociology'
    ],
    # STEM (科学、技术、工程、数学)
    'stem': [
        'college_medicine', 'electrical_engineering', 'machine_learning'
    ],
    # Others (其他)
    'others': [
        'chinese_civil_service_exam', 'logical', 'sports_science'
    ]
}

def load_results(json_path):
    """加载JSON结果文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def aggregate_by_category(subjects_data, category_mapping):
    """按类别聚合结果"""
    category_results = defaultdict(lambda: {
        'total_correct': 0,
        'total_samples': 0,
        'subjects': []
    })
    
    # 反向映射：从subject到category
    subject_to_category = {}
    for category, subjects in category_mapping.items():
        for subject in subjects:
            subject_to_category[subject] = category
    
    # 聚合数据
    for subject, data in subjects_data.items():
        if subject in subject_to_category:
            category = subject_to_category[subject]
            category_results[category]['total_correct'] += data['n_correct']
            category_results[category]['total_samples'] += data['n_samples']
            category_results[category]['subjects'].append(subject)
    
    # 计算准确率
    for category in category_results:
        total = category_results[category]['total_samples']
        if total > 0:
            category_results[category]['accuracy'] = (
                category_results[category]['total_correct'] / total * 100
            )
        else:
            category_results[category]['accuracy'] = 0.0
    
    return dict(category_results)

def main():
    base_dir = Path('/data/chenhang/codes/Yue-Benchmark/output_llama3.2-1bcanwithllama8b-instruct-dialougedata_final_nolora_same_param_withllama8b/alpha0.1_lr1e-8')
    ablation_path = base_dir / 'ablation/cmmlu_eval/cmmlu_yue_results.json'
    hot_path = base_dir / 'hot/cmmlu_eval/cmmlu_yue_results.json'
    base_path = Path('/data/chenhang/codes/Yue-Benchmark/modelA/cmmlu_eval/cmmlu_yue_results.json')
    
    # 加载数据
    ablation_data = load_results(ablation_path)
    hot_data = load_results(hot_path)
    base_data = load_results(base_path)
    
    ablation_subjects = ablation_data['ablation']['0shot']['subjects']
    hot_subjects = hot_data['hot']['0shot']['subjects']
    base_subjects = base_data['modelA']['0shot']['subjects']
    
    # 按类别聚合
    base_categories = aggregate_by_category(base_subjects, CATEGORY_MAPPING)
    ablation_categories = aggregate_by_category(ablation_subjects, CATEGORY_MAPPING)
    hot_categories = aggregate_by_category(hot_subjects, CATEGORY_MAPPING)
    
    # 类别名称映射（用于显示）
    category_names = {
        'humanities': 'Humanities',
        'social_science': 'Social Science',
        'stem': 'STEM',
        'others': 'Others'
    }
    
    # 打印表格
    print('\n' + '='*120)
    print('CMMLU评估结果 - 按类别聚合 (alpha0.1_lr1e-8)')
    print('='*120)
    print(f"{'Category':<20} {'Base Acc':>12} {'Base C/T':>15} {'Ablation Acc':>14} {'Ablation C/T':>17} {'Hot Acc':>12} {'Hot C/T':>15} {'Hot-Base':>10} {'Hot-Abl':>10}")
    print('-'*120)
    
    categories_order = ['humanities', 'social_science', 'stem', 'others']
    rows_data = []
    
    for cat_key in categories_order:
        cat_name = category_names[cat_key]
        base_cat = base_categories[cat_key]
        abl_cat = ablation_categories[cat_key]
        hot_cat = hot_categories[cat_key]
        
        diff_hot_base = hot_cat['accuracy'] - base_cat['accuracy']
        diff_hot_abl = hot_cat['accuracy'] - abl_cat['accuracy']
        
        print(f"{cat_name:<20} {base_cat['accuracy']:>11.2f}% {base_cat['total_correct']:>4}/{base_cat['total_samples']:<4} "
              f"{abl_cat['accuracy']:>13.2f}% {abl_cat['total_correct']:>4}/{abl_cat['total_samples']:<4} "
              f"{hot_cat['accuracy']:>11.2f}% {hot_cat['total_correct']:>4}/{hot_cat['total_samples']:<4} "
              f"{diff_hot_base:>+9.2f} {diff_hot_abl:>+9.2f}")
        
        rows_data.append({
            'category': cat_name,
            'base_acc': base_cat['accuracy'],
            'base_correct': base_cat['total_correct'],
            'base_total': base_cat['total_samples'],
            'ablation_acc': abl_cat['accuracy'],
            'ablation_correct': abl_cat['total_correct'],
            'ablation_total': abl_cat['total_samples'],
            'hot_acc': hot_cat['accuracy'],
            'hot_correct': hot_cat['total_correct'],
            'hot_total': hot_cat['total_samples'],
            'diff_hot_base': diff_hot_base,
            'diff_hot_abl': diff_hot_abl
        })
    
    # 总体平均
    base_summary = base_data['modelA']['0shot']['summary']
    abl_summary = ablation_data['ablation']['0shot']['summary']
    hot_summary = hot_data['hot']['0shot']['summary']
    diff_hot_base_avg = hot_summary['average_accuracy'] - base_summary['average_accuracy']
    diff_hot_abl_avg = hot_summary['average_accuracy'] - abl_summary['average_accuracy']
    
    print('-'*120)
    print(f"{'=== AVERAGE ===':<20} {base_summary['average_accuracy']:>11.2f}% {base_summary['total_correct']:>4}/{base_summary['total_samples']:<4} "
          f"{abl_summary['average_accuracy']:>13.2f}% {abl_summary['total_correct']:>4}/{abl_summary['total_samples']:<4} "
          f"{hot_summary['average_accuracy']:>11.2f}% {hot_summary['total_correct']:>4}/{hot_summary['total_samples']:<4} "
          f"{diff_hot_base_avg:>+9.2f} {diff_hot_abl_avg:>+9.2f}")
    print('='*120)
    
    # 创建柱状图
    categories = [category_names[cat] for cat in categories_order]
    base_values = [base_categories[cat]['accuracy'] for cat in categories_order]
    ablation_values = [ablation_categories[cat]['accuracy'] for cat in categories_order]
    hot_values = [hot_categories[cat]['accuracy'] for cat in categories_order]
    
    x = np.arange(len(categories))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, base_values, width, label='Base Model', alpha=0.8)
    plt.bar(x, ablation_values, width, label='Ablation', alpha=0.8)
    plt.bar(x + width, hot_values, width, label='Hot', alpha=0.8)
    
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('CMMLU Category Accuracies (alpha0.1_lr1e-8)', fontsize=14, fontweight='bold')
    plt.xticks(x, categories, rotation=15, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_dir = Path('/home/chenhang/optimal_trans/vis_res')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'cmmlu_category_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存到: {output_path}")
    
    # 保存CSV
    import csv
    csv_path = output_dir / 'cmmlu_category_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Base Acc (%)', 'Base Correct/Total', 
                        'Ablation Acc (%)', 'Ablation Correct/Total',
                        'Hot Acc (%)', 'Hot Correct/Total', 'Hot-Base Diff', 'Hot-Ablation Diff'])
        for row in rows_data:
            writer.writerow([
                row['category'],
                f"{row['base_acc']:.2f}",
                f"{row['base_correct']}/{row['base_total']}",
                f"{row['ablation_acc']:.2f}",
                f"{row['ablation_correct']}/{row['ablation_total']}",
                f"{row['hot_acc']:.2f}",
                f"{row['hot_correct']}/{row['hot_total']}",
                f"{row['diff_hot_base']:+.2f}",
                f"{row['diff_hot_abl']:+.2f}"
            ])
        # 添加平均行
        writer.writerow([
            'Average',
            f"{base_summary['average_accuracy']:.2f}",
            f"{base_summary['total_correct']}/{base_summary['total_samples']}",
            f"{abl_summary['average_accuracy']:.2f}",
            f"{abl_summary['total_correct']}/{abl_summary['total_samples']}",
            f"{hot_summary['average_accuracy']:.2f}",
            f"{hot_summary['total_correct']}/{hot_summary['total_samples']}",
            f"{diff_hot_base_avg:+.2f}",
            f"{diff_hot_abl_avg:+.2f}"
        ])
    
    print(f"CSV文件已保存到: {csv_path}")
    
    # 打印详细统计
    print("\n" + "="*120)
    print("详细统计信息")
    print("="*120)
    for cat_key in categories_order:
        cat_name = category_names[cat_key]
        base_cat = base_categories[cat_key]
        abl_cat = ablation_categories[cat_key]
        hot_cat = hot_categories[cat_key]
        
        print(f"\n{cat_name}:")
        print(f"  包含子类别: {', '.join([s.replace('_', ' ').title() for s in base_cat['subjects']])}")
        print(f"  Base: {base_cat['accuracy']:.2f}% ({base_cat['total_correct']}/{base_cat['total_samples']})")
        print(f"  Ablation: {abl_cat['accuracy']:.2f}% ({abl_cat['total_correct']}/{abl_cat['total_samples']}) "
              f"[vs Base: {abl_cat['accuracy'] - base_cat['accuracy']:+.2f}%]")
        print(f"  Hot: {hot_cat['accuracy']:.2f}% ({hot_cat['total_correct']}/{hot_cat['total_samples']}) "
              f"[vs Base: {hot_cat['accuracy'] - base_cat['accuracy']:+.2f}%, "
              f"vs Ablation: {hot_cat['accuracy'] - abl_cat['accuracy']:+.2f}%]")
    
    print("\n" + "="*120)
    print("总结")
    print("="*120)
    print(f"Base Model平均: {base_summary['average_accuracy']:.2f}%")
    print(f"Ablation平均: {abl_summary['average_accuracy']:.2f}% (vs Base: {abl_summary['average_accuracy'] - base_summary['average_accuracy']:+.2f}%)")
    print(f"Hot平均: {hot_summary['average_accuracy']:.2f}% (vs Base: {diff_hot_base_avg:+.2f}%, vs Ablation: {diff_hot_abl_avg:+.2f}%)")

if __name__ == '__main__':
    main()
