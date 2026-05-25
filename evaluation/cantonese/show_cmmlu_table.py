#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

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

print('\n' + '='*150)
print('CMMLU评估结果详细对比表 - alpha0.1_lr1e-8 (包含Base Model)')
print('='*150)
print(f"{'Subject':<30} {'Base Acc':>10} {'Base C/T':>12} {'Ablation Acc':>12} {'Ablation C/T':>15} {'Hot Acc':>12} {'Hot C/T':>15} {'Hot-Abl':>8} {'Hot-Base':>8}")
print('-'*150)

for subject in sorted(ablation_subjects.keys()):
    base = base_subjects[subject]
    abl = ablation_subjects[subject]
    hot = hot_subjects[subject]
    diff_hot_abl = hot['accuracy'] - abl['accuracy']
    diff_hot_base = hot['accuracy'] - base['accuracy']
    subject_name = subject.replace('_', ' ').title()
    print(f"{subject_name:<30} {base['accuracy']:>9.2f}% {base['n_correct']:>3}/{base['n_samples']:<4} {abl['accuracy']:>11.2f}% {abl['n_correct']:>4}/{abl['n_samples']:<4} {hot['accuracy']:>11.2f}% {hot['n_correct']:>4}/{hot['n_samples']:<4} {diff_hot_abl:>+7.2f} {diff_hot_base:>+7.2f}")

base_summary = base_data['modelA']['0shot']['summary']
abl_summary = ablation_data['ablation']['0shot']['summary']
hot_summary = hot_data['hot']['0shot']['summary']
diff_hot_abl_avg = hot_summary['average_accuracy'] - abl_summary['average_accuracy']
diff_hot_base_avg = hot_summary['average_accuracy'] - base_summary['average_accuracy']
print('-'*150)
print(f"{'=== AVERAGE ===':<30} {base_summary['average_accuracy']:>9.2f}% {base_summary['total_correct']:>3}/{base_summary['total_samples']:<4} {abl_summary['average_accuracy']:>11.2f}% {abl_summary['total_correct']:>4}/{abl_summary['total_samples']:<4} {hot_summary['average_accuracy']:>11.2f}% {hot_summary['total_correct']:>4}/{hot_summary['total_samples']:<4} {diff_hot_abl_avg:>+7.2f} {diff_hot_base_avg:>+7.2f}")
print('='*150)

# 统计信息
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

improved_abl = sum(1 for _, diff_abl, _, _, _, _ in improvements if diff_abl > 0)
declined_abl = sum(1 for _, diff_abl, _, _, _, _ in improvements if diff_abl < 0)
unchanged_abl = sum(1 for _, diff_abl, _, _, _, _ in improvements if diff_abl == 0)

improved_base = sum(1 for _, _, diff_base, _, _, _ in improvements if diff_base > 0)
declined_base = sum(1 for _, _, diff_base, _, _, _ in improvements if diff_base < 0)

print(f"\n相对于Ablation: 改进={improved_abl}, 下降={declined_abl}, 不变={unchanged_abl}")
print(f"相对于Base Model: 改进={improved_base}, 下降={declined_base}")
print(f"\nBase Model平均: {base_summary['average_accuracy']:.2f}%")
print(f"Ablation平均: {abl_summary['average_accuracy']:.2f}% (vs Base: {abl_summary['average_accuracy'] - base_summary['average_accuracy']:+.2f}%)")
print(f"Hot平均: {hot_summary['average_accuracy']:.2f}% (vs Base: {diff_hot_base_avg:+.2f}%, vs Ablation: {diff_hot_abl_avg:+.2f}%)")
