#!/usr/bin/env python
"""
通用的评估结果收集脚本

递归遍历指定目录下的所有 results_*.json 文件，
自动识别指标（排除 stderr），按主要指标排序并输出到 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _iter_result_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    """
    递归遍历目录，找到所有 results_*.json 文件
    
    返回: (文件路径, 相对于root的路径)
    """
    root = root.resolve()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith("results_") and filename.endswith(".json"):
                file_path = Path(dirpath) / filename
                rel_path = file_path.relative_to(root)
                yield file_path, rel_path


def _load_results(path: Path) -> Dict[str, Any]:
    """加载 JSON 结果文件"""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as e:
        logging.warning(f"无法加载 {path}: {e}")
        return {}


def _collect_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    从结果中收集所有指标（排除 stderr）
    
    返回: {task_alias: {metric: value}}
    """
    details: Dict[str, Dict[str, float]] = {}
    
    results_dict = results.get("results", results)
    
    for task, metadata in results_dict.items():
        if not isinstance(metadata, dict):
            continue
        
        alias = metadata.get("alias", task)
        if not alias or not isinstance(alias, str):
            alias = task
        
        for metric, value in metadata.items():
            # 跳过 stderr 指标
            if "stderr" in metric.lower() or "std" in metric.lower():
                continue
            
            # 只处理数值类型
            if isinstance(value, (int, float)):
                details.setdefault(alias, {})[metric] = float(value)
    
    return details


def _get_primary_metric(metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
    """
    自动选择主要指标用于排序
    
    优先级: acc_norm,none > acc,none > 其他 acc 相关 > 其他
    """
    all_metrics = set()
    for task_metrics in metrics.values():
        all_metrics.update(task_metrics.keys())
    
    # 优先级列表
    priority_metrics = [
        "acc_norm,none",
        "acc,none",
    ]
    
    # 先尝试优先级指标
    for metric in priority_metrics:
        if metric in all_metrics:
            # 检查是否有足够的任务包含这个指标
            count = sum(1 for task_metrics in metrics.values() if metric in task_metrics)
            if count > 0:
                return metric
    
    # 如果没有找到，找所有 acc 相关的指标
    acc_metrics = [m for m in all_metrics if "acc" in m.lower() and "," in m]
    if acc_metrics:
        # 选择出现频率最高的
        metric_counts = defaultdict(int)
        for task_metrics in metrics.values():
            for metric in acc_metrics:
                if metric in task_metrics:
                    metric_counts[metric] += 1
        if metric_counts:
            return max(metric_counts.items(), key=lambda x: x[1])[0]
    
    # 最后选择任意一个指标
    if all_metrics:
        return sorted(all_metrics)[0]
    
    return None


def _calculate_mean_score(
    metrics: Dict[str, Dict[str, float]], 
    metric_name: Optional[str]
) -> Optional[float]:
    """计算指定指标在所有任务上的平均值"""
    if not metric_name:
        return None
    
    values = [
        task_metrics[metric_name] 
        for task_metrics in metrics.values() 
        if metric_name in task_metrics
    ]
    
    if not values:
        return None
    
    return sum(values) / len(values)


def collect_all_results(
    root_dir: Path,
    metric_priority: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Set[str], Set[str]]:
    """
    收集所有评估结果
    
    返回: (结果列表, 所有任务别名集合, 所有指标集合)
    """
    all_results: List[Dict[str, Any]] = []
    all_tasks: Set[str] = set()
    all_metrics: Set[str] = set()
    
    logging.info(f"开始遍历目录: {root_dir}")
    
    for file_path, rel_path in _iter_result_files(root_dir):
        logging.debug(f"处理文件: {rel_path}")
        
        results_data = _load_results(file_path)
        if not results_data:
            continue
        
        metrics = _collect_metrics(results_data)
        if not metrics:
            continue
        
        # 更新全局集合
        all_tasks.update(metrics.keys())
        for task_metrics in metrics.values():
            all_metrics.update(task_metrics.keys())
        
        # 选择主要指标
        primary_metric = _get_primary_metric(metrics)
        if metric_priority:
            # 如果指定了优先级，优先使用
            for metric in metric_priority:
                if metric in all_metrics:
                    primary_metric = metric
                    break
        
        mean_score = _calculate_mean_score(metrics, primary_metric)
        
        all_results.append({
            "file_path": str(file_path),
            "rel_path": str(rel_path),
            "dir_path": str(rel_path.parent) if rel_path.parent != Path(".") else "",
            "metrics": metrics,
            "primary_metric": primary_metric,
            "mean_score": mean_score,
            "task_count": len(metrics),
        })
    
    logging.info(f"找到 {len(all_results)} 个结果文件")
    
    # 按主要指标排序
    if all_results:
        # 确定全局主要指标
        global_primary = all_results[0]["primary_metric"]
        if metric_priority:
            for metric in metric_priority:
                if any(metric in r["metrics"] for r in all_results):
                    global_primary = metric
                    break
        
        # 重新计算所有结果的主要指标分数
        for result in all_results:
            if global_primary:
                result["mean_score"] = _calculate_mean_score(result["metrics"], global_primary)
                result["primary_metric"] = global_primary
        
        # 排序
        all_results.sort(
            key=lambda r: r["mean_score"] if r["mean_score"] is not None else float("-inf"),
            reverse=True
        )
    
    return all_results, all_tasks, all_metrics


def write_csv(
    results: List[Dict[str, Any]],
    all_tasks: Set[str],
    all_metrics: Set[str],
    csv_path: Path,
    primary_metric: Optional[str] = None,
) -> None:
    """将结果写入 CSV 文件"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建列名
    # 基础列
    base_columns = [
        "file_path",
        "rel_path",
        "dir_path",
        "primary_metric",
        "mean_score",
        "task_count",
    ]
    
    # 按任务和指标组织列
    task_columns: List[Tuple[str, str, str]] = []  # (task, metric, column_name)
    sorted_tasks = sorted(all_tasks)
    sorted_metrics = sorted(all_metrics)
    
    for task in sorted_tasks:
        for metric in sorted_metrics:
            column_name = f"{task}__{metric}"
            task_columns.append((task, metric, column_name))
    
    fieldnames = base_columns + [col for _, _, col in task_columns]
    
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row: Dict[str, Any] = {
                "file_path": result["file_path"],
                "rel_path": result["rel_path"],
                "dir_path": result["dir_path"],
                "primary_metric": result["primary_metric"] or "",
                "mean_score": "" if result["mean_score"] is None else f"{result['mean_score']:.6f}",
                "task_count": result["task_count"],
            }
            
            # 填充任务指标
            task_metrics = result["metrics"]
            for task, metric, column_name in task_columns:
                value = task_metrics.get(task, {}).get(metric)
                row[column_name] = "" if value is None else f"{value:.6f}"
            
            writer.writerow(row)
    
    logging.info(f"已写入 CSV: {csv_path}")


def render_summary(
    results: List[Dict[str, Any]],
    primary_metric: Optional[str],
    top_k: Optional[int] = None,
) -> str:
    """生成文本摘要"""
    lines = []
    
    if primary_metric:
        lines.append(f"主要指标: {primary_metric}")
    lines.append(f"总共找到 {len(results)} 个结果文件")
    lines.append("")
    
    if top_k:
        results = results[:top_k]
        lines.append(f"显示前 {top_k} 个结果:")
        lines.append("")
    
    # 表头
    header = ["排名", "相对路径", "主要指标均值", "任务数"]
    template = "{:>4} {:<60} {:>12} {:>6}"
    lines.append(template.format(*header))
    lines.append(template.format(*["-" * len(h) for h in header]))
    
    # 数据行
    for idx, result in enumerate(results, 1):
        mean_score = result["mean_score"]
        score_str = f"{mean_score * 100:.2f}%" if mean_score is not None else "N/A"
        lines.append(
            template.format(
                idx,
                result["rel_path"][:60],
                score_str,
                result["task_count"],
            )
        )
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用评估结果收集工具：递归遍历目录，收集所有 results_*.json 并输出 CSV"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default='/data/chenhang/codes/lm-evaluation-harness/output/llama3.2-1bid-instruct-ori',
        help="评估结果根目录（将递归遍历所有子目录）",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default='eval_results_indo.csv',
        help="输出 CSV 文件路径（默认：根目录下的 eval_results.csv）",
    )
    parser.add_argument(
        "--metric",
        action="append",
        help="指定主要指标优先级（可多次指定，按顺序尝试）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="在摘要中只显示前 k 个结果",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印摘要，只生成 CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志",
    )
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # 确定 CSV 输出路径
    if args.csv_path is None:
        args.csv_path = args.root / "eval_results.csv"
    
    # 收集结果
    results, all_tasks, all_metrics = collect_all_results(
        args.root,
        metric_priority=args.metric,
    )
    
    if not results:
        logging.warning("未找到任何结果文件")
        return
    
    # 确定主要指标
    primary_metric = results[0]["primary_metric"] if results else None
    if args.metric:
        for metric in args.metric:
            if metric in all_metrics:
                primary_metric = metric
                break
    
    # 打印摘要
    if not args.quiet:
        print(render_summary(results, primary_metric, args.top_k))
        print()
    
    # 写入 CSV
    write_csv(results, all_tasks, all_metrics, args.csv_path, primary_metric)
    
    logging.info(f"完成！共处理 {len(results)} 个结果文件")
    logging.info(f"发现 {len(all_tasks)} 个任务，{len(all_metrics)} 种指标")


if __name__ == "__main__":
    main()
