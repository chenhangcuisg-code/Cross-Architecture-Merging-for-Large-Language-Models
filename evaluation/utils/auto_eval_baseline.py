#!/usr/bin/env python
"""
自动评估脚本：对比 baseline 与各实验组，输出表现最好的配置

支持多输出目录（如 output_nolora_final_nolora_general 与 output_nolora_final_nolora_gen_qwen），
分别识别 baseline，计算各实验组相对 baseline 的提升，并输出最优配置。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_results(path: Path) -> Dict[str, Any]:
    """加载 JSON 结果文件"""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as e:
        print(f"无法加载 {path}: {e}")
        return {}


def collect_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """从结果中收集所有指标（排除 stderr）"""
    details: Dict[str, Dict[str, float]] = {}
    results_dict = results.get("results", results)

    for task, metadata in results_dict.items():
        if not isinstance(metadata, dict):
            continue
        alias = metadata.get("alias", task)
        if not alias or not isinstance(alias, str):
            alias = task
        for metric, value in metadata.items():
            if "stderr" in metric.lower() or "std" in metric.lower():
                continue
            if isinstance(value, (int, float)):
                details.setdefault(alias, {})[metric] = float(value)
    return details


def get_primary_metric(metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
    """选择主要指标"""
    all_metrics = set()
    for task_metrics in metrics.values():
        all_metrics.update(task_metrics.keys())
    for m in ["acc_norm,none", "acc,none"]:
        if m in all_metrics and any(m in tm for tm in metrics.values()):
            return m
    acc_metrics = [x for x in all_metrics if "acc" in x.lower() and "," in x]
    if acc_metrics:
        cnt = defaultdict(int)
        for tm in metrics.values():
            for am in acc_metrics:
                if am in tm:
                    cnt[am] += 1
        if cnt:
            return max(cnt.items(), key=lambda x: x[1])[0]
    return sorted(all_metrics)[0] if all_metrics else None


def mean_score(metrics: Dict[str, Dict[str, float]], metric_name: Optional[str]) -> Optional[float]:
    """计算指定指标在所有任务上的平均值"""
    if not metric_name:
        return None
    values = [tm[metric_name] for tm in metrics.values() if metric_name in tm]
    return sum(values) / len(values) if values else None


def is_baseline(rel_path: str) -> bool:
    """根据路径判断是否为 baseline"""
    parts = Path(rel_path).parts
    return "baseline" in [p.lower() for p in parts]


def extract_config(rel_path: str) -> str:
    """从路径提取配置标识（alpha_lr + hot/nohot/ablation）"""
    parts = Path(rel_path).parts
    config_parts = []
    for p in parts:
        if p.startswith("alpha") and "_lr" in p:
            config_parts.append(p)
            break
    for p in parts:
        if p in ("hot", "nohot", "ablation"):
            config_parts.append(p)
            break
    return "/".join(config_parts) if config_parts else rel_path


# 需要包含的 benchmark 及各自的主指标（按优先级）
BENCHMARKS: Dict[str, List[str]] = {
    "boolq": ["acc,none"],
    "ceval-valid": ["acc_norm,none", "acc,none"],
    "cmmlu": ["acc_norm,none", "acc,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "humaneval": ["pass@1,create_test", "pass@1"],
    "mmlu": ["acc_norm,none", "acc,none"],
    "wikitext": ["word_perplexity,none", "byte_perplexity,none"],
    "winogrande": ["acc,none"],
}


def get_benchmark_score(metrics: Dict[str, float]) -> Optional[float]:
    """根据 benchmark 名从 metrics 中取最佳可用指标值"""
    for bench, metric_list in BENCHMARKS.items():
        for m in metric_list:
            if m in metrics:
                return metrics[m]
    return None


def extract_benchmark_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """提取指定 benchmark 的指标（每个 benchmark 一个指标）"""
    out: Dict[str, float] = {}
    for bench in BENCHMARKS:
        tm = metrics.get(bench)
        if not tm:
            continue
        for m in BENCHMARKS[bench]:
            if m in tm:
                out[bench] = tm[m]
                break
    return out


def get_run_type(config: str) -> str:
    """从配置提取类型：nohot / hot / ablation"""
    parts = config.split("/")
    for p in parts:
        if p in ("nohot", "hot", "ablation"):
            return p
    return "other"


def iter_result_files(root: Path):
    """递归遍历目录，找到所有 results_*.json 文件"""
    root = root.resolve()
    for fp in root.rglob("results_*.json"):
        if not fp.is_file():
            continue
        try:
            rel = fp.relative_to(root)
        except ValueError:
            continue
        yield fp, str(rel)


def process_dir(
    root: Path,
    group_name: str,
    primary_metric: Optional[str] = None,
    dedupe: str = "latest",
) -> Tuple[Optional[float], Dict[str, Dict[str, Any]], Optional[str], Dict[str, float]]:
    """
    处理单个输出目录。
    返回: (baseline_mean, {config: {score, improvement, path, ...}}, primary_metric)
    """
    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for file_path, rel_path in iter_result_files(root):
        data = load_results(file_path)
        if not data:
            continue
        metrics = collect_metrics(data)
        if not metrics:
            continue

        pm = primary_metric or get_primary_metric(metrics)
        score = mean_score(metrics, pm)
        if score is None:
            continue

        config = "baseline" if is_baseline(rel_path) else extract_config(rel_path)
        records[config].append({
            "score": score,
            "path": str(file_path),
            "rel_path": rel_path,
            "metrics": metrics,
            "primary_metric": pm,
        })

    # 去重：同一 config 多个结果时，取 latest（按 path 字符串排序，一般时间戳在 path 里）
    merged: Dict[str, Dict[str, Any]] = {}
    for config, items in records.items():
        if dedupe == "latest":
            items = sorted(items, key=lambda x: x["path"], reverse=True)
        else:
            items = sorted(items, key=lambda x: x["score"], reverse=True)
        best = items[0]
        merged[config] = {
            "score": best["score"],
            "path": best["path"],
            "rel_path": best["rel_path"],
            "primary_metric": best["primary_metric"],
            "metrics": best["metrics"],  # 细分结果：各任务的指标
        }

    baseline = merged.get("baseline")
    baseline_mean = baseline["score"] if baseline else None
    primary_metric = baseline["primary_metric"] if baseline else pm

    # 计算相对 baseline 的提升
    for config, r in merged.items():
        if config == "baseline":
            r["improvement"] = 0.0
            r["improvement_pct"] = 0.0
        elif baseline_mean is not None:
            delta = r["score"] - baseline_mean
            r["improvement"] = delta
            r["improvement_pct"] = (delta / baseline_mean) * 100 if baseline_mean else 0
        else:
            r["improvement"] = None
            r["improvement_pct"] = None

    baseline_metrics: Dict[str, float] = {}
    if baseline:
        baseline_metrics = extract_benchmark_metrics(baseline.get("metrics", {}))
    return baseline_mean, merged, primary_metric, baseline_metrics


def main():
    parser = argparse.ArgumentParser(
        description="自动评估：对比 baseline，输出表现最好的配置"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            "output_nolora_final_nolora_general",
            "output_nolora_final_nolora_gen_qwen",
        ],
        help="要评估的输出目录列表",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="项目根目录（默认为脚本所在目录）",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="指定主要指标（如 acc_norm,none）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("eval_baseline_summary.csv"),
        help="汇总结果输出 CSV 路径（默认: eval_baseline_summary.csv）",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="禁用 CSV 输出",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    all_rows: List[Dict[str, Any]] = []  # 所有结果行，用于 CSV
    best_per_group: List[Dict[str, Any]] = []  # 每组最优（跨类型）

    print("=" * 80)
    print("自动评估：基于 baseline 对比各实验组")
    print("nohot / hot / ablation 分别单独排序")
    print("=" * 80)

    for dir_name in args.dirs:
        d = root / dir_name
        if not d.is_dir():
            print(f"\n[跳过] 目录不存在: {d}")
            continue

        group_name = dir_name
        baseline_mean, merged, pm, baseline_metrics = process_dir(d, group_name, args.metric)

        print(f"\n### {group_name}")
        print(f"    主要指标: {pm}")
        if baseline_mean is not None:
            print(f"    Baseline 均值: {baseline_mean * 100:.2f}%")
        else:
            print("    未找到 baseline，仅按绝对分数排序")

        # 先加入 baseline 行
        if baseline_mean is not None and baseline_metrics:
            base_row = {
                "group": group_name,
                "run_type": "baseline",
                "rank_in_type": 0,
                "config": "baseline",
                "score": baseline_mean,
                "baseline": baseline_mean,
                "improvement": 0.0,
                "improvement_pct": 0.0,
            }
            for bench, val in baseline_metrics.items():
                base_row[bench] = val
                base_row[f"{bench}_base"] = val
                base_row[f"{bench}_delta"] = 0.0
            all_rows.append(base_row)

        candidates = [(k, v) for k, v in merged.items() if k != "baseline" and v.get("improvement") is not None]
        if not candidates:
            print("    无其他实验组可与 baseline 对比")
            continue

        # 按类型分组：nohot / hot / ablation
        by_type: Dict[str, List[Tuple[str, Dict]]] = {"nohot": [], "hot": [], "ablation": []}
        for cfg, inf in candidates:
            t = get_run_type(cfg)
            if t in by_type:
                by_type[t].append((cfg, inf))

        best_in_group = None
        best_imp = float("-inf")

        for run_type in ("nohot", "hot", "ablation"):
            items = by_type[run_type]
            if not items:
                continue
            items.sort(key=lambda x: x[1]["improvement"], reverse=True)
            label = "纯 merge (ablation)" if run_type == "ablation" else run_type

            print(f"\n    --- {label} 组单独排序 ---")
            for rank, (cfg, inf) in enumerate(items, 1):
                imp = inf["improvement"] * 100
                pct = inf["improvement_pct"]
                row = {
                    "group": group_name,
                    "run_type": run_type,
                    "rank_in_type": rank,
                    "config": cfg,
                    "score": inf["score"],
                    "baseline": baseline_mean,
                    "improvement": inf["improvement"],
                    "improvement_pct": inf["improvement_pct"],
                }
                # 按 benchmark 汇总：每个 benchmark 一个指标
                bench_scores = extract_benchmark_metrics(inf.get("metrics") or {})
                for bench, score_val in bench_scores.items():
                    base_val = baseline_metrics.get(bench)
                    row[bench] = score_val
                    row[f"{bench}_base"] = base_val if base_val is not None else ""
                    row[f"{bench}_delta"] = (score_val - base_val) if base_val is not None else ""
                all_rows.append(row)

                print(f"        {rank}. {cfg}: {inf['score']*100:.2f}% (提升 {imp:+.2f}%, {pct:+.2f}%)")

                if inf["improvement"] > best_imp:
                    best_imp = inf["improvement"]
                    best_in_group = (cfg, inf)

        if best_in_group:
            cfg, inf = best_in_group
            best_per_group.append({
                "group": group_name,
                "config": cfg,
                "score": inf["score"],
                "baseline": baseline_mean,
                "improvement": inf["improvement"],
                "improvement_pct": inf["improvement_pct"],
            })

    # 跨组比较
    if len(best_per_group) >= 2:
        print("\n" + "=" * 80)
        print("跨组比较（相对 baseline 提升最大的配置）")
        print("=" * 80)
        overall_best = max(best_per_group, key=lambda x: x["improvement"] if x["improvement"] is not None else float("-inf"))
        print(f"\n表现最好的一组: {overall_best['group']}")
        print(f"    配置: {overall_best['config']}")
        print(f"    分数: {overall_best['score'] * 100:.2f}%")
        print(f"    Baseline: {overall_best['baseline'] * 100:.2f}%")
        print(f"    相对 baseline 提升: {overall_best['improvement'] * 100:+.2f}% ({overall_best['improvement_pct']:+.2f}%)")
    elif len(best_per_group) == 1:
        print("\n" + "=" * 80)
        b = best_per_group[0]
        print(f"唯一评估组 {b['group']} 中表现最好: {b['config']}")
        print(f"    提升: {b['improvement']*100:+.2f}% (相对 baseline)")

    # 默认 CSV 输出（含细分结果）
    if not args.no_csv and args.csv and all_rows:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        base_fields = ["group", "run_type", "rank_in_type", "config", "score", "baseline", "improvement", "improvement_pct"]
        bench_cols = sorted({k for r in all_rows for k in r if k not in base_fields})
        fieldnames = base_fields + bench_cols
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                row = {fn: (f"{r[fn]:.6f}" if isinstance(r.get(fn), float) else r.get(fn, "")) for fn in fieldnames}
                w.writerow(row)
        print(f"\n已写入: {args.csv}（含 {len(bench_cols)} 个 benchmark 的指标）")


if __name__ == "__main__":
    main()
