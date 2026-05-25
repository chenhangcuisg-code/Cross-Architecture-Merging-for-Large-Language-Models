#!/usr/bin/env python
"""
读取 hot/no-hot/untrained 的最新 evaluation 结果并按照 hot 排序，输出
带有对比数据的最终统计表格并写入 CSV（每个 alpha/lr 组合一行）。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


 

ALPHA_LR_RE = re.compile(r"alpha(?P<alpha>[-+0-9.eE]+)_lr(?P<lr>[-+0-9.eE]+)")

DEFAULT_METRIC_CANDIDATES = ("acc_norm,none", "acc,none")


DEFAULT_BASELINE_RESULTS = {
    "results": {
        "medmcqa": {
            "alias": "medmcqa",
            "acc,none": 0.37437245995696866,
            "acc_stderr,none": 0.0074837257065046145,
            "acc_norm,none": 0.37437245995696866,
            "acc_norm_stderr,none": 0.0074837257065046145,
        },
        "medqa_4options": {
            "alias": "medqa_4options",
            "acc,none": 0.3715632364493323,
            "acc_stderr,none": 0.013548886496306766,
            "acc_norm,none": 0.3715632364493323,
            "acc_norm_stderr,none": 0.013548886496306766,
        },
        "mmlu_anatomy": {
            "alias": "anatomy",
            "acc,none": 0.4666666666666667,
            "acc_stderr,none": 0.043097329010363554,
        },
        "mmlu_clinical_knowledge": {
            "alias": "clinical_knowledge",
            "acc,none": 0.45660377358490567,
            "acc_stderr,none": 0.030656748696739435,
        },
        "mmlu_college_biology": {
            "alias": "college_biology",
            "acc,none": 0.4722222222222222,
            "acc_stderr,none": 0.04174752578923185,
        },
        "mmlu_college_medicine": {
            "alias": "college_medicine",
            "acc,none": 0.36416184971098264,
            "acc_stderr,none": 0.03669072477416908,
        },
        "mmlu_medical_genetics": {
            "alias": "medical_genetics",
            "acc,none": 0.39,
            "acc_stderr,none": 0.04902071300001974,
        },
        "mmlu_professional_medicine": {
            "alias": "professional_medicine",
            "acc,none": 0.45588235294117646,
            "acc_stderr,none": 0.03025437257397669,
        },
    }
}


@dataclass(frozen=True)
class VariantDefinition:
    key: str
    dir_name: str
    label: str


DEFAULT_VARIANTS = (
    VariantDefinition("hot", "hot", "HOT"),
    VariantDefinition("nohot", "nohot", "No-hot"),
    VariantDefinition("untrained", "ablation_untrained_hot_fused", "Untrained hot fusions"),
)


def _iter_result_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith("results_") and filename.endswith(".json"):
                yield Path(dirpath) / filename


def _latest_result_file(variant_root: Path) -> Optional[Path]:
    latest: Optional[Path] = None
    for candidate in _iter_result_files(variant_root):
        if latest is None or candidate.name > latest.name:
            latest = candidate
    return latest


def _load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _collect_variant_details(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    details: Dict[str, Dict[str, float]] = {}
    for task, metadata in results.items():
        if not isinstance(metadata, dict):
            continue
        alias = metadata.get("alias", task)
        for metric, value in metadata.items():
            if isinstance(value, (int, float)):
                norm_metric = metric.lower()
                if "stderr" in norm_metric or "std" in norm_metric:
                    continue
                details.setdefault(alias, {})[metric] = float(value)
    return details


def _mean_metric_from_details(
    details: Dict[str, Dict[str, float]],
    metric_candidates: Sequence[str],
) -> Tuple[Optional[str], Optional[float]]:
    for metric in metric_candidates:
        values = [alias_metrics[metric] for alias_metrics in details.values() if metric in alias_metrics]
        if values:
            return metric, sum(values) / len(values)
    return None, None


def _metric_columns(variant_key: str, details: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[str, str, str]]:
    return {
        f"{variant_key}__{alias}__{metric}": (variant_key, alias, metric)
        for alias, alias_metrics in details.items()
        for metric in alias_metrics
    }


def collect_batch_rows(
    root_dir: Path,
    variant_defs: Sequence[VariantDefinition] = DEFAULT_VARIANTS,
    metric_candidates: Sequence[str] = DEFAULT_METRIC_CANDIDATES,
    baseline_details: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, str, str]], Dict[str, float]]:
    combos: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen_columns: Dict[str, Tuple[str, str, str]] = {}
    baseline_means: Dict[str, float] = {}

    if baseline_details:
        seen_columns.update(_metric_columns("baseline", baseline_details))
        baseline_means = _metric_candidate_means(baseline_details, metric_candidates)

    for alpha_dir in sorted(root_dir.iterdir()):
        if not alpha_dir.is_dir():
            continue
        match = ALPHA_LR_RE.fullmatch(alpha_dir.name)
        if not match:
            continue
        alpha_label = match.group("alpha")
        lr_label = match.group("lr")
        try:
            alpha_value = float(alpha_label)
        except ValueError:
            alpha_value = float("nan")
        try:
            lr_value = float(lr_label)
        except ValueError:
            lr_value = float("nan")

        key = (alpha_label, lr_label)
        record = combos.setdefault(
            key,
            {
                "alpha_label": alpha_label,
                "lr_label": lr_label,
                "alpha": alpha_value,
                "lr": lr_value,
                "variants": {},
            },
        )

        for variant in variant_defs:
            variant_root = alpha_dir / variant.dir_name
            if not variant_root.exists():
                continue
            result_path = _latest_result_file(variant_root)
            if result_path is None:
                continue
            metrics = _load_results(result_path).get("results", {})
            details = _collect_variant_details(metrics)
            if not details:
                continue
            mean_metric, mean_score = _mean_metric_from_details(details, metric_candidates)
            seen_columns.update(_metric_columns(variant.key, details))
            record["variants"][variant.key] = {
                "details": details,
                "mean_metric": mean_metric,
                "mean_score": mean_score,
                "result_path": result_path,
            }

    rows: List[Dict[str, Any]] = []
    for record in combos.values():
        hot_entry = record["variants"].get("hot")
        if hot_entry is None:
            continue
        variants = record["variants"]
        hot_mean = hot_entry.get("mean_score")
        sort_metric = hot_entry.get("mean_metric")
        baseline_mean = baseline_means.get(sort_metric) if sort_metric else None
        delta_baseline = (
            hot_mean - baseline_mean if hot_mean is not None and baseline_mean is not None else None
        )
        sort_score = delta_baseline if delta_baseline is not None else hot_mean
        rows.append(
            {
                "alpha_label": record["alpha_label"],
                "lr_label": record["lr_label"],
                "alpha": record["alpha"],
                "lr": record["lr"],
                "variants": variants,
                "sort_metric": sort_metric,
                "sort_score": sort_score,
                "hot_mean": hot_mean,
                "baseline_mean": baseline_mean,
                "delta_baseline": delta_baseline,
            }
        )

    rows.sort(
        key=lambda row: row["sort_score"] if row["sort_score"] is not None else float("-inf"),
        reverse=True,
    )
    return rows, seen_columns, baseline_means


def _mean_for_metric(details: Dict[str, Dict[str, float]], metric: str) -> Optional[float]:
    if not metric:
        return None
    values = [alias_metrics[metric] for alias_metrics in details.values() if metric in alias_metrics]
    if not values:
        return None
    return sum(values) / len(values)


def _metric_candidate_means(
    details: Dict[str, Dict[str, float]], metric_candidates: Sequence[str]
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for metric in metric_candidates:
        mean = _mean_for_metric(details, metric)
        if mean is not None:
            metrics[metric] = mean
    return metrics


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "   -"
    return f"{value * 100:6.2f}%"


def _format_delta(value: Optional[float]) -> str:
    if value is None:
        return "   -"
    return f"{value * 100:+6.2f}%"


def render_summary(rows: Sequence[Dict[str, Any]], metric: str) -> str:
    header = [
        "Alpha",
        "LR",
        f"Hot ({metric})",
        f"Baseline ({metric})",
        f"No-hot ({metric})",
        f"Untrained ({metric})",
        "Hot-Baseline",
        "Hot-No",
        "Hot-Untrained",
    ]
    template = "{:>7} {:>8} {:>14} {:>17} {:>15} {:>16} {:>12} {:>10} {:>13}"
    lines = [template.format(*header), template.format(*["-" * len(col) for col in header])]

    for row in rows:
        hot_metric = row.get("hot_mean")
        baseline_metric = row.get("baseline_mean")
        nohot_details = row["variants"].get("nohot", {}).get("details", {})
        untrained_details = row["variants"].get("untrained", {}).get("details", {})
        nohot_metric = _mean_for_metric(nohot_details, metric)
        untrained_metric = _mean_for_metric(untrained_details, metric)
        delta_baseline = row.get("delta_baseline")
        delta_nohot = hot_metric - nohot_metric if hot_metric is not None and nohot_metric is not None else None
        delta_untrained = (
            hot_metric - untrained_metric if hot_metric is not None and untrained_metric is not None else None
        )
        lines.append(
            template.format(
                row["alpha_label"],
                row["lr_label"],
                _format_percent(hot_metric),
                _format_percent(baseline_metric),
                _format_percent(nohot_metric),
                _format_percent(untrained_metric),
                _format_delta(delta_baseline),
                _format_delta(delta_nohot),
                _format_delta(delta_untrained),
            )
        )
    footer = [
        "所有值均为各 task 平均指标（hot 排序），nohot/untrained/baseline 为参考基线。",
        "CSV 文件同时包含每个 variant 的所有可用指标。",
    ]
    return "\n".join(lines + [""] + footer)


def write_csv(
    rows: Sequence[Dict[str, Any]],
    detail_columns: Dict[str, Tuple[str, str, str]],
    variant_defs: Sequence[VariantDefinition],
    csv_path: Path,
    baseline_details: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_means: Optional[Dict[str, float]] = None,
    summary_metric: Optional[str] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    variant_order = {variant.key: idx for idx, variant in enumerate(variant_defs)}
    def _column_sort_key(column: str) -> Tuple[int, str]:
        variant_key = column.split("__", 1)[0]
        return (variant_order.get(variant_key, len(variant_defs)), column)

    sorted_columns = sorted(detail_columns.items(), key=lambda item: _column_sort_key(item[0]))
    fieldnames = [
        "alpha_label",
        "lr_label",
        "sort_metric",
        "sort_score",
        "hot_mean",
        "baseline_mean",
        "delta_baseline",
    ] + [col for col, _ in sorted_columns]

    ordered_rows: List[Dict[str, Any]] = []
    if baseline_details:
        baseline_mean = baseline_means.get(summary_metric) if summary_metric and baseline_means else None
        ordered_rows.append(
            {
                "alpha_label": "baseline",
                "lr_label": "",
                "sort_metric": summary_metric or "",
                "sort_score": baseline_mean,
                "hot_mean": baseline_mean,
                "baseline_mean": baseline_mean,
                "delta_baseline": 0.0 if baseline_mean is not None else None,
                "variants": {"baseline": {"details": baseline_details}},
            }
        )
    ordered_rows.extend(rows)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            payload: Dict[str, Any] = {
                "alpha_label": row["alpha_label"],
                "lr_label": row["lr_label"],
                "sort_metric": row["sort_metric"] or "",
                "sort_score": "" if row["sort_score"] is None else f"{row['sort_score']:.6f}",
                "hot_mean": "" if row.get("hot_mean") is None else f"{row['hot_mean']:.6f}",
                "baseline_mean": "" if row.get("baseline_mean") is None else f"{row['baseline_mean']:.6f}",
                "delta_baseline": "" if row.get("delta_baseline") is None else f"{row['delta_baseline']:.6f}",
            }
            for column, (variant_key, alias, metric) in sorted_columns:
                value = (
                    row["variants"]
                    .get(variant_key, {})
                    .get("details", {})
                    .get(alias, {})
                    .get(metric)
                )
                payload[column] = "" if value is None else f"{value:.6f}"
            writer.writerow(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量读取 hot/nohot/untrained latest evaluation，排序并写入 CSV")
    parser.add_argument("--root", type=Path, default='/data/chenhang/codes/lm-evaluation-harness/output/llama3-eval/llama_medical_llama3_train_alpha_lr_search', help="alpha/lr 层级的 evaluation 根目录")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(__file__).resolve().parent / "hot_eval_batch.csv",
        help="输出 CSV 文件",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=list(DEFAULT_METRIC_CANDIDATES),
        help="排序优先 metric（默认按 acc_norm,none 再 acc,none）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="只保留排序前 k 条",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="仅生成 CSV，不打印摘要表",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="baseline JSON（默认为脚本内置的 baseline）",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("从 %s 收集 hot/nohot/untrained 评估", args.root)
    if args.baseline_file:
        logging.info("使用 baseline 文件：%s", args.baseline_file)
        baseline_source = _load_results(args.baseline_file).get("results", {})
    else:
        baseline_source = DEFAULT_BASELINE_RESULTS["results"]
    baseline_details = _collect_variant_details(baseline_source)

    rows, detail_columns, baseline_means = collect_batch_rows(
        args.root, metric_candidates=args.metric, baseline_details=baseline_details
    )
    if not rows:
        logging.warning("未找到任何 hot result，退出")
        return
    if args.top_k is not None:
        rows = rows[: args.top_k]

    summary_metric = args.metric[0] if args.metric else ""
    if not summary_metric and rows:
        summary_metric = rows[0]["sort_metric"] or ""

    if not args.quiet:
        print(render_summary(rows, summary_metric))

    write_csv(
        rows,
        detail_columns,
        DEFAULT_VARIANTS,
        args.csv_path,
        baseline_details=baseline_details,
        baseline_means=baseline_means,
        summary_metric=summary_metric,
    )
    logging.info("写入 CSV：%s", args.csv_path)


if __name__ == "__main__":
    main()

