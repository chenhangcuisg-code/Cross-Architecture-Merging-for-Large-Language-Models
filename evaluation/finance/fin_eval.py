#!/usr/bin/env python
"""
自动计算 output_fin 文件夹下所有 pred.json 文件的准确率指标并输出 CSV。
"""

import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def extract_answer_from_prediction(prediction: str) -> str:
    """
    从 prediction 文本中提取答案（通常是第一个出现的数字/金额）。
    优先提取第一行，如果第一行包含数字则使用第一行，否则查找第一个包含数字的片段。
    """
    if not prediction:
        return ""
    
    # 先移除首尾空白
    prediction = prediction.strip()
    
    # 尝试提取第一行（通常答案在第一行）
    first_line = prediction.split("\n")[0].strip()
    
    # 如果第一行包含数字或货币符号，使用第一行
    if re.search(r"[\d$]", first_line):
        return first_line
    
    # 否则，查找第一个包含数字/货币的片段（最多前200个字符）
    search_text = prediction[:200]
    # 匹配模式：$数字 或 数字（可能包含逗号、小数点、百分号）
    match = re.search(r"\$?[\d,]+(?:\.\d+)?%?", search_text)
    if match:
        # 提取匹配的片段及其前后少量上下文（最多50个字符）
        start = max(0, match.start() - 20)
        end = min(len(prediction), match.end() + 20)
        return prediction[start:end].strip()
    
    # 如果都没找到，返回第一行（即使没有数字）
    return first_line


def normalize_answer(answer: str) -> str:
    """
    标准化答案：小写、移除标点与多余空白。
    特别注意：移除括号及其内容，只匹配括号前的数字部分。
    例如 "$32,095 (in millions)" -> "$32,095"
    """
    if not answer:
        return ""
    answer = answer.strip()
    # 移除括号及其内容，例如 "(in millions)" -> ""
    answer = re.sub(r"\([^)]*\)", "", answer)
    # 转换为小写
    answer = answer.lower()
    # 移除标点（但保留数字和基本字符）
    answer = re.sub(rf"[{re.escape(string.punctuation)}]", " ", answer)
    # 移除多余空白
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()


def compute_token_f1(prediction: str, reference: str) -> float:
    """使用词条交集计算 token-level F1（若任一为空返回 0）。"""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_accuracy(pred_file: Path, debug: bool = False) -> Dict[str, float]:
    """
    计算 pred.json 文件的准确率指标。
    
    Args:
        pred_file: pred.json 文件路径
        debug: 是否输出调试信息（显示前几个匹配示例）
    
    Returns:
        Dict with keys: 'exact_match', 'token_f1', 'total_samples'
    """
    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    if not predictions:
        return {"exact_match": 0.0, "token_f1": 0.0, "total_samples": 0}
    
    exact_count = 0
    f1_sum = 0.0
    total = len(predictions)
    debug_count = 0
    max_debug = 5  # 最多显示5个调试示例
    
    for row in predictions:
        reference = row.get("reference", "").strip()
        prediction = row.get("prediction", "").strip()
        
        # 从 prediction 中提取答案（通常是第一个数字/金额）
        pred_answer = extract_answer_from_prediction(prediction)
        
        ref_norm = normalize_answer(reference)
        pred_norm = normalize_answer(pred_answer)
        
        # 精确匹配
        is_match = ref_norm and pred_norm and ref_norm == pred_norm
        if is_match:
            exact_count += 1
        
        # Token F1
        f1 = compute_token_f1(pred_norm, ref_norm)
        f1_sum += f1
        
        # 调试输出
        if debug and debug_count < max_debug:
            print(f"  [示例 {debug_count + 1}]")
            print(f"    Reference: {reference}")
            print(f"    Prediction (原始): {prediction[:100]}...")
            print(f"    Prediction (提取): {pred_answer}")
            print(f"    Reference (标准化): {ref_norm}")
            print(f"    Prediction (标准化): {pred_norm}")
            print(f"    匹配: {'✓' if is_match else '✗'}, F1: {f1:.4f}")
            debug_count += 1
    
    return {
        "exact_match": (exact_count / total) * 100 if total else 0.0,
        "token_f1": (f1_sum / total) * 100 if total else 0.0,
        "total_samples": total,
    }


def find_all_pred_files(root_dir: Path) -> List[Dict[str, str]]:
    """
    递归查找 root_dir 下所有 pred.json 文件。
    
    Returns:
        List of dicts with keys: 'path', 'relative_path', 'folder_name', 'variant'
    """
    pred_files = []
    
    for pred_file in root_dir.rglob("pred.json"):
        # 获取相对于 root_dir 的路径
        relative_path = pred_file.relative_to(root_dir)
        
        # 提取文件夹结构信息
        parts = relative_path.parts
        if len(parts) >= 2:
            # 假设结构为: alpha0.01_lr1e-6/variant/pred.json
            folder_name = parts[0]  # e.g., "alpha0.01_lr1e-6"
            variant = parts[1] if len(parts) > 1 else ""  # e.g., "hot", "ablation"
        else:
            folder_name = ""
            variant = ""
        
        pred_files.append({
            "path": str(pred_file),
            "relative_path": str(relative_path),
            "folder_name": folder_name,
            "variant": variant,
        })
    
    return pred_files


def fin_eval(root_dir: Path, output_csv: Optional[Path] = None, debug: bool = False) -> List[Dict]:
    """
    自动计算所有 pred.json 文件的准确率并返回结果列表。
    
    Args:
        root_dir: output_fin 文件夹路径
        output_csv: 输出 CSV 文件路径，如果为 None 则使用默认路径
    
    Returns:
        List of dicts containing evaluation results
    """
    if output_csv is None:
        output_csv = root_dir / "fin_eval_results.csv"
    
    # 查找所有 pred.json 文件
    pred_files = find_all_pred_files(root_dir)
    
    if not pred_files:
        print(f"警告: 在 {root_dir} 下未找到任何 pred.json 文件")
        return []
    
    # 计算每个文件的准确率
    results = []
    for pred_info in pred_files:
        pred_file = Path(pred_info["path"])
        try:
            if debug and len(results) == 0:  # 只对第一个文件进行详细调试
                print(f"\n[调试模式] 处理文件: {pred_info['relative_path']}")
                metrics = calculate_accuracy(pred_file, debug=True)
            else:
                metrics = calculate_accuracy(pred_file, debug=False)
            results.append({
                "folder_name": pred_info["folder_name"],
                "variant": pred_info["variant"],
                "relative_path": pred_info["relative_path"],
                "exact_match": metrics["exact_match"],
                "token_f1": metrics["token_f1"],
                "total_samples": metrics["total_samples"],
            })
            print(f"已处理: {pred_info['relative_path']} - EM: {metrics['exact_match']:.2f}%, F1: {metrics['token_f1']:.2f}%")
        except Exception as e:
            print(f"错误: 处理 {pred_info['relative_path']} 时出错: {e}")
            continue
    
    # 按 folder_name 和 variant 排序
    results.sort(key=lambda x: (x["folder_name"], x["variant"]))
    
    # 写入 CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["folder_name", "variant", "relative_path", "exact_match", "token_f1", "total_samples"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "folder_name": row["folder_name"],
                "variant": row["variant"],
                "relative_path": row["relative_path"],
                "exact_match": f"{row['exact_match']:.4f}",
                "token_f1": f"{row['token_f1']:.4f}",
                "total_samples": row["total_samples"],
            })
    
    print(f"\n结果已保存到: {output_csv}")
    print(f"共处理 {len(results)} 个 pred.json 文件")
    
    return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自动计算 output_fin 文件夹下所有 pred.json 的准确率")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).parent / "output_fin",
        help="output_fin 文件夹路径（默认: evaluation/output_fin）",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="输出 CSV 文件路径（默认: root_dir/fin_eval_results.csv）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，显示前几个匹配示例",
    )
    
    args = parser.parse_args()
    
    if not args.root_dir.exists():
        print(f"错误: 路径不存在: {args.root_dir}")
        return
    
    fin_eval(args.root_dir, args.output_csv, debug=args.debug)


if __name__ == "__main__":
    main()

