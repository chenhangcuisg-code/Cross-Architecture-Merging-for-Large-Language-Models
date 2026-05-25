#!/usr/bin/env python
"""
通用的 FinanceQA / 变换器 LLM 评估工具。
基于 Hugging Face Transformers，加载 AfterQuery/FinanceQA 数据集并生成回答，统计
精确匹配与 token-level F1 等常用指标。
"""
import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from string import Template

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


PROMPT_TEMPLATE = """Context:
${context}

Question:
${question}

Answer:"""


def parse_args():
    parser = argparse.ArgumentParser(description="评估 Hugging Face LLM 在 FinanceQA 上的表现。")
    parser.add_argument("--model", required=True, help="Hugging Face 模型 ID 或本地目录")
    parser.add_argument("--dataset", default="AfterQuery/FinanceQA", help="默认使用 FinanceQA 数据集")
    parser.add_argument("--split", default="test", help="加载数据集中要评估的 split")
    parser.add_argument("--max_samples", type=int, default=200, help="最多评估的样本数（None 为全部）")
    parser.add_argument("--max_context_chars", type=int, default=4096, help="每条样本上下文裁剪字符数")
    parser.add_argument("--prompt_template", default=PROMPT_TEMPLATE, help="评估请求的提示模板")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", help="是否启用采样生成")
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--save_predictions", type=Path, help="生成回答后保存 json")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_answer(answer: str) -> str:
    """降低大小写并移除标点与多余空白。"""
    if not answer:
        return ""
    answer = answer.strip().lower()
    answer = re.sub(rf"[{re.escape(string.punctuation)}]", " ", answer)
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


# ---- FIXED ----
def build_prompt(template: str, context, question: str, max_context_chars: int) -> str:
    # 保证 context 一定是字符串
    if context is None:
        context = ""
    else:
        context = str(context)

    # 裁剪 context
    if max_context_chars and len(context) > max_context_chars:
        context = context[:max_context_chars]

    template_obj = Template(template)
    return template_obj.safe_substitute(context=context, question=question).strip()
# ----------------


def evaluate_examples(generator, examples, args):
    predictions = []
    for idx, example in enumerate(tqdm(examples, desc="Evaluating", unit="sample")):
        # ---- FIXED ----
        context_text = example.get("context")
        if context_text is None:
            context_text = ""
        else:
            context_text = str(context_text)
        # ----------------

        question = example.get("question", "").strip()
        reference = example.get("answer", "").strip()

        prompt = build_prompt(args.prompt_template, context_text, question, args.max_context_chars)

        try:
            generation = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,
                return_full_text=False,
            )
        except Exception:
            logging.exception("第 %s 条样本生成失败", idx)
            generated_text = ""
        else:
            generated_text = generation[0]["generated_text"].strip()

        predictions.append(
            {
                "index": idx,
                "question": question,
                "reference": reference,
                "prediction": generated_text,
                "context_length": len(context_text),
            }
        )
    return predictions


def aggregate_metrics(predictions):
    exact_count = 0
    f1_sum = 0.0
    total = len(predictions)
    for row in predictions:
        ref_norm = normalize_answer(row["reference"])
        pred_norm = normalize_answer(row["prediction"])
        if ref_norm and pred_norm and ref_norm == pred_norm:
            exact_count += 1
        f1_sum += compute_token_f1(pred_norm, ref_norm)
    return {
        "exact_match": (exact_count / total) * 100 if total else 0.0,
        "token_f1": (f1_sum / total) * 100 if total else 0.0,
        "examples": total,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    torch.manual_seed(args.seed)
    device_id = 0 if torch.cuda.is_available() and not args.no_cuda else -1

    logging.info("加载 %s split=%s，样本上限 %s", args.dataset, args.split, args.max_samples)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
    logging.info("实际评估样本 %s", len(dataset))

    logging.info("加载 LLM：%s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        generation_config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
        ),
        return_full_text=False,
    )

    predictions = evaluate_examples(generator, list(dataset), args)
    metrics = aggregate_metrics(predictions)
    logging.info(
        "评估结果：exact_match=%.2f%% token_f1=%.2f%%（共 %d 条样本）",
        metrics["exact_match"],
        metrics["token_f1"],
        metrics["examples"],
    )

    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as fp:
            json.dump(predictions, fp, ensure_ascii=False, indent=2)
        logging.info("预测结果已保存到 %s", args.save_predictions)


if __name__ == "__main__":
    main()
