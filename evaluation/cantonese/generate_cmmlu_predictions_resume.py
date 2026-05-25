#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMMLU Cantonese Prediction Generation Script (Resume Version)
只处理未完成的科目
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np

# -------------------------
#  固定随机种子（保证评测可复现）
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
set_seed(42)

def load_model_and_tokenizer(model_path: str, device: str = "cuda:0", dtype: str = "float16"):
    """加载模型和分词器"""
    print(f"加载模型: {model_path}")

    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map={"": device} if torch.cuda.is_available() else "cpu"
    )

    if torch.cuda.is_available():
        model = model.to(device)

    return model, tokenizer

def generate_answer(model, tokenizer, prompt: str, device: str = "cuda:0", max_new_tokens: int = 256) -> str:
    """确定性生成答案"""

    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # -------------------------
    #  Deterministic generation
    # -------------------------
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # 禁用随机采样 → greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,             # 避免部分模型返回 legacy cache 列表触发 DynamicCache 报错
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if input_text in generated_text:
        prediction = generated_text[len(input_text):].strip()
    else:
        prediction = generated_text.strip()

    return prediction

def load_cmmlu_data(data_dir: str) -> dict:
    """加载 CMMLU 数据"""
    data = {}

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    for file_path in sorted(list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json"))):
        subject = file_path.stem
        data[subject] = []

        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            data[subject].append(item)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                    if isinstance(items, list):
                        data[subject] = items
        except Exception as e:
            print(f"警告: 加载文件 {file_path} 时出错: {e}")
            continue

    if not data:
        raise ValueError(f"在 {data_dir} 中未找到任何数据文件")

    print(f"成功加载 {len(data)} 个 subject")
    return data

def generate_predictions_resume(
    model_path: str,
    data_dir: str,
    output_dir: str,
    completed_subjects: list,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_new_tokens: int = 256,
):
    """只处理未完成的科目"""
    print(f"加载数据从: {data_dir}")
    data = load_cmmlu_data(data_dir)

    print(f"加载模型: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, device, dtype)

    os.makedirs(output_dir, exist_ok=True)

    # 过滤出未完成的科目
    remaining_subjects = {k: v for k, v in data.items() if k not in completed_subjects}

    print(f"总科目数: {len(data)}, 已完成: {len(completed_subjects)}, 剩余: {len(remaining_subjects)}")
    print(f"剩余科目: {list(remaining_subjects.keys())}")

    for subject, items in remaining_subjects.items():
        print(f"\n处理 subject: {subject} (共 {len(items)} 条)")

        predictions = []

        for idx, item in enumerate(tqdm(items, desc=f"{subject}")):
            question = item.get('question', '')
            choices = item.get('choices', [])

            prompt = f"以下是关于{subject}的多项选择题，请只回答选项字母。\n\n{question}\n\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\n答案："

            try:
                prediction = generate_answer(model, tokenizer, prompt, device, max_new_tokens)
                predictions.append({
                    'question': question,
                    'choices': choices,
                    'prediction': prediction,
                    'answer': item.get('answer', '')
                })
            except Exception as e:
                print(f"处理问题时出错: {e}")
                predictions.append({
                    'question': question,
                    'choices': choices,
                    'prediction': 'ERROR',
                    'answer': item.get('answer', '')
                })

        # 保存预测
        output_file = os.path.join(output_dir, f"cmmlu-yue-{subject}-0shot.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        print(f"保存预测到: {output_file} (共 {len(predictions)} 条)")

    print(f"\n所有剩余预测已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='生成 CMMLU Cantonese 预测（断点续传版）')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--completed_subjects", type=str, nargs='*', default=[],
                       help="已完成的科目列表")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    generate_predictions_resume(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        completed_subjects=args.completed_subjects,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()





