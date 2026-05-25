#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CMMLU Cantonese Prediction Generation Script

This script generates predictions for CMMLU (Chinese Multi-task Language
Understanding) benchmark in Cantonese using a trained model.

This script is part of the evaluation utilities for Cross-Architecture Merging.
It works with the Yue-Benchmark repository: https://github.com/jiangjyjy/Yue-Benchmark

The generated predictions can be evaluated using evaluate_cmmlu_yue.py.

Usage:
    python generate_cmmlu_predictions.py \\
        --model_path <model_path> \\
        --data_dir <yue_benchmark_data_dir> \\
        --output_dir <output_dir> \\
        --num_shots 0 \\
        --device cuda:0
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path: str, device: str = "cuda:0", dtype: str = "float16"):
    """Load model and tokenizer"""
    print(f"Loading model: {model_path}")
    
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
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def format_prompt(question: str, choices: List[str], num_shots: int = 0, few_shot_examples: List[Dict] = None) -> str:
    """Format the prompt"""
    prompt = ""
    
    # Add few-shot examples
    if num_shots > 0 and few_shot_examples:
        for example in few_shot_examples[:num_shots]:
            ex_question = example.get('question', '')
            ex_choices = example.get('choices', [])
            # If choices is a list, use it directly; otherwise construct from A, B, C, D fields
            if not ex_choices:
                ex_choices = []
                for opt in ['A', 'B', 'C', 'D']:
                    if opt in example:
                        ex_choices.append(example[opt])
            ex_answer = example.get('answer', '')
            
            prompt += f"{ex_question}\n"
            for i, choice in enumerate(ex_choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += f"Answer: {ex_answer}\n\n"
    
    # Add current question
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt


def generate_answer(model, tokenizer, prompt: str, device: str = "cuda:0", max_new_tokens: int = 512) -> str:
    """Generate answer"""
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract generated part (remove input)
    if input_text in generated_text:
        prediction = generated_text[len(input_text):].strip()
    else:
        prediction = generated_text.strip()
    
    return prediction


def load_cmmlu_data(data_dir: str) -> Dict[str, List[Dict]]:
    """Load CMMLU data
    
    Supports two formats:
    1. JSON file, each file contains an array with elements: question, A, B, C, D, answer, no
    2. JSONL file, one JSON object per line
    """
    data = {}
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Iterate through all subject files (supports .jsonl and .json)
    for file_path in sorted(list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json"))):
        subject = file_path.stem
        data[subject] = []
        
        try:
            if file_path.suffix == '.jsonl':
                # JSONL format: one JSON object per line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            data[subject].append(item)
            else:  # .json
                # JSON format: file contains an array
                with open(file_path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                    if isinstance(items, list):
                        data[subject] = items
                    elif isinstance(items, dict):
                        # If it's a dict, try to extract the list
                        for key, value in items.items():
                            if isinstance(value, list):
                                data[subject].extend(value)
                            else:
                                data[subject].append(value)
        except Exception as e:
            print(f"Warning: Error loading file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not data:
        raise ValueError(f"No data files found in {data_dir}")
    
    print(f"Successfully loaded {len(data)} subjects with {sum(len(items) for items in data.values())} items in total")
    return data


def generate_predictions(
    model_path: str,
    data_dir: str,
    output_dir: str,
    num_shots: int = 0,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_new_tokens: int = 512,
    batch_size: int = 1,
):
    """Generate all predictions"""
    print(f"Loading data from: {data_dir}")
    data = load_cmmlu_data(data_dir)
    
    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, device, dtype)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions for each subject
    oom_skip_count = 0
    for subject, items in data.items():
        print(f"\nProcessing subject: {subject} ({len(items)} items)")
        
        predictions = {}
        
        # Prepare few-shot examples (select from first few questions of current subject)
        few_shot_examples = None
        if num_shots > 0:
            few_shot_examples = []
            for ex_item in items[:min(num_shots * 2, len(items))]:  # Get more as candidates
                ex_question = ex_item.get('question', '')
                if 'choices' in ex_item:
                    ex_choices = ex_item['choices']
                else:
                    ex_choices = []
                    for opt in ['A', 'B', 'C', 'D']:
                        if opt in ex_item:
                            ex_choices.append(ex_item[opt])
                ex_answer = ex_item.get('answer', '')
                if ex_question and ex_choices and ex_answer:
                    few_shot_examples.append({
                        'question': ex_question,
                        'choices': ex_choices,
                        'answer': ex_answer
                    })
            few_shot_examples = few_shot_examples[:num_shots]  # Only take needed number
        
        for idx, item in enumerate(tqdm(items, desc=f"{subject}")):
            question = item.get('question', '')
            
            # Handle options: supports two formats
            # Format1: choices list
            # Format2: A, B, C, D fields
            if 'choices' in item:
                choices = item['choices']
            else:
                # Construct choices list from A, B, C, D fields
                choices = []
                for option in ['A', 'B', 'C', 'D']:
                    if option in item:
                        choices.append(item[option])
            
            gold = item.get('answer', '')
            
            # For few-shot, exclude current question itself
            current_few_shot = None
            if num_shots > 0 and few_shot_examples:
                # Exclude current question, select from other examples
                current_few_shot = [ex for ex in few_shot_examples if ex['question'] != question]
                if len(current_few_shot) < num_shots:
                    # If not enough after exclusion, select from all examples (may include current question, but usually won't)
                    current_few_shot = few_shot_examples[:num_shots]
            
            # Format prompt
            prompt = format_prompt(question, choices, num_shots=num_shots, few_shot_examples=current_few_shot)
            item_id = item.get('no', item.get('id', item.get('question_id', str(idx))))
            
            # Generate prediction (skip and count on OOM)
            try:
                prediction = generate_answer(model, tokenizer, prompt, device, max_new_tokens)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_skip_count += 1
                    continue  # skip current sample, leave output empty
                raise
            
            # Save results
            predictions[str(item_id)] = {
                'prediction': prediction,
                'gold': gold,
                'origin_prompt': prompt,
                'question': question,
                'choices': choices,
            }
        
        # Save prediction file (format: cmmlu-yue-{subject}-{num_shots}shot.json)
        output_file = os.path.join(output_dir, f"cmmlu-yue-{subject}-{num_shots}shot.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"Saved predictions to: {output_file} ({len(predictions)} items)")
    
    if oom_skip_count > 0:
        print(f"\n[OOM] Skipped {oom_skip_count} samples, output left empty")
    print(f"\nAll predictions saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate CMMLU Cantonese predictions')
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_dir", type=str, required=True, help="CMMLU data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of few-shot examples (0 or 5)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    args = parser.parse_args()
    
    generate_predictions(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_shots=args.num_shots,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
