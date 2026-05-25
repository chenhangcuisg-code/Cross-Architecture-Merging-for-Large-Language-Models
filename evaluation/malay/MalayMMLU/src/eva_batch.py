import argparse
import pandas as pd
import os
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from numpy import argmax
import torch
from utils import predict_classification_causal as predict_classification
from utils import predict_classification_causal_by_letter as predict_classification_by_letter

device = "cuda"

# usage: python evaluate_batch.py --by_letter --shot 0 --task=MalayMMLU --base_model=google/gemma-2b-it --output_folder=$HOME/MalayMMLU/output/ --token $TOKEN --data_folder=data/

def find_all_json_files(data_folder):
    """
    递归查找文件夹下所有 JSON 文件（包括子文件夹）
    返回文件路径列表，按路径排序
    """
    json_files = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        raise ValueError(f"数据文件夹不存在: {data_folder}")
    
    # 递归查找所有 .json 文件
    for json_file in data_path.rglob("*.json"):
        json_files.append(str(json_file))
    
    # 按路径排序
    json_files.sort()
    
    return json_files

def prepare_data(playground, model_name, tokenizer, task, data_file):
    """
    从指定的 JSON 文件准备数据
    """
    if task == "MalayMMLU":
        inputs = []
        outputs = []
        outputs_options = []
        key2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        shot = 0
        
        data = pd.read_json(data_file)
        
        if playground:
            data = data.iloc[:500]
        
        for idx, row in data.iterrows():
            ques = data.iloc[idx]['prompt']
            
            if "llama" in model_name.lower():
                p = f"Berikut adalah soalan aneka pilihan tentang {row['subject']}. Sila berikan jawapan sahaja.\n\n" + ques
                chat = [{"role": "user", "content": p}]
                chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + "\nJawapan:"
            else:
                p = f"Berikut adalah soalan aneka pilihan tentang {row['subject']}. Sila berikan jawapan sahaja.\n\n" + ques + "\nJawapan:"
                chat = [{"role": "user", "content": p}]
                chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            inputs.append(chat)
            idx_label = key2id[row['key']]
            outputs.append(idx_label)
            outputs_options.append(row['options'])
        
        return inputs, outputs, outputs_options
    else:
        raise ValueError(f"不支持的任务类型: {task}")

def prepare_data_few_shot(shot, model_name, tokenizer, task, data_file):
    """
    从指定的 JSON 文件准备 few-shot 数据
    """
    if task == "MalayMMLU":
        inputs = []
        outputs = []
        outputs_options = []
        key2id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        
        data = pd.read_json(data_file)
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            if "llama" in model_name.lower():
                p = data.iloc[i][f'full_question_{shot}shot_llama']
                chat = [{"role": "user", "content": p}]
                chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) + "Jawapan:"
            else:
                p = data.iloc[i][f'full_question_{shot}shot']
                chat = [{"role": "user", "content": p}]
                chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            inputs.append(chat)
            idx_label = key2id[row['key']]
            outputs.append(idx_label)
            outputs_options.append(row['options'])
        
        return inputs, outputs, outputs_options
    else:
        raise ValueError(f"不支持的任务类型: {task}")

def process_single_file(data_file, model, tokenizer, args, model_name):
    """
    处理单个 JSON 文件，返回结果 DataFrame
    """
    print(f"\n处理文件: {data_file}")
    
    # 准备数据
    if args.shot == 0:
        inputs, golds, outputs_options = prepare_data(
            args.playground, model_name, tokenizer, args.task, data_file
        )
    else:
        inputs, golds, outputs_options = prepare_data_few_shot(
            args.shot, model_name, tokenizer, args.task, data_file
        )
    
    if len(inputs) == 0:
        print(f"警告: 文件 {data_file} 没有数据")
        return None
    
    # 预测
    preds = []
    probs = []
    
    for idx in tqdm(range(len(inputs)), desc=f"处理 {Path(data_file).name}"):
        if not args.by_letter:  # full answer probability
            out = predict_classification(model, tokenizer, inputs[idx], outputs_options[idx], device)
            prob = [o.cpu().detach().item() for o in out]
            pred = argmax(prob)
            preds.append(pred)
            probs.append(prob)
        else:  # first token probability
            conf, pred = predict_classification_by_letter(model, tokenizer, inputs[idx], outputs_options[idx], device)
            preds.append(pred)
    
    # 创建结果 DataFrame
    output_df = pd.DataFrame()
    output_df['data_file'] = [data_file] * len(inputs)  # 记录来源文件
    output_df['input'] = inputs
    output_df['golds'] = golds
    output_df['options'] = outputs_options
    output_df['preds'] = preds
    
    if not args.by_letter and len(probs) > 0:
        output_df['probs'] = probs
    
    # 计算准确率
    accuracy = (output_df['golds'] == output_df['preds']).mean()
    print(f"文件 {Path(data_file).name} 准确率: {accuracy:.4f}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by_letter",
                        action='store_true',
                        help="Use this flag to calculate first token accuracy. For calculating full answer accuracy, do not include this flag in args")
    parser.add_argument("--base_model",
                         type=str,
                         help="Path to pretrained model",
                         required=True)
    parser.add_argument("--output_folder",
                        type=str,
                        default="output",
                        required=True,
                        help="Folder where the output will be saved")
    parser.add_argument("--playground",
                        type=bool,
                        default=False,
                        help="Set this to True to enable playground mode (default: False).")
    parser.add_argument("--task",
                        type=str,
                        default="MalayMMLU",
                        help="Specify the task to be executed (default: 'MalayMMLU').")
    parser.add_argument("--shot",
                        type=int,
                        default=0,
                        help="Provide the number of shots: 0,1,2 or 3")
    parser.add_argument("--token",
                        type=str,
                        help='Specify the HuggingFace token')
    parser.add_argument("--data_folder",
                        type=str,
                        default="data",
                        help="Folder containing JSON data files (will recursively search subfolders)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 查找所有 JSON 文件
    print(f"正在搜索文件夹: {args.data_folder}")
    json_files = find_all_json_files(args.data_folder)
    
    if len(json_files) == 0:
        print(f"错误: 在 {args.data_folder} 及其子文件夹中未找到任何 JSON 文件")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件:")
    for f in json_files:
        print(f"  - {f}")
    
    # 加载模型和 tokenizer
    tokenizer_class = AutoTokenizer
    model_class = LlamaForCausalLM if ('llama' in args.base_model and ("Llama-3" not in args.base_model and "Llama-2" not in args.base_model)) else AutoModelForCausalLM
    
    print(f"\n加载模型: {args.base_model}")
    tokenizer = tokenizer_class.from_pretrained(args.base_model, token=args.token, trust_remote_code=True)
    model = model_class.from_pretrained(args.base_model, token=args.token, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda")
    model.eval()
    
    print(f"任务: {args.task}")
    print(f"Shot: {args.shot}")
    print(f"By letter: {args.by_letter}")
    
    # 批量处理所有文件
    all_results = []
    
    for data_file in json_files:
        try:
            result_df = process_single_file(data_file, model, tokenizer, args, args.base_model)
            if result_df is not None:
                all_results.append(result_df)
        except Exception as e:
            print(f"处理文件 {data_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) == 0:
        print("错误: 没有成功处理任何文件")
        return
    
    # 合并所有结果
    print("\n合并所有结果...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 排序：先按文件路径，再按索引
    combined_df = combined_df.sort_values(by=['data_file', combined_df.index])
    
    # 保存结果
    model_name_short = args.base_model.split("/")[-1]
    save_file = f'{args.output_folder}/{args.task}_result_{model_name_short}_{args.by_letter}_{args.shot}shot_batch.csv'
    
    combined_df.to_csv(save_file, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到: {save_file}")
    print(f"总共处理了 {len(combined_df)} 条数据，来自 {len(all_results)} 个文件")
    
    # 计算总体准确率
    overall_accuracy = (combined_df['golds'] == combined_df['preds']).mean()
    print(f"总体准确率: {overall_accuracy:.4f}")
    
    # 按文件统计准确率
    print("\n各文件准确率统计:")
    file_stats = combined_df.groupby('data_file').apply(
        lambda x: (x['golds'] == x['preds']).mean()
    ).reset_index(name='accuracy')
    file_stats = file_stats.sort_values('accuracy', ascending=False)
    
    for _, row in file_stats.iterrows():
        file_name = Path(row['data_file']).name
        print(f"  {file_name}: {row['accuracy']:.4f}")
    
    # 保存统计信息
    stats_file = f'{args.output_folder}/{args.task}_stats_{model_name_short}_{args.by_letter}_{args.shot}shot_batch.csv'
    file_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"\n统计信息已保存到: {stats_file}")

if __name__ == "__main__":
    main()

