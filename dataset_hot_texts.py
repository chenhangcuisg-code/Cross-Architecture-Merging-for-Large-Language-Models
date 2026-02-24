# -*- coding: utf-8 -*-
"""
dataset_hot_texts.py

把 “根据 --data / --data-split / --max-samples 生成 texts 列表” 的逻辑
从 run_activs_and_hot.py 里抽出来，避免主脚本越来越乱。

目前支持：
- malay / indonesian / eng  -> dataset_general_texts.load_general_english_texts
- gsm8k                      -> dataset_gsm8k.load_gsm8k_texts
- medical                    -> HF: Shekswess/medical_llama3_instruct_dataset
"""

from typing import Optional
import os
import random
import time

import torch
from datasets import load_dataset, Dataset, get_dataset_split_names
from transformers import AutoTokenizer

# 配置 Hugging Face 镜像站点（如果未设置）
if "HF_ENDPOINT" not in os.environ:
    # 使用中国镜像站点，提高下载速度
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[Config] Using Hugging Face mirror: {os.environ['HF_ENDPOINT']}")
else:
    print(f"[Config] Using Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")

# 同时设置 HF_HUB_ENABLE_HF_TRANSFER 以加速下载（如果支持）
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None  # optional; only needed for vision SFT datasets

from typing import List, Optional

from data_loading.dataset_general_texts import load_general_english_texts
from data_loading.dataset_gsm8k import load_gsm8k_texts

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

 
MEDICAL_DATASET_NAME = "Shekswess/medical_llama3_instruct_dataset"
FINANCE_DATASET_NAME = "Josephgflowers/Finance-Instruct-500k"
FINANCE_ALPACA_DATASET_NAME = "gbharti/finance-alpaca"
ALPACA_DATASET_NAME = "tatsu-lab/alpaca"
CANTONESE_DIALOGUE_DATASET_NAME = "stvlynn/Cantonese-Dialogue"
CANTONESE_COT_DATASET_NAME = "indiejoseph/cantonese-cot"
# (existing dataset constants)
THAI_FINEWEB_DATASET_NAME = "Suraponn/thai_instruction_sft"  # primary Thai SFT dataset ChavyvAkvar/fineweb-2-1M-Sample-ThaiSuraponn/thai_instruction_sft
THAI_FINEWEB_LEGACY_DATASET_NAME = "ChavyvAkvar/fineweb-2-1M-Sample-Thai"
FINEWEB_SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless Thai language assistant."
)
ALPACA_SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless AI assistant."
)
CANTONESE_SYSTEM_PROMPT = (
    "You are a helpful, honest, and harmless Cantonese language assistant."
)

CHAT_SYSTEM_TAG = "<|start_header_id|>system<|end_header_id|>"
CHAT_USER_TAG = "<|start_header_id|>user<|end_header_id|>"
CHAT_ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"
CHAT_EOT = "<|eot_id|>"


CHAT_TEMPLATE_CONFIGS = {
    "llama3": {
        "system_prompt": FINEWEB_SYSTEM_PROMPT,
    },
    "qwen2": {
        "system_prompt": FINEWEB_SYSTEM_PROMPT,
    },
    "qwen2.5": {
        "system_prompt": FINEWEB_SYSTEM_PROMPT,
    },
}


def _should_use_apply_chat_template(tokenizer: Optional[AutoTokenizer]) -> bool:
    """
    检测是否应该使用 tokenizer.apply_chat_template 而不是手动构建格式。
    对于 Qwen2.5 模型，推荐使用 apply_chat_template。
    """
    if tokenizer is None:
        return False
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    # 检查 tokenizer 名称或配置中是否包含 qwen2.5
    tokenizer_name = getattr(tokenizer, "name_or_path", "") or ""
    if "qwen2.5" in tokenizer_name.lower() or "qwen2_5" in tokenizer_name.lower():
        return True
    # 检查是否有 chat_template 属性（Qwen2.5 通常有）
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        # 进一步检查是否是 Qwen 系列的模板
        if "qwen" in str(tokenizer.chat_template).lower():
            return True
    return False


def _role_to_tag(role: str) -> str:
    role = (role or "").lower()
    if role == "assistant":
        return CHAT_ASSISTANT_TAG
    if role == "system":
        return CHAT_SYSTEM_TAG
    return CHAT_USER_TAG


def build_chat_template(
    system_prompt: str,
    user_text: str,
    assistant_text: str = "",
    *,
    bos_token: str = "",
    eos_token: str = "",
    tokenizer: Optional[AutoTokenizer] = None,
    use_apply_chat_template: bool = False,
) -> str:
    """
    把系统/用户/助手内容拼成 LLaMA3/Qwen2/Qwen2.5 的 chat 模板字符串。
    
    如果提供了 tokenizer 且 use_apply_chat_template=True，会优先使用 tokenizer.apply_chat_template
    （推荐用于 Qwen2.5 模型）。否则使用手动构建的 LLaMA3 兼容格式。
    """
    # 优先使用 tokenizer.apply_chat_template（推荐用于 Qwen2.5）
    if tokenizer is not None and use_apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not bool(assistant_text),
            )
            return text
        except Exception as e:
            print(f"[Warn] apply_chat_template failed: {e}, falling back to manual format")
    
    # 回退到手动构建的 LLaMA3 兼容格式（适用于 LLaMA3、Qwen2、Qwen2.5）
    system_block = CHAT_SYSTEM_TAG + "\n" + system_prompt + CHAT_EOT
    user_block = CHAT_USER_TAG + "\n" + user_text + CHAT_EOT
    assistant_block = CHAT_ASSISTANT_TAG + "\n" + assistant_text + CHAT_EOT
    return bos_token + system_block + user_block + assistant_block + eos_token


def build_chat_from_messages(
    messages: list[dict],
    *,
    system_prompt: str,
    assistant_fallback: str,
    bos_token: str = "",
    eos_token: str = "",
) -> str:
    """
    将 messages 列表拼成完整的 chat 模板（包含 system block）。
    """
    chat_parts = []
    chat_parts.append(CHAT_SYSTEM_TAG + "\n" + system_prompt + CHAT_EOT)
    assistant_seen = False
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        tag = _role_to_tag(msg.get("role"))
        chat_parts.append(tag + "\n" + content + CHAT_EOT)
        if tag == CHAT_ASSISTANT_TAG:
            assistant_seen = True
    if not assistant_seen and assistant_fallback:
        chat_parts.append(CHAT_ASSISTANT_TAG + "\n" + assistant_fallback + CHAT_EOT)
    return bos_token + "".join(chat_parts) + eos_token


INSTRUCTION_KEYS = (
    "instruction",
    "Instruction",
    "prompt",
    "Prompt",
    "input",
    "Input",
    "text",
    "Text",
)
ANSWER_KEYS = (
    "answer",
    "Answer",
    "output",
    "Output",
    "response",
    "Response",
)


def _first_field_value(example: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value:
            return value
    return ""


def _normalize_messages(value) -> list[dict]:
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict)]
    return []

def build_sft_dataset(
    data: str,
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    统一入口：根据 data 构建 tokenized SFT dataset（LLaMA3 chat 格式）。
    
    data:
      - 'malay_sft'          -> Malaysian instruction SFT
      - 'medical_sft'        -> Shekswess/medical_llama3_instruct_dataset
      - 'indoconv_sft'       -> izzulgod/indonesian-conversation
      - 'geo3k_text_sft'     -> Geometry3K text-only
      - 'onevision_text_sft' -> OneVision CLEVR text-only
      - 'alpaca_sft'         -> tatsu-lab/alpaca 指令 SFT
    """
    data_lower = data.lower()

    if data_lower == "malay_sft":
        return build_sft_dataset_from_malaysian_sft(
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
            max_length=max_length,
        )

    elif data_lower == "medical_sft":
        return build_medical_llama3_sft_dataset(
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
            max_length=max_length,
        )

    elif data_lower == "indoconv_sft":
        return build_indoconv_sft_dataset(
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
            max_length=max_length,
        )

    elif data_lower == "geo3k_text_sft":
        return build_geo3k_text_sft_dataset(
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
            max_length=max_length,
        )

    elif data_lower == "onevision_text_sft":
        return build_onevision_clevr_text_sft_dataset(
            tokenizer=tokenizer,
            subset="dvqa_train_200k.jsonl",
            split=split,
            max_samples=max_samples,
            max_length=max_length,
        )

    elif data_lower == "alpaca_sft":
        return build_alpaca_dataset(
            tokenizer=tokenizer,
            split=split,
            max_samples=max_samples,
            max_length=max_length,
        )

    else:
        raise ValueError(
            f"未知的 SFT 数据集 data='{data}'. "
            "可用：malay_sft / medical_sft / indoconv_sft / geo3k_text_sft / onevision_text_sft / alpaca_sft"
        )

def load_medical_llama3_texts(
    split: str = "train",
    max_samples: Optional[int] = None,
    field: str = "prompt",
    mode: str = "sft",
    bos_token: str = "",
) -> List[str]:
    def build_medical_llama3_sft_text(example, bos_token=""):
        """
        从医疗数据 example 构建 LLaMA3 chat 文本（不做 tokenize）。
        """
        p = (example.get("prompt") or "").strip()

        if not p:
            instr = (example.get("instruction") or "").strip()
            inp = (example.get("input") or "").strip()
            out = (example.get("output") or "").strip()

            system_prompt = (
                instr
                if instr
                else "You are a helpful, honest, and harmless medical AI assistant."
            )
            user_text = inp or "Please answer the medical question."

            p = (
                "<|start_header_id|>system<|end_header_id|>\n"
                + system_prompt
                + "<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n"
                + user_text
                + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n"
                + out
                + "<|eot_id|>"
            )

        if bos_token and not p.startswith(bos_token):
            p = bos_token + p

        return p

    """
    加载 Shekswess/medical_llama3_instruct_dataset。
    mode = "raw" → 返回某个字段，如 prompt/input/output
    mode = "sft" → 返回 LLaMA3 chat 格式字符串
    """
    if load_dataset is None:
        raise ImportError("需要安装 datasets，请运行: pip install datasets")

    ds = load_dataset(MEDICAL_DATASET_NAME, split=split)

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    # ---- SFT mode ----
    if mode == "sft":
        texts = [
            build_medical_llama3_sft_text(ex, bos_token=bos_token)
            for ex in ds
        ]
        return texts

    # ---- raw mode ----
    if field not in ds.column_names:
        raise KeyError(
            f"列 `{field}` 不在 {MEDICAL_DATASET_NAME} 中，可选：{ds.column_names}"
        )

    return list(ds[field])
def load_finance_instruct_texts(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    加载 Josephgflowers/Finance-Instruct-500k 金融指令数据，返回 LLaMA3 chat 格式的纯文本列表。

    数据集列：
      - system:   系统提示（有些样本可能为空）
      - user:     用户问题 / 指令
      - assistant: 模型回答

    我们拼成 LLaMA 3 chat 模板：

        <|start_header_id|>system<|end_header_id|> ... <|eot_id|>
        <|start_header_id|>user<|end_header_id|>   ... <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|> ... <|eot_id|>
    """
    if load_dataset is None:
        raise ImportError(
            "需要安装 `datasets` 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    ds = load_dataset(FINANCE_DATASET_NAME, split=split)

    required_cols = {"system", "user", "assistant"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise KeyError(
            f"数据集 {FINANCE_DATASET_NAME} 缺少列: {missing}，"
            f"当前列为: {ds.column_names}"
        )

    texts: List[str] = []

    default_system_prompt = (
        "You are a helpful, honest, and harmless AI assistant specialized "
        "in finance and economics."
    )

    for ex in ds:
        system = (ex.get("system") or "").strip()
        user = (ex.get("user") or "").strip()
        assistant = (ex.get("assistant") or "").strip()

        if not system:
            system = default_system_prompt

        text = (
            "<|start_header_id|>system<|end_header_id|>\n"
            + system
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + assistant
            + "<|eot_id|>"
        )
        texts.append(text)

        if max_samples is not None and len(texts) >= max_samples:
            break

    return texts


def load_alpaca_texts(
    split: str = "train",
    max_samples: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> List[str]:
    """
    加载 tatsu-lab/alpaca 指令数据，返回 LLaMA3 chat 格式的纯文本列表。
    """
    if load_dataset is None:
        raise ImportError(
            "需要安装 datasets 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    dataset_name = dataset_name or ALPACA_DATASET_NAME
    print(f"[Data][Alpaca] Loading {dataset_name} split={split} max_samples={max_samples}")
    ds = load_dataset(dataset_name, split=split)

    limit = max_samples if max_samples is not None and max_samples > 0 else None
    texts: List[str] = []
    empty_output_count = 0
    sample_count = 0
    
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()

        if not instr:
            instr = "Provide a helpful response."

        # 检查 output 是否为空
        if not out:
            empty_output_count += 1
            if empty_output_count <= 3:  # 只打印前3个警告
                print(f"[Warning] 发现空的 output 字段 (instruction: {instr[:50]}...)")

        # 构建 user content：如果 input 不为空，则组合 instruction 和 input
        user_content = instr
        if inp:
            user_content = f"{instr}\n\nInput:\n{inp}"

        # 转换为 LLaMA3 chat 格式
        # 格式：<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>
        #      <|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>
        #      <|start_header_id|>assistant<|end_header_id|>\n{output}<|eot_id|>
        text = (
            "<|start_header_id|>system<|end_header_id|>\n"
            + ALPACA_SYSTEM_PROMPT
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user_content
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + out
            + "<|eot_id|>"
        )

        texts.append(text)
        sample_count += 1
        
        # 打印前3个样本的格式验证信息
        if sample_count <= 3:
            print(f"[Debug] 样本 {sample_count}:")
            print(f"  Instruction: {instr[:80]}...")
            print(f"  Input: {inp[:80] if inp else '(empty)'}...")
            print(f"  Output: {out[:80]}...")
            print(f"  转换后的文本长度: {len(text)} 字符")
            print(f"  包含 <|start_header_id|>: {'<|start_header_id|>' in text}")
            print(f"  包含 <|eot_id|>: {'<|eot_id|>' in text}")
            print(f"  包含 assistant header: {'<|start_header_id|>assistant<|end_header_id|>' in text}")
            print(f"  包含 output 内容: {out[:50] in text if out else 'N/A (output为空)'}")
        
        if limit is not None and len(texts) >= limit:
            break

    if empty_output_count > 0:
        print(f"[Warning] 总共发现 {empty_output_count} 个空的 output 字段")
    print(f"[Info] 加载了 {len(texts)} 个样本，每个样本包含完整的 instruction 和 output")
    print(f"[Info] 格式：LLaMA3 chat 格式（包含 system, user, assistant 三个部分）")

    return texts


def load_texts(
    data: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    mode: str = "raw",      # ★ 新增参数：raw / sft
    thai_dataset_name: Optional[str] = None,
) -> List[str] | List[dict]:
    """
    根据 data / split / max_samples 返回 List[str]

    data:
      - malay / indonesian / eng / cantonese  → 通用语料
      - gsm8k                     → GSM8K
      - medical                  → 默认 SFT（LLaMA3 chat）
      - alpaca                   → tatsu-lab/alpaca 指令 SFT（LLaMA3 chat）
    """
    data_lower = data.lower()

    if data_lower in {"malay", "indonesian", "eng", "cantonese"}:
        print(f"[Info] Loading general texts subset={data_lower}, mode={mode}")
        texts = load_general_english_texts(
            subset=data_lower, split=split, max_samples=max_samples
        )

    elif data_lower == "gsm8k":
        print(f"[Info] Loading GSM8K, mode={mode}")
        texts = load_gsm8k_texts(
            subset="main", split=split, max_samples=max_samples
        )

    elif data_lower == "finance":
        print(
            f"[Info] (finance) Loading {FINANCE_DATASET_NAME}, "
            f"split={split}, max_samples={max_samples}"
        )
        texts = load_finance_instruct_texts(
            split=split,
            max_samples=max_samples,
        )

    elif data_lower == "medical":
        print(
            f"[Info] (medical) Loading {MEDICAL_DATASET_NAME}, "
            f"split={split}, max_samples={max_samples}, field=prompt"
        )
        texts = load_medical_llama3_texts(
            split=split,
            max_samples=max_samples,
            field="prompt",  # 如需用 input/output 改这里即可
        )

    elif data_lower == "alpaca":
        print(
            f"[Info] (alpaca) Loading {ALPACA_DATASET_NAME}, "
            f"split={split}, max_samples={max_samples}"
        )
        texts = load_alpaca_texts(
            split=split,
            max_samples=max_samples,
        )

    elif data_lower in {"fineweb_thai", "thai_fineweb"}:
        dataset_to_load = thai_dataset_name or THAI_FINEWEB_DATASET_NAME
        print(
            f"[Info] (thai) Loading {dataset_to_load}, "
            f"split={split}, max_samples={max_samples}"
        )
        texts = load_fineweb_thai_texts(
            split=split,
            max_samples=max_samples,
            dataset_name=dataset_to_load,
        )

    else:
        raise ValueError(
            "未知数据集，可选：malay / indonesian / eng / cantonese / gsm8k / medical / alpaca"
        )

    print(f"[Info] Loaded {len(texts)} texts.")
    return texts


# -*- coding: utf-8 -*-
"""
dataset_hot_texts.py

集中所有「数据相关」代码：
- Malaysian-SFT 指令 SFT
- Shekswess/medical_llama3_instruct_dataset 医疗 SFT
- Geometry3K：纯文本 SFT + Qwen2-VL 多模态 collator
- LLaVA-OneVision CLEVR：纯文本 SFT + Qwen2-VL 多模态 collator
- 通用印尼/马来/英文无监督语料（C4 等）
- izzulgod/indonesian-conversation 多轮 SFT
"""


# ==========================================
# Malaysian-SFT 指令数据：一条一条样本 + random_all（LLaMA 3 chat 格式）
# ==========================================


def build_sft_dataset_from_malaysian_sft(
    tokenizer: AutoTokenizer,
    split: str = "google_translate_camel_ai",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    使用 mesolitica/Malaysian-SFT 做指令微调 (instruction tuning)。

    每一条样本会被格式化成 chat 格式：
    - 对于 Qwen2.5 模型：自动使用 tokenizer.apply_chat_template（推荐）
    - 对于其他模型（LLaMA3、Qwen2）：使用手动构建的 LLaMA3 兼容格式：
      
      <|start_header_id|>system<|end_header_id|>
      {system_prompt}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      {user_content}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>
      {output}{eos}
    """
    print(f"[Data] Malaysian-SFT mode = {split}")

    # ========== Step 1：装载 / 采样原始样本 ==========
    if split == "random_all":
        try:
            all_splits = get_dataset_split_names("mesolitica/Malaysian-SFT")
            print(f"[Data] Found splits in mesolitica/Malaysian-SFT: {all_splits}")
        except Exception as e:
            print(f"[Data][Warn] get_dataset_split_names failed: {e}")
            all_splits = ["google_translate_camel_ai"]
            print(f"[Data][Warn] Fallback to fixed split list: {all_splits}")

        all_samples = []
        for sp in all_splits:
            print(f"[Data] Loading split: {sp}")
            ds_sp = load_dataset("mesolitica/Malaysian-SFT", split=sp)
            all_samples.extend(ds_sp)

        total = len(all_samples)
        print(f"[Data] Total samples across ALL splits = {total}")

        if max_samples is None or max_samples <= 0 or max_samples >= total:
            max_samples = total
            print(f"[Data] max_samples is None / <=0 / >= total, using all {total} samples")
        else:
            print(f"[Data] Randomly selecting {max_samples} samples from {total}")
            random.shuffle(all_samples)
            all_samples = all_samples[:max_samples]

        raw_ds = Dataset.from_list(all_samples)

    else:
        print(f"[Data] Loading mesolitica/Malaysian-SFT split={split}")
        raw_ds = load_dataset("mesolitica/Malaysian-SFT", split=split)

        if max_samples is not None and max_samples > 0:
            max_samples = min(max_samples, len(raw_ds))
            print(f"[Data] Subsampling to first {max_samples} examples")
            raw_ds = raw_ds.select(range(max_samples))

    # ========== Step 2：格式化为 chat 文本（自动检测 Qwen2.5 使用 apply_chat_template）==========
    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    system_prompt = "You are a helpful, honest, and harmless AI assistant."
    use_apply_template = _should_use_apply_chat_template(tokenizer)

    def format_example(example):
        instr = (example.get("prompt_input") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        if inp:
            user_content = instr + "\n\nInput:\n" + inp
        else:
            user_content = instr

        # 对于 Qwen2.5，优先使用 apply_chat_template
        if use_apply_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # 手动构建 LLaMA3 兼容格式（适用于 LLaMA3、Qwen2）
            text = (
                bos
                + "<|start_header_id|>system<|end_header_id|>\n"
                + system_prompt
                + "<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n"
                + user_content
                + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n"
                + out
                + eos
            )

        return {"text": text}

    if use_apply_template:
        print("[Data] Formatting Malaysian-SFT using tokenizer.apply_chat_template (Qwen2.5 compatible)...")
    else:
        print("[Data] Formatting Malaysian-SFT as LLaMA3 chat strings...")
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting Malaysian-SFT",
    )

    # ========== Step 3：Tokenize ==========
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data] Tokenizing SFT dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing SFT dataset",
    )

    return tokenized_ds


# ==========================================
# 新增：Shekswess/medical_llama3_instruct_dataset (医疗 SFT)
# ==========================================


def build_medical_llama3_sft_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    使用 Shekswess/medical_llama3_instruct_dataset 做指令微调。

    数据集列：
      - instruction: 基本固定，"Answer the question truthfully, you are a medical professional."
      - input      : 问题
      - output     : 回答
      - prompt     : 已经拼好的 LLaMA3 chat 串（system/user/assistant + <|eot_id|>）

    我们优先直接用 prompt 字段；如果某条样本缺 prompt，则退化为根据
    instruction / input / output 简单拼一个 LLaMA3 chat 格式。
    """
    print(
        f"[Data][Medical-LLaMA3] Loading "
        f"Shekswess/medical_llama3_instruct_dataset split={split}"
    )
    raw_ds = load_dataset(
        "Shekswess/medical_llama3_instruct_dataset",
        split=split,
    )

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(f"[Data][Medical-LLaMA3] Subsampling to first {max_samples} examples")
        raw_ds = raw_ds.select(range(max_samples))

    bos = tokenizer.bos_token or ""

    def format_example(example):
        # 1) 正常路径：用原始 prompt
        p = (example.get("prompt") or "").strip()

        # 2) 兜底路径：如果 prompt 缺失，就自己拼一个聊天模板
        if not p:
            instr = (example.get("instruction") or "").strip()
            inp = (example.get("input") or "").strip()
            out = (example.get("output") or "").strip()

            system_prompt = (
                instr
                if instr
                else "You are a helpful, honest, and harmless medical AI assistant."
            )
            user_text = inp or "Please answer the medical question."

            p = (
                "<|start_header_id|>system<|end_header_id|>\n"
                + system_prompt
                + "<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n"
                + user_text
                + "<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n"
                + out
                + "<|eot_id|>"
            )

        text = bos + p if bos and not p.startswith(bos) else p
        return {"text": text}

    print("[Data][Medical-LLaMA3] Formatting as LLaMA3 chat strings...")
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting medical_llama3_instruct_dataset",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Medical-LLaMA3] Tokenizing dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing medical_llama3_instruct_dataset",
    )

    return tokenized_ds


def build_alpaca_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    dataset_name: Optional[str] = None,
) -> Dataset:
    """
    使用 tatsu-lab/alpaca 数据构建 LLaMA3 chat 格式的训练集。
    """
    dataset_name = dataset_name or ALPACA_DATASET_NAME
    if load_dataset is None:
        raise ImportError(
            "需要安装 datasets 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    print(f"[Data][Alpaca] Loading {dataset_name} split={split}")
    raw_ds = load_dataset(dataset_name, split=split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(f"[Data][Alpaca] Subsampling to first {max_samples} examples")
        raw_ds = raw_ds.select(range(max_samples))

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    def format_example(example):
        instr = (example.get("instruction") or "").strip()
        if not instr:
            instr = "Provide a helpful response."
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        user_content = instr
        if inp:
            user_content = f"{instr}\n\nInput:\n{inp}"

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + ALPACA_SYSTEM_PROMPT
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user_content
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + out
            + eos
        )
        return {"text": text}

    print("[Data][Alpaca] Formatting as LLaMA3 chat strings...")
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting alpaca dataset",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Alpaca] Tokenizing dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing alpaca dataset",
    )

    return tokenized_ds

def load_finance_alpaca_texts(
    split: str = "train",
    max_samples: Optional[int] = None,
    dataset_name: Optional[str] = None,
) -> List[str]:
    """
    加载 gbharti/finance-alpaca 金融指令数据，返回 LLaMA3 chat 格式的纯文本列表。
    
    数据集列（Alpaca 格式）：
      - instruction: 指令/问题
      - input:       输入（可能为空）
      - output:      回答
    """
    if load_dataset is None:
        raise ImportError(
            "需要安装 datasets 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    dataset_name = dataset_name or FINANCE_ALPACA_DATASET_NAME
    print(f"[Data][Finance-Alpaca] Loading {dataset_name} split={split} max_samples={max_samples}")
    ds = load_dataset(dataset_name, split=split)

    limit = max_samples if max_samples is not None and max_samples > 0 else None
    texts: List[str] = []
    
    finance_system_prompt = (
        "You are a helpful, honest, and harmless AI assistant specialized "
        "in finance and economics."
    )
    
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()

        if not instr:
            instr = "Provide a helpful response."

        user_content = instr
        if inp:
            user_content = f"{instr}\n\nInput:\n{inp}"

        text = (
            "<|start_header_id|>system<|end_header_id|>\n"
            + finance_system_prompt
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user_content
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + out
            + "<|eot_id|>"
        )

        texts.append(text)
        if limit is not None and len(texts) >= limit:
            break

    return texts

def build_finance_alpaca_sft_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    dataset_name: Optional[str] = None,
) -> Dataset:
    """
    使用 gbharti/finance-alpaca 做指令微调（LLaMA3 chat 格式）。
    
    数据集列（Alpaca 格式）：
      - instruction: 指令/问题
      - input:       输入（可能为空）
      - output:      回答
    """
    dataset_name = dataset_name or FINANCE_ALPACA_DATASET_NAME
    if load_dataset is None:
        raise ImportError(
            "需要安装 datasets 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    print(f"[Data][Finance-Alpaca] Loading {dataset_name} split={split}")
    raw_ds = load_dataset(dataset_name, split=split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(f"[Data][Finance-Alpaca] Subsampling to first {max_samples} examples")
        raw_ds = raw_ds.select(range(max_samples))

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    finance_system_prompt = (
        "You are a helpful, honest, and harmless AI assistant "
        "specialized in finance and economics."
    )

    def format_example(example):
        instr = (example.get("instruction") or "").strip()
        if not instr:
            instr = "Provide a helpful response."
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        user_content = instr
        if inp:
            user_content = f"{instr}\n\nInput:\n{inp}"

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + finance_system_prompt
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user_content
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + out
            + eos
        )
        return {"text": text}

    print("[Data][Finance-Alpaca] Formatting as LLaMA3 chat strings...")
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting finance-alpaca dataset",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Finance-Alpaca] Tokenizing dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing finance-alpaca dataset",
    )

    return tokenized_ds

def build_finance_instruct_sft_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    dataset_name: Optional[str] = None,
) -> Dataset:
    """
    使用 Josephgflowers/Finance-Instruct-500k 做指令微调（LLaMA3 chat 格式）。

    数据集列：
      - system:   系统提示（有些样本可能为空）
      - user:     用户问题 / 指令
      - assistant: 模型回答

    我们统一拼成：

      <|start_header_id|>system<|end_header_id|>
      {system_prompt}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      {user_content}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>
      {assistant}{eos}
    """
    dataset_name = dataset_name or FINANCE_DATASET_NAME
    print(
        f"[Data][Finance] Loading {dataset_name} split={split}"
    )
    raw_ds = load_dataset(
        dataset_name,
        split=split,
    )

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(f"[Data][Finance] Subsampling to first {max_samples} examples")
        raw_ds = raw_ds.select(range(max_samples))

    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    default_system_prompt = (
        "You are a helpful, honest, and harmless AI assistant "
        "specialized in finance and economics."
    )

    def format_example(example):
        system = (example.get("system") or "").strip()
        user = (example.get("user") or "").strip()
        assistant = (example.get("assistant") or "").strip()

        if not system:
            system = default_system_prompt

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + system
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + assistant
            + eos
        )
        return {"text": text}

    print("[Data][Finance] Formatting Finance-Instruct-500k as LLaMA3 chat strings...")
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting Finance-Instruct-500k (LLaMA3 chat)",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Finance] Tokenizing Finance-Instruct-500k dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing Finance-Instruct-500k dataset",
    )

    return tokenized_ds

# ==========================================
# Qwen2-VL: hiyouga/geometry3k 多模态 VQA 数据集
# ==========================================


def load_geo3k_raw_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
):
    """
    Geometry3K 数据集结构：
      - images: List[Image]
      - problem: str
      - answer: str
    """
    print(f"[Data][Geo3K] Loading hiyouga/geometry3k split={split}")
    ds = load_dataset("hiyouga/geometry3k", split=split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(ds))
        print(f"[Data][Geo3K] Subsampling to first {max_samples} examples")
        ds = ds.select(range(max_samples))

    print(ds)
    return ds


def build_geo3k_text_sft_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    将 hiyouga/geometry3k 视为**纯文本指令数据**来做 SFT，
    主要用于 llama / tinyllava 模型：
    """
    print(f"[Data][Geo3K-Text] Loading geometry3k as text-only, split={split}")
    ds = load_dataset("hiyouga/geometry3k", split=split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(ds))
        print(f"[Data][Geo3K-Text] Subsampling to first {max_samples} examples")
        ds = ds.select(range(max_samples))

    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    system_prompt = "You are a helpful, honest, and harmless AI assistant for solving geometry problems."

    def format_example(example):
        problem = (example.get("problem") or "").strip()
        answer = (example.get("answer") or "").strip()

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + system_prompt
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + problem
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + answer
            + eos
        )
        return {"text": text}

    print("[Data][Geo3K-Text] Formatting geometry3k as LLaMA3 chat strings (TEXT ONLY, drop images)...")
    ds_text = ds.map(
        format_example,
        remove_columns=ds.column_names,
        desc="Formatting Geo3K (text-only chat)",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Geo3K-Text] Tokenizing Geo3K text-only dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing Geo3K text-only dataset",
    )

    return tokenized_ds


class QwenGeo3KCollator:
    """
    用于 Qwen2-VL + Geometry3K 的多模态 collator。
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features):
        conversations = []
        for ex in features:
            imgs = ex["images"]
            problem = ex["problem"]
            answer = ex["answer"]

            user_content = []
            for img in imgs:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": problem})

            messages = [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                },
            ]
            conversations.append(messages)

        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(conversations)

        batch = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch["labels"] = batch["input_ids"].clone()
        return batch


# ==========================================
# LLaVA-OneVision-1.5-Instruct-Data / CLEVR
# ==========================================


def _strip_leading_image_token(text: str) -> str:
    """
    数据中 user 的 content 以 `<image>\n` 开头；
    对于我们自己构建的对话格式，可以把这个占位符去掉，
    真实图像由 Qwen2-VL 的 image 输入负责。
    """
    if not isinstance(text, str):
        return ""
    t = text.lstrip()
    if t.startswith("<image>"):
        t = t[len("<image>") :]
        while t.startswith("\n"):
            t = t[1:]
    return t


def load_onevision_clevr_raw_dataset(
    subset="dvqa_train_200k.jsonl",
    split="train",
    max_samples=5000,
    seed=42,
):
    print(f"[Data][OneVision] Streaming subset={subset}, split={split}")
    random.seed(seed)

    ds_stream = load_dataset(
        "OpenGVLab/InternVL-Chat-V1-2-SFT-Data", #OpenGVLab/InternVL-Chat-V1-2-SFT-Data mvp-lab/LLaVA-OneVision-1.5-Instruct-Data
        subset,
        split=split,
        streaming=True,
    )

    reservoir = []
    n = 0

    for item in ds_stream:
        n += 1
        if len(reservoir) <  max_samples:
            reservoir.append(item)
        else:
            j = random.randint(1, n)
            if j <= max_samples:
                reservoir[j - 1] = item

    print(f"[Data][OneVision] Randomly sampled {len(reservoir)} items from streaming dataset")
    return reservoir

def build_onevision_clevr_text_sft_dataset(
    tokenizer: AutoTokenizer,
    subset: str = "dvqa_train_200k.jsonl",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
 



    ds = load_onevision_clevr_raw_dataset(
        subset=subset,
        split=split,
        max_samples=max_samples,
    )
    # raw 是 Python list，需要转换
    ds = Dataset.from_list(ds)

    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    system_prompt = (
        "You are a helpful, honest, and harmless vision-language AI assistant. "
        "The user may refer to an image, but you only see the text here."
    )

    def format_example(example):
        messages = example.get("conversations") or []

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + system_prompt
            + "<|eot_id|>"
        )

        for i, msg in enumerate(messages):
            role = (msg.get("role") or "user").strip()
            content = (msg.get("content") or "")
            content = _strip_leading_image_token(content).strip()

            if role not in ("user", "assistant"):
                role = "user"

            is_last = i == len(messages) - 1

            if role == "assistant" and is_last:
                text += (
                    f"<|start_header_id|>{role}<|end_header_id|>\n"
                    + content
                    + eos
                )
            else:
                text += (
                    f"<|start_header_id|>{role}<|end_header_id|>\n"
                    + content
                    + "<|eot_id|>"
                )

        return {"text": text}

    print("[Data][OneVision-Text] Formatting as LLaMA3 chat strings (multi-turn, text only)...")
    ds_text = ds.map(
        format_example,
        remove_columns=ds.column_names,
        desc="Formatting OneVision CLEVR as LLaMA3 chat (text-only)",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][OneVision-Text] Tokenizing OneVision CLEVR text-only dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing OneVision CLEVR text-only dataset",
    )

    return tokenized_ds

class TinyLlavaOneVisionCLEVRDataCollator:
    """
    TinyLLaVA + LLaVA-OneVision / InternVL-Chat-V1-2-SFT-Data 多模态 collator。

    输入数据（示意）：
      {
        "image": <PIL.Image 或路径>,
        "conversations": [
          {"role": "user" / "assistant", "content": "..."}  # 如果你自己预处理过
          或
          {"from": "human" / "gpt", "value": "<image>\\nQuestion ..."}  # 原始 InternVL 格式
        ],
      }

    这里做的事：
      - 把第一条 human/user 消息里的 `<image>` 占位符去掉文字里的那一份
      - 在文本里插入 TinyLLaVA 习惯的 `<image>` token
      - 用 tokenizer 做文本编码
      - 用 image_processor 做图像预处理，打包到 `images` / `image_sizes` 里
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
        max_length: int = 2048,
        image_token: str = "<image>",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_token = image_token

    def __call__(self, features):
        texts = []
        images = []

        for ex in features:
            image = ex["image"]
            raw_msgs = ex.get("conversations") or []

            # 把原始对话整理成 TinyLLaVA 文本模板
            # 这里用非常朴素的形式：
            #   USER: <image>\n{question}
            #   ASSISTANT: {answer}
            conv_str_parts = []

            for i, m in enumerate(raw_msgs):
                # 兼容两种字段命名：role/content vs from/value
                role = (m.get("role") or m.get("from") or "user").strip().lower()
                content = (m.get("content") or m.get("value") or "").strip()

                # 去掉内容里的占位符 <image>
                content = _strip_leading_image_token(content).strip()

                if role in ("human", "user"):
                    if i == 0:
                        # 第一条 human：插入图像 token
                        conv_str_parts.append(
                            f"USER: {self.image_token}\n{content}\n"
                        )
                    else:
                        conv_str_parts.append(
                            f"USER: {content}\n"
                        )
                else:
                    # 统一当作 assistant
                    conv_str_parts.append(
                        f"ASSISTANT: {content}\n"
                    )

            if not conv_str_parts:
                # 没有有效消息就跳过该样本
                continue

            conv_text = "".join(conv_str_parts).strip()
            texts.append(conv_text)
            images.append(image)

        # --- 文本部分：tokenize ---
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()

        # --- 图像部分：用 TinyLLaVA 的 image_processor 预处理 ---
        image_tensors = []
        image_sizes = []

        for img in images:
            # 兼容两种接口：.preprocess(...) 或 __call__(...)
            if hasattr(self.image_processor, "preprocess"):
                processed = self.image_processor.preprocess(img, return_tensors="pt")
            else:
                processed = self.image_processor(img, return_tensors="pt")

            if isinstance(processed, dict):
                if "pixel_values" in processed:
                    img_tensor = processed["pixel_values"][0]
                else:
                    # 保底：取 dict 里的第一个键
                    first_key = list(processed.keys())[0]
                    img_tensor = processed[first_key][0]
            else:
                # processed 直接是 Tensor / list[Tensor]
                img_tensor = processed[0]

            image_tensors.append(img_tensor)
            # 记录下 H, W 作为 image_sizes（TinyLLaVA forward 里可以用得到）
            image_sizes.append([img_tensor.shape[-2], img_tensor.shape[-1]])

        images_batch = torch.stack(image_tensors, dim=0)
        image_sizes_tensor = torch.tensor(image_sizes, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "image_sizes": image_sizes_tensor,
        }
        return batch

class QwenOneVisionCLEVRDataCollator:
    """
    用于 Qwen2-VL + LLaVA-OneVision CLEVR 子集的多模态 collator。

    每条样本结构（示意）：
      {
        "image": <PIL.Image>,
        "conversations": [
          {"role": "user", "content": "<image>\\nQuestion..."},
          {"role": "assistant", "content": "Answer..."},
          ...
        ],
        ...
      }

    我们会：
      - 把第一条 user message 里的 `<image>` 占位符去掉
      - 把图像挂到第一条 user 的 content 上
      - 其余轮次只保留纯文本 content
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features):
        conversations = []

        for ex in features:
            image = ex["image"]
            raw_msgs = ex.get("conversations") or []

            msgs = []
            for i, m in enumerate(raw_msgs):
                role = (m.get("role") or "user").strip()
                if role not in ("user", "assistant", "system"):
                    role = "user"

                content_text = (m.get("content") or "")
                content_text = _strip_leading_image_token(content_text).strip()

                content_items = []

                # 只在第一条 user 消息上挂图像
                if role == "user" and i == 0:
                    content_items.append({"type": "image", "image": image})

                if content_text:
                    content_items.append({"type": "text", "text": content_text})

                if not content_items:
                    content_items.append({"type": "text", "text": ""})

                msgs.append(
                    {
                        "role": role,
                        "content": content_items,
                    }
                )

            if not msgs:
                continue

            conversations.append(msgs)

        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )

        image_inputs, video_inputs = process_vision_info(conversations)

        batch = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch["labels"] = batch["input_ids"].clone()
        return batch


# ==========================================
# 通用英文 / 多语言无监督文本：支持 Indonesian / Malay 等
# ==========================================


def load_general_english_texts(subset="indonesian", split="train", max_samples=2000):
    """
    subset: ["common", "wiki", "imdb", "c4", "indonesian", "malay", "cantonese"]
    """
    streaming = False
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "hf_cache")

    if subset == "common":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    elif subset == "wiki":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    elif subset == "imdb":
        ds = load_dataset("imdb", split=split)

    elif subset == "c4":
        ds = load_dataset("allenai/mc4", "en", split=split)

    elif subset == "indonesian":
        try:
            print(
                f"[Data][Indonesian] Trying local izzulgod/indonesian-conversation "
                f"cache_dir={cache_dir}"
            )
            ds = load_dataset(
                "izzulgod/indonesian-conversation",
                split="train",
                cache_dir=cache_dir,
            )
            streaming = False

            if max_samples is None or max_samples <= 0:
                max_samples = len(ds)
            max_samples = min(max_samples, len(ds))

            texts = []
            for ex in ds.select(range(max_samples)):
                msgs = _normalize_messages(ex.get("messages") or [])
                if msgs:
                    turns = []
                    for m in msgs:
                        role = (m.get("role") or "").strip()
                        content = _first_field_value(
                            m, ("content", "value", "text")
                        )
                        if content:
                            turns.append(f"{role}: {content}".strip())
                    if turns:
                        texts.append("\n".join(turns))
                        continue

                line = _first_field_value(
                    ex, ("text", "question", "prompt", "instruction", "input")
                )
                if line:
                    texts.append(line)

            if texts:
                return texts
            else:
                print(
                    "[Data][Indonesian][Warn] Local fallback returned 0 texts, "
                    "falling back to online mC4 streaming."
                )
        except Exception as e:
            print(
                f"[Data][Indonesian][Warn] Local indonesian-conversation fallback failed: {e}; "
                "falling back to online mC4 streaming."
            )

        ds = load_dataset(
            "allenai/c4",
            "id",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        streaming = True

    elif subset == "malay":
        try:
            print(
                f"[Data][Malay] Trying local mesolitica/Malaysian-SFT "
                f"cache_dir={cache_dir}"
            )
            ds = load_dataset(
                "mesolitica/Malaysian-SFT",
                split="google_translate_camel_ai",
                cache_dir=cache_dir,
            )
            streaming = False

            if max_samples is None or max_samples <= 0:
                max_samples = len(ds)
            max_samples = min(max_samples, len(ds))

            texts = []
            for ex in ds.select(range(max_samples)):
                src = _first_field_value(
                    ex,
                    ("prompt_input", "input", "instruction", "question", "query"),
                )
                tgt = _first_field_value(ex, ("output", "answer", "response"))
                if src and tgt:
                    combined = f"{src}\n{tgt}"
                else:
                    combined = src or tgt
                if combined:
                    texts.append(combined.strip())

            if texts:
                return texts
            else:
                print(
                    "[Data][Malay][Warn] Local Malaysian-SFT fallback returned 0 texts, "
                    "falling back to online mC4 streaming."
                )

        except Exception as e:
            print(
                f"[Data][Malay][Warn] Local Malaysian-SFT fallback failed: {e}; "
                "falling back to online mC4 streaming."
            )

        ds = load_dataset(
            "allenai/c4",
            "ms",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        streaming = True

    elif subset == "cantonese":
        ds = load_dataset(
            "jed351/cantonese-wikipedia",
            split=split,
            streaming=True,
        )
        streaming = True

    else:
        raise ValueError(f"Unknown subset: {subset}")

    if max_samples is None or max_samples <= 0:
        max_samples = 200000

    if streaming:
        texts = []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            if "text" in sample and sample["text"] is not None:
                texts.append(sample["text"])
        return texts

    num = min(max_samples, len(ds))
    texts = [x["text"] for x in ds.select(range(num)) if "text" in x]
    return texts


def build_indonesian_dataset(
    tokenizer: AutoTokenizer,
    subset: str = "indonesian",
    split: str = "train",
    max_samples: int = 2000,
    max_length: int = 2048,
) -> Dataset:
    """
    将通用文本语料转成 LLaMA 语言模型训练数据。
    支持：indonesian / malay / eng / cantonese 等纯文本预训练数据。
    """
    print(
        f"[Data][General-Text] Loading general texts: subset={subset}, "
        f"split={split}, max_samples={max_samples}"
    )
    texts = load_general_english_texts(
        subset=subset,
        split=split,
        max_samples=max_samples,
    )

    print(f"[Data][General-Text] Loaded {len(texts)} texts")

    raw_ds = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print(f"[Data][General-Text] Tokenizing dataset (subset={subset})...")
    tokenized_ds = raw_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing general texts ({subset})",
    )

    return tokenized_ds


def load_fineweb_thai_texts(
    split: str = "train",
    max_samples: Optional[int] = None,
    dataset_name: str = THAI_FINEWEB_DATASET_NAME,
    local_dataset_path: Optional[str] = None,
) -> List[dict]:
    """
    加载 Suraponn/thai_instruction_sft 或其他 Thai SFT 数据的结构化样本。
    返回的 dict 包括 instruction / answer / messages 等字段，按 dataset_name 兼容 legacy 同结构。
    
    Args:
        split: 数据集分割（train/val/test）
        max_samples: 最大样本数
        dataset_name: Hugging Face 数据集名称（如果使用远程加载）
        local_dataset_path: 本地数据集路径（如果提供，优先使用本地数据）
    """
    if load_dataset is None:
        raise ImportError("需要安装 datasets，请运行: pip install datasets")

    if max_samples is None or max_samples <= 0:
        max_samples = 200000

    # 如果未显式传入本地路径，尝试从环境变量读取（用于 huggingface-cli 预下载的缓存）
    if local_dataset_path is None:
        env_local = os.environ.get("FINEWEB_THAI_LOCAL_DATASET")
        if env_local:
            local_dataset_path = env_local
            print(f"[Data][FineWeb-Thai] Use env FINEWEB_THAI_LOCAL_DATASET={local_dataset_path}")

    # 优先使用本地数据集
    if local_dataset_path is not None and os.path.exists(local_dataset_path):
        print(
            f"[Data][FineWeb-Thai] Loading from local path: {local_dataset_path} "
            f"split={split} max_samples={max_samples}"
        )
        # 从本地路径加载数据集
        # 支持多种格式：目录路径（包含 dataset_dict.json）或直接的 parquet/json 文件
        if os.path.isdir(local_dataset_path):
            ds = load_dataset(
                local_dataset_path,
                split=split,
                streaming=False,  # 本地数据不需要流式加载
            )
        else:
            # 如果是文件路径，尝试作为数据集文件加载
            raise ValueError(f"Local dataset path must be a directory, got: {local_dataset_path}")
    else:
        if local_dataset_path is not None:
            print(f"[Warning] Local dataset path not found: {local_dataset_path}, falling back to remote loading")
        print(
            f"[Data][FineWeb-Thai] Streaming {dataset_name} "
            f"split={split} max_samples={max_samples}"
        )
        # 镜像站点已通过 HF_ENDPOINT 环境变量配置，load_dataset 会自动使用
        ds = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
        )

    examples: List[dict] = []
    
    # 如果是本地数据集，可能需要限制样本数
    if local_dataset_path is not None and os.path.exists(local_dataset_path):
        # 本地数据集，直接迭代
        for idx, ex in enumerate(ds):
            if max_samples is not None and idx >= max_samples:
                break
            instruction = _first_field_value(ex, INSTRUCTION_KEYS)
            answer = _first_field_value(ex, ANSWER_KEYS)
            messages = _normalize_messages(ex.get("messages") or ex.get("Messages"))

            if not instruction and not messages:
                text = (ex.get("text") or "").strip()
                if text:
                    instruction = text

            example = {
                "instruction": instruction,
                "answer": answer,
                "messages": messages,
            }
            examples.append(example)
    else:
        # 流式数据集
        for ex in ds:
            instruction = _first_field_value(ex, INSTRUCTION_KEYS)
            answer = _first_field_value(ex, ANSWER_KEYS)
            messages = _normalize_messages(ex.get("messages") or ex.get("Messages"))

            if not instruction and not messages:
                text = (ex.get("text") or "").strip()
                if text:
                    instruction = text

            example = {
                "instruction": instruction,
                "answer": answer,
                "messages": messages,
            }
            examples.append(example)
            if len(examples) >= max_samples:
                break

    print(f"[Data][FineWeb-Thai] Collected {len(examples)} structured samples.")
    return examples


def build_fineweb_thai_chat_texts(
    examples: List[dict],
    *,
    template: str = "llama3",
    system_prompt: Optional[str] = None,
    bos: str = "",
    eos: str = "",
    assistant_text: Optional[str] = None,
) -> List[str]:
    """
    将 raw Thai fineweb 语料包成聊天格式文本。
    """
    template_config = CHAT_TEMPLATE_CONFIGS.get(template, CHAT_TEMPLATE_CONFIGS["llama3"])
    system_prompt = system_prompt or template_config.get("system_prompt", "")
    assistant_text = assistant_text or template_config.get("assistant_text", "")

    formatted = []
    for ex in examples:
        inst = (ex.get("instruction") or "")
        ans = (ex.get("answer") or "")
        msgs = None

        if msgs:
            chat = build_chat_from_messages(
                msgs,
                system_prompt=system_prompt,
                assistant_fallback=assistant_text or ans,
                bos_token=bos,
                eos_token=eos,
            )
        else:
            user_text = inst or ""
            assistant_text_final = assistant_text or ans
            if not user_text and not assistant_text_final:
                continue
            chat = build_chat_template(
                system_prompt=system_prompt,
                user_text=user_text,
                assistant_text=assistant_text_final,
                bos_token=bos,
                eos_token=eos,
            )
        if chat:
            formatted.append(chat)
    return formatted


def build_fineweb_thai_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    dataset_name: Optional[str] = None,
    local_dataset_path: Optional[str] = None,
) -> Dataset:
    """
    将 FineWeb Thai 结构化样本（instruction/answer/messages）转成 LLaMA 语言模型训练数据。
    dataset_name 可选 legacy 的 ChavyvAkvar/fineweb-2-1M-Sample-Thai。
    
    Args:
        tokenizer: 分词器
        split: 数据集分割
        max_samples: 最大样本数
        max_length: 最大序列长度
        dataset_name: Hugging Face 数据集名称（如果使用远程加载）
        local_dataset_path: 本地数据集路径（如果提供，优先使用本地数据）
    """
    print(
        f"[Data][FineWeb-Thai] Building dataset split={split} "
        f"max_samples={max_samples} max_length={max_length}"
    )
    if local_dataset_path:
        print(f"[Data][FineWeb-Thai] Using local dataset: {local_dataset_path}")
    texts = load_fineweb_thai_texts(
        split=split,
        max_samples=max_samples,
        dataset_name=dataset_name or THAI_FINEWEB_DATASET_NAME,
        local_dataset_path=local_dataset_path,
    )
    chat_texts = build_fineweb_thai_chat_texts(
        texts,
        template="llama3",
        system_prompt=FINEWEB_SYSTEM_PROMPT,
        bos=tokenizer.bos_token or "",
        eos=tokenizer.eos_token or "",
    )
    raw_ds = Dataset.from_dict({"text": chat_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][FineWeb-Thai] Tokenizing dataset...")
    tokenized_ds = raw_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing FineWeb Thai dataset",
    )

    return tokenized_ds


# ==========================================
# izzulgod/indonesian-conversation 多轮 SFT（LLaMA3 chat）
# ==========================================


def build_indoconv_sft_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    使用 izzulgod/indonesian-conversation 做多轮聊天 SFT（LLaMA3 chat 格式）。

    数据格式:
      {
        "messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
        ]
      }
    """
    print(f"[Data][IndoConv] Loading izzulgod/indonesian-conversation split={split}")
    raw_ds = load_dataset("indonlp/indonlu", name="smsa", split="train")

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(f"[Data][IndoConv] Subsampling to first {max_samples} examples")
        raw_ds = raw_ds.select(range(max_samples))

    eos = tokenizer.eos_token or ""
    bos = tokenizer.bos_token or ""

    system_prompt = "You are a helpful, honest, and harmless AI assistant for Indonesian language conversations."

    def format_example(example):
        messages = example.get("messages") or []

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + system_prompt
            + "<|eot_id|>"
        )

        for i, msg in enumerate(messages):
            role = (msg.get("role") or "user").strip()
            content = (msg.get("content") or "").strip()

            if role not in ("user", "assistant"):
                role = "user"

            is_last = i == len(messages) - 1

            if role == "assistant" and is_last:
                text += (
                    f"<|start_header_id|>{role}<|end_header_id|>\n"
                    + content
                    + eos
                )
            else:
                text += (
                    f"<|start_header_id|>{role}<|end_header_id|>\n"
                    + content
                    + "<|eot_id|>"
                )

        return {"text": text}

    print("[Data][IndoConv] Formatting as LLaMA3 chat strings (multi-turn)...")
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting izzulgod/indonesian-conversation (LLaMA3 chat)",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][IndoConv] Tokenizing dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing izzulgod/indonesian-conversation",
    )

    return tokenized_ds


# ==========================================
# stvlynn/Cantonese-Dialogue 粤语对话数据集
# ==========================================


def build_cantonese_dialogue_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    使用 stvlynn/Cantonese-Dialogue 做指令微调（LLaMA3 chat 格式）。

    数据集列：
      - instruction: 指令/问题
      - input:       输入（可能为空）
      - output:      回答

    我们拼成 LLaMA3 chat 格式：
      <|start_header_id|>system<|end_header_id|>
      {system_prompt}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      {user_content}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>
      {output}{eos}
    """
    if load_dataset is None:
        raise ImportError(
            "需要安装 datasets 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    print(
        f"[Data][Cantonese-Dialogue] Loading {CANTONESE_DIALOGUE_DATASET_NAME} split={split}"
    )
    raw_ds = load_dataset(CANTONESE_DIALOGUE_DATASET_NAME, split=split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(
            f"[Data][Cantonese-Dialogue] Subsampling to first {max_samples} examples"
        )
        raw_ds = raw_ds.select(range(max_samples))

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    def format_example(example):
        instr = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        if not instr:
            instr = "请回答以下问题。"

        user_content = instr
        if inp:
            user_content = f"{instr}\n\nInput:\n{inp}"

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + CANTONESE_SYSTEM_PROMPT
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user_content
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + out
            + eos
        )
        return {"text": text}

    print(
        "[Data][Cantonese-Dialogue] Formatting as LLaMA3 chat strings..."
    )
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting Cantonese-Dialogue (LLaMA3 chat)",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Cantonese-Dialogue] Tokenizing dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing Cantonese-Dialogue dataset",
    )

    return tokenized_ds


# ==========================================
# indiejoseph/cantonese-cot 粤语思维链数据集
# ==========================================


def build_cantonese_cot_dataset(
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 2048,
) -> Dataset:
    """
    使用 indiejoseph/cantonese-cot 做指令微调（LLaMA3 chat 格式）。

    数据集列：
      - instruction: 指令/问题
      - input:       输入（通常为空或固定值）
      - output:      回答（包含思维链推理过程）

    我们拼成 LLaMA3 chat 格式：
      <|start_header_id|>system<|end_header_id|>
      {system_prompt}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      {user_content}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>
      {output}{eos}
    """
    if load_dataset is None:
        raise ImportError(
            "需要安装 datasets 包才能加载 Hugging Face 数据集：\n"
            "  pip install datasets"
        )

    print(
        f"[Data][Cantonese-CoT] Loading {CANTONESE_COT_DATASET_NAME} split={split}"
    )
    raw_ds = load_dataset(CANTONESE_COT_DATASET_NAME, split=split)

    if max_samples is not None and max_samples > 0:
        max_samples = min(max_samples, len(raw_ds))
        print(
            f"[Data][Cantonese-CoT] Subsampling to first {max_samples} examples"
        )
        raw_ds = raw_ds.select(range(max_samples))

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    def format_example(example):
        instr = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        out = (example.get("output") or "").strip()

        if not instr:
            instr = "请回答以下问题。"

        user_content = instr
        if inp:
            user_content = f"{instr}\n\nInput:\n{inp}"

        text = (
            bos
            + "<|start_header_id|>system<|end_header_id|>\n"
            + CANTONESE_SYSTEM_PROMPT
            + "<|eot_id|>"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user_content
            + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n"
            + out
            + eos
        )
        return {"text": text}

    print(
        "[Data][Cantonese-CoT] Formatting as LLaMA3 chat strings..."
    )
    ds_text = raw_ds.map(
        format_example,
        remove_columns=raw_ds.column_names,
        desc="Formatting cantonese-cot (LLaMA3 chat)",
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    print("[Data][Cantonese-CoT] Tokenizing dataset...")
    tokenized_ds = ds_text.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing cantonese-cot dataset",
    )

    return tokenized_ds

