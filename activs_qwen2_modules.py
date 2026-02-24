# activs_qwen2_modules.py
from typing import List, Dict, Tuple, Optional
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def get_qwen2_block_activations_by_module_neuron(
    prompts: List[str],
    model_name: str = "Qwen/Qwen2-7B-Instruct",  # 支持 Qwen2 和 Qwen2.5（如 Qwen/Qwen2.5-7B-Instruct）
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 4,
    top_neuron_dir: Optional[str] = None,  # 保存 top-k neuron 索引的目录，如 "top_neurons_qwen2"
    topk_per_layer: int = 128,  # ✅ 每层选取多少个 neuron
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
    """
    与 LLaMA3 版本一致的接口 & 行为：
      - 返回两份结果：
          per_layer: dict[str, List[Tensor]]
                     key ∈ {
                       "Q","K","V","O","fwd_up","fwd_down",
                       "Q_pre","K_pre","V_pre","O_pre","fwd_up_pre","fwd_down_pre"
                     }
                     per_layer[key][ℓ] 形状 [T, dim_ℓ(key)]
          per_module_concat: dict[str, Tensor]
                     把各层在特征维拼接 -> [T, sum_ℓ dim_ℓ(key)]
      - 额外：在 top_neuron_dir 下保存每层 top-k neuron 的索引：
          top_neurons_{kind}_layer_{li}.pt
      
    支持 Qwen2 和 Qwen2.5 系列模型（如 Qwen/Qwen2.5-7B-Instruct）
    """
    # 加载模型（Qwen2.5 与 Qwen2 架构相同，使用相同的加载方式）
    # 对大模型启用8bit量化以节省显存
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if dtype is None else dtype,
        "attn_implementation": "eager",
    }

    # 对Qwen2.5-32B启用量化以减少显存使用
    use_quantization = False
    # 启用量化，使用8bit量化以节省显存
    if "Qwen2.5-32B" in model_name:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )
            model_kwargs["torch_dtype"] = None  # 量化时不需要指定dtype
            model_kwargs["device_map"] = "auto"  # 使用accelerate自动设备映射，避免.to()调用
            use_quantization = True
            print(f"[Info] 启用8bit量化 for {model_name}")
        except ImportError:
            print(f"[Warning] BitsAndBytesConfig不可用，继续使用FP16")
    else:
        print(f"[Info] 使用FP16模式 for {model_name} (量化已禁用)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, **model_kwargs
    )

    # 只有非量化模型才调用.to(device)，量化模型使用device_map自动处理
    if not use_quantization:
        model = model.to(device)

    # 加载 tokenizer（Qwen2.5 使用相同的 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 层数
    try:
        L = model.config.num_hidden_layers
    except Exception:
        L = max(
            int(m.group(1))
            for n, _ in model.named_modules()
            for m in [re.search(r"\.layers\.(\d+)\.", n)]
            if m
        ) + 1

    # 准备缓冲：post 与 pre 分开存
    base_kinds = ["Q", "K", "V", "O", "fwd_up", "fwd_down"]
    kinds = base_kinds + [k + "_pre" for k in base_kinds]
    bufs: Dict[str, List[List[torch.Tensor]]] = {k: [[] for _ in range(L)] for k in kinds}

    # token 掩码均值池化（[B,S,H] -> [B,H]；极少见 [B,H] 直接返回）
    def masked_mean(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            return h
        m = attn_mask.unsqueeze(-1).to(dtype=h.dtype, device=h.device)
        h_sum = (h * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return h_sum / denom

    current_mask = {"attn": None}
    hooks = []

    name2kind = [
        (r"\.self_attn\.q_proj$", "Q"),
        (r"\.self_attn\.k_proj$", "K"),
        (r"\.self_attn\.v_proj$", "V"),
        (r"\.self_attn\.o_proj$", "O"),
        (r"\.mlp\.up_proj$",      "fwd_up"),
        (r"\.mlp\.down_proj$",    "fwd_down"),
    ]

    def layer_idx_from_name(name: str) -> Optional[int]:
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def make_hook(kind: str, li: int):
        # 一个 forward_hook 同时拿 pre(in) / post(out)
        def _hook(module, inp, out):
            x_pre = inp[0] if isinstance(inp, (tuple, list)) else inp
            x_post = out[0] if isinstance(out, (tuple, list)) else out

            pooled_pre  = masked_mean(x_pre,  current_mask["attn"])  # [B, H_in]
            pooled_post = masked_mean(x_post, current_mask["attn"])  # [B, H_out]

            bufs[kind + "_pre"][li].append(pooled_pre.detach().to("cpu"))
            bufs[kind][li].append(pooled_post.detach().to("cpu"))
        return _hook

    # 注册 hook
    for name, mod in model.named_modules():
        li = layer_idx_from_name(name)
        if li is None or not (0 <= li < L):
            continue
        for pat, kind in name2kind:
            if re.search(pat, name):
                hooks.append(mod.register_forward_hook(make_hook(kind, li)))
                break

    # 分批前向（显存不足时逐条重试，仍失败则跳过并记录）
    from tqdm import tqdm

    oom_skip_count = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Qwen2/Qwen2.5"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        current_mask["attn"] = enc["attention_mask"]
        try:
            _ = model(**enc, use_cache=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 逐条重试
                for j, single_prompt in enumerate(batch):
                    try:
                        enc_single = tokenizer([single_prompt], return_tensors="pt", padding=True,
                                               truncation=True, max_length=512)
                        enc_single = {k: v.to(device) for k, v in enc_single.items()}
                        current_mask["attn"] = enc_single["attention_mask"]
                        _ = model(**enc_single, use_cache=False)
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e2:
                        if "out of memory" in str(e2).lower() or "OutOfMemoryError" in type(e2).__name__:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            oom_skip_count += 1
                            continue
                        raise
            else:
                raise

    if oom_skip_count > 0:
        print(f"\n[Qwen2/Qwen2.5] 推理中因显存不足跳过 {oom_skip_count} 条样本")

    # 移除 hook
    for h in hooks:
        h.remove()

    # 拼批得到 [T,H]
    per_layer: Dict[str, List[torch.Tensor]] = {}
    for kind in kinds:
        per_layer[kind] = [
            torch.cat(bufs[kind][li], dim=0) if len(bufs[kind][li]) > 0 else torch.empty(0)
            for li in range(L)
        ]

    # 跨层在特征维拼接
    per_module_concat: Dict[str, torch.Tensor] = {
        kind: (
            torch.cat(per_layer[kind], dim=1)
            if len(per_layer[kind]) > 0 and any(t.numel() > 0 for t in per_layer[kind])
            else torch.empty(0)
        )
        for kind in kinds
    }

    # ✅ 新增：保存每层 top-k neuron 索引
    if top_neuron_dir:
        os.makedirs(top_neuron_dir, exist_ok=True)
        print(f"\n[Qwen2/Qwen2.5] Saving top-{topk_per_layer} neurons per layer to {top_neuron_dir}")
        top_neurons: Dict[str, Dict[int, torch.Tensor]] = {k: {} for k in kinds}

        for kind in kinds:
            for li in range(L):
                acts = per_layer[kind][li]  # [T, H_l] 或 empty
                if torch.is_tensor(acts) and acts.numel() > 0:
                    strengths = acts.abs().mean(dim=0)  # [H_l]
                    H = strengths.numel()
                    k_eff = min(topk_per_layer, H)
                    # 取绝对值平均激活最大的 k 个 neuron
                    top_idx = torch.topk(
                        strengths, k=k_eff, largest=True, sorted=False
                    ).indices.to(torch.long)
                    # 去重保险
                    top_idx = torch.unique(top_idx)
                    if top_idx.numel() == 0:
                        top_idx = torch.arange(min(k_eff, H), dtype=torch.long)

                    top_neurons[kind][li] = top_idx.cpu()

                    save_path = os.path.join(
                        top_neuron_dir, f"top_neurons_{kind}_layer_{li}.pt"
                    )
                    torch.save(top_neurons[kind][li], save_path)

        print("[Qwen2/Qwen2.5] Top neuron indices saved successfully.")

    # ✅ 打印简单激活统计（只看 post）
    print("\n[Qwen2/Qwen2.5 Activation Stats]")
    for k in base_kinds:
        for li in range(L):
            x = per_layer[k][li]
            if torch.is_tensor(x) and x.numel() > 0:
                print(f"  {k:<8s} layer {li:<2d}: mean={x.mean():.6f}, std={x.std():.6f}, shape={tuple(x.shape)}")
                break

    return per_layer, per_module_concat
