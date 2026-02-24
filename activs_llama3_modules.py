# activs_llama3_modules.py
from typing import List, Dict, Tuple, Optional
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def get_llama3_block_activations_by_module(
    prompts: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B",  # 或 "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 4,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
    import json, os
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers import AutoModelForCausalLM

    config_path = os.path.join(model_name, "config.json")
    with open(config_path, "r") as f:
        raw_cfg = json.load(f)

    # 修补 rope_scaling 兼容旧版
    if isinstance(raw_cfg.get("rope_scaling"), dict) and "rope_type" in raw_cfg["rope_scaling"]:
        factor = float(raw_cfg["rope_scaling"].get("factor", 1.0))
        raw_cfg["rope_scaling"] = {"type": "dynamic", "factor": factor}

    # 旧 transformers 没有 AutoConfig.from_dict，用 LlamaConfig 替代
    if "model_type" not in raw_cfg:   # 必要，否则构造失败
        raw_cfg["model_type"] = "llama"

    cfg = LlamaConfig(**raw_cfg)


    # 🔧 兼容旧版 transformers — 手动降级 rope_scaling
    if isinstance(cfg.rope_scaling, dict) and "rope_type" in cfg.rope_scaling:
        factor = float(cfg.rope_scaling.get("factor", 1.0))
        cfg.rope_scaling = {"type": "dynamic", "factor": factor}   # <= 旧版能接受的格式

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",   # 建议先禁 flash-attn
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,    max_length=512,
    )


    # 层数
    try:
        L = model.config.num_hidden_layers
    except Exception:
        L = max(
            int(m.group(1))
            for n,_ in model.named_modules()
            for m in [re.search(r"\.layers\.(\d+)\.", n)]
            if m
        ) + 1

    # 准备缓冲：post 与 pre 分开存
    base_kinds = ["Q","K","V","O","fwd_up","fwd_down"]
    kinds = base_kinds + [k + "_pre" for k in base_kinds]
    bufs: Dict[str, List[List[torch.Tensor]]] = {k: [ [] for _ in range(L) ] for k in kinds}

    # token 掩码均值池化（[B,S,H] -> [B,H] -> 累到 [T,H]）
    def masked_mean(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            return h
        m = attn_mask.unsqueeze(-1).to(dtype=h.dtype)  # 不强制 device
        h_sum = (h * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return h_sum / denom


    current_mask = {"attn": None}
    hooks = []

    # Llama3 的模块名
    name2kind = [
        (r"\.self_attn\.q_proj$", "Q"),
        (r"\.self_attn\.k_proj$", "K"),
        (r"\.self_attn\.v_proj$", "V"),
        (r"\.self_attn\.o_proj$", "O"),
        (r"\.mlp\.up_proj$",      "fwd_up"),
        (r"\.mlp\.down_proj$",    "fwd_down"),
        # 如果也想采集 gate_proj，解除下面两行注释并把 base_kinds 里加上 "fwd_gate"
        # (r"\.mlp\.gate_proj$",   "fwd_gate"),
    ]

    def layer_idx_from_name(name: str) -> Optional[int]:
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def make_hook(kind: str, li: int):
        # 一个 forward_hook 同时拿 pre(in) / post(out)
        def _hook(module, inp, out):
            # inp: tuple，通常形状 [B,S,H]；out: [B,S,H’]
            x_pre = inp[0] if isinstance(inp, (tuple, list)) else inp
            x_post = out[0] if isinstance(out, (tuple, list)) else out
            pooled_pre  = masked_mean(x_pre,  current_mask["attn"])  # [B,H_in]
            pooled_post = masked_mean(x_post, current_mask["attn"])  # [B,H_out]
            bufs[kind + "_pre"][li].append(pooled_pre.detach().to("cpu"))
            bufs[kind][li].append(pooled_post.detach().to("cpu"))
        return _hook

    for name, mod in model.named_modules():
        li = layer_idx_from_name(name)
        if li is None or not (0 <= li < L):
            continue
        for pat, kind in name2kind:
            if re.search(pat, name):
                hooks.append(mod.register_forward_hook(make_hook(kind, li)))
                break

    from tqdm import tqdm

    oom_skip_count = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,    max_length=512)

        # 转成 Tensor 后再移动
        enc = {k: torch.tensor(v, device=device) if not torch.is_tensor(v) else v.to(device)
            for k, v in enc.items()}

        enc = {k: v.to(device) for k, v in enc.items()}  # 全部放到 GPU
        current_mask["attn"] = enc["attention_mask"]     # ✅ 保证已经在 GPU

        try:
            _ = model(**enc, use_cache=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for single_prompt in batch:
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
        print(f"\n[LLaMA3] 推理中因显存不足跳过 {oom_skip_count} 条样本")

    for h in hooks:
        h.remove()

    # 拼批得到 [T,H]
    per_layer: Dict[str, List[torch.Tensor]] = {}
    for kind in kinds:
        per_layer[kind] = [ torch.cat(bufs[kind][li], dim=0) if len(bufs[kind][li]) > 0 else torch.empty(0)
                            for li in range(L) ]

    per_module_concat: Dict[str, torch.Tensor] = {
        kind: torch.cat(per_layer[kind], dim=1)  # [T, sum over layers H_ℓ(kind)]
        for kind in kinds
    }
    print("\n[LLaMA3 Activation Stats]")
    for k in base_kinds:
        for li in range(L):
            x = per_layer[k][li]
            if torch.is_tensor(x) and x.numel() > 0:
                print(f"  {k:<8s} layer {li:<2d}: mean={x.mean():.6f}, std={x.std():.6f}, shape={tuple(x.shape)}")
                break
    return per_layer, per_module_concat


# activs_llama3_modules.py
from typing import List, Dict, Tuple, Optional
import re
import os
import sys
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def get_llama3_block_activations_by_module_neuron(
    prompts: List[str],
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 4,
    top_neuron_dir: Optional[str] = None,  # 保存 top-k neuron 索引的目录，如 "top_neurons_llama3"
    topk_per_layer: int = 128,  #pre 128             # ✅ 新增：每层选取的 top neuron 数
    max_length: int = 2048,  # ✅ 新增：最大序列长度（默认2048，用于包含完整的alpaca output）  #原始512
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
    import json

    from transformers.models.llama.configuration_llama import LlamaConfig

    config_path = os.path.join(model_name, "config.json")
    with open(config_path, "r") as f:
        raw_cfg = json.load(f)

    # 修补 rope_scaling 兼容旧版
    if isinstance(raw_cfg.get("rope_scaling"), dict) and "rope_type" in raw_cfg["rope_scaling"]:
        factor = float(raw_cfg["rope_scaling"].get("factor", 1.0))
        raw_cfg["rope_scaling"] = {"type": "dynamic", "factor": factor}

    if "model_type" not in raw_cfg:
        raw_cfg["model_type"] = "llama"

    cfg = LlamaConfig(**raw_cfg)

    if isinstance(cfg.rope_scaling, dict) and "rope_type" in cfg.rope_scaling:
        factor = float(cfg.rope_scaling.get("factor", 1.0))
        cfg.rope_scaling = {"type": "dynamic", "factor": factor}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        torch_dtype=torch.bfloat16 if dtype is None else dtype,
        attn_implementation="eager",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,    max_length=512,
    )
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 层数检测
    try:
        L = model.config.num_hidden_layers
    except Exception:
        L = max(
            int(m.group(1))
            for n, _ in model.named_modules()
            for m in [re.search(r"\.layers\.(\d+)\.", n)]
            if m
        ) + 1

    base_kinds = ["Q", "K", "V", "O", "fwd_up", "fwd_down"]
    kinds = base_kinds + [k + "_pre" for k in base_kinds]
    bufs: Dict[str, List[List[torch.Tensor]]] = {k: [[] for _ in range(L)] for k in kinds}

    # token 掩码均值池化
    def masked_mean(h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if h.dim() == 2:
            return h
        m = attn_mask.unsqueeze(-1).to(dtype=h.dtype)
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
        (r"\.mlp\.up_proj$", "fwd_up"),
        (r"\.mlp\.down_proj$", "fwd_down"),
        # (r"\.mlp\.gate_proj$", "fwd_gate"),
    ]

    def layer_idx_from_name(name: str) -> Optional[int]:
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def make_hook(kind: str, li: int):
        def _hook(module, inp, out):
            x_pre = inp[0] if isinstance(inp, (tuple, list)) else inp
            x_post = out[0] if isinstance(out, (tuple, list)) else out
            pooled_pre = masked_mean(x_pre, current_mask["attn"])
            pooled_post = masked_mean(x_post, current_mask["attn"])
            bufs[kind + "_pre"][li].append(pooled_pre.detach().to("cpu"))
            bufs[kind][li].append(pooled_post.detach().to("cpu"))
        return _hook

    for name, mod in model.named_modules():
        li = layer_idx_from_name(name)
        if li is None or not (0 <= li < L):
            continue
        for pat, kind in name2kind:
            if re.search(pat, name):
                hooks.append(mod.register_forward_hook(make_hook(kind, li)))
                break

    # 前向批处理（显存不足时逐条重试，仍失败则跳过并记录）
    from tqdm import tqdm

    oom_skip_count = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        current_mask["attn"] = enc["attention_mask"]
        try:
            _ = model(**enc, use_cache=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for single_prompt in batch:
                    try:
                        enc_single = tokenizer([single_prompt], return_tensors="pt", padding=True,
                                               truncation=True, max_length=max_length)
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
        print(f"\n[LLaMA3] 推理中因显存不足跳过 {oom_skip_count} 条样本")

    for h in hooks:
        h.remove()

    # 拼批
    per_layer: Dict[str, List[torch.Tensor]] = {}
    for kind in kinds:
        per_layer[kind] = [
            torch.cat(bufs[kind][li], dim=0) if len(bufs[kind][li]) > 0 else torch.empty(0)
            for li in range(L)
        ]

    per_module_concat: Dict[str, torch.Tensor] = {
        kind: torch.cat(per_layer[kind], dim=1) if len(per_layer[kind]) > 0 and any(t.numel() > 0 for t in per_layer[kind]) else torch.empty(0)
        for kind in kinds
    }

    # ✅ 新增部分：保存高激活 neuron
    if top_neuron_dir:
        try:
            os.makedirs(top_neuron_dir, exist_ok=True)
            if not os.path.exists(top_neuron_dir):
                raise OSError(f"无法创建目录: {top_neuron_dir}")
            if not os.path.isdir(top_neuron_dir):
                raise OSError(f"路径不是目录: {top_neuron_dir}")
            print(f"\n[LLaMA3] Saving top-{topk_per_layer} neurons per layer to {top_neuron_dir}")
        except Exception as e:
            print(f"[ERROR] 无法创建 top_neuron_dir {top_neuron_dir}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise
        
        top_neurons: Dict[str, Dict[int, torch.Tensor]] = {k: {} for k in kinds}
        saved_count = 0
        for kind in kinds:
            for li in range(L):
                acts = per_layer[kind][li]
                if torch.is_tensor(acts) and acts.numel() > 0:
                    strengths = acts.abs().mean(dim=0)  # [H]
                    H = strengths.numel()
                    k_eff = min(topk_per_layer, H)
                    top_idx = torch.topk(strengths, k=k_eff, largest=True, sorted=False).indices.to(torch.long)
                    top_idx = torch.unique(top_idx)
                    if top_idx.numel() == 0:
                        top_idx = torch.arange(min(k_eff, H), dtype=torch.long)
                    top_neurons[kind][li] = top_idx.cpu()
                    save_path = os.path.join(top_neuron_dir, f"top_neurons_{kind}_layer_{li}.pt")
                    try:
                        torch.save(top_neurons[kind][li], save_path)
                        # 验证文件是否成功保存
                        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                            saved_count += 1
                        else:
                            print(f"[WARN] 文件保存后验证失败: {save_path}", file=sys.stderr)
                    except Exception as e:
                        print(f"[ERROR] 保存 top neuron 文件失败 {save_path}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()

        print(f"[LLaMA3] Top neuron indices saved successfully. Total files: {saved_count}")

    # ✅ 输出激活统计
    print("\n[LLaMA3 Activation Stats]")
    for k in base_kinds:
        for li in range(L):
            x = per_layer[k][li]
            if torch.is_tensor(x) and x.numel() > 0:
                print(f"  {k:<8s} layer {li:<2d}: mean={x.mean():.6f}, std={x.std():.6f}, shape={tuple(x.shape)}")
                break

    return per_layer, per_module_concat
