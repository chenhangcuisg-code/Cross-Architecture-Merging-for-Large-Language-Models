# -*- coding: utf-8 -*-

import os
import glob
import argparse
import shutil
import tempfile
from typing import Optional

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)

from safetensors.torch import load_file  # For loading *.safetensors

# ==========================================
# TinyLLaVA support (optional)
# ==========================================
from tinyllava.model import load_pretrained_model

try:
    # Official tinyllava library, typically provides load_pretrained_model entry point
    from tinyllava.model import (  # type: ignore
        load_pretrained_model as tinyllava_load_pretrained_model,
    )
except Exception:  # May not exist in environment tinyllava
    tinyllava_load_pretrained_model = None

# ==========================================
# LoRA related
# ==========================================
from peft import LoraConfig, get_peft_model

# ==========================================
# Import utilities from generate_hot_residual.py
# ==========================================
from generate_hot_residual import enable_hot_residual_for_model

# ==========================================
# Dataset / collator: all imported from dataset_hot_texts.py import
# ==========================================
from dataset_hot_texts import (
    build_sft_dataset_from_malaysian_sft,
    build_medical_llama3_sft_dataset,
    build_alpaca_dataset,
    load_geo3k_raw_dataset,
    build_geo3k_text_sft_dataset,
    QwenGeo3KCollator,
    load_onevision_clevr_raw_dataset,
    build_onevision_clevr_text_sft_dataset,
    QwenOneVisionCLEVRDataCollator,
    build_indonesian_dataset,
    build_indoconv_sft_dataset,
    build_finance_instruct_sft_dataset,
    build_finance_alpaca_sft_dataset,
    build_fineweb_thai_dataset,
    build_cantonese_dialogue_dataset,
    build_cantonese_cot_dataset,
)
from data_loading.dataset_gsm8k import load_gsm8k_texts
from datasets import Dataset


# ==========================================
# Transport-based residual: manually restore + freeze
# ==========================================


def _load_state_dict_from_dir(model_dir: str) -> dict:
    """
    Supports:
      - *.safetensors
      - pytorch_model*.bin
    """
    sft_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if sft_files:
        print(f"[HOT] Found safetensors files: {sft_files}")
        state_dict = {}
        for f in sft_files:
            print(f"[HOT] Loading {f}")
            sd_part = load_file(f)
            state_dict.update(sd_part)
        return state_dict

    bin_files = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
    if bin_files:
        if len(bin_files) > 1:
            raise RuntimeError(f"Multiple pytorch_model*.bin shards found: {bin_files}, please merge or adapt.")
        print(f"[HOT] Loading bin checkpoint: {bin_files[0]}")
        return torch.load(bin_files[0], map_location="cpu")

    raise FileNotFoundError(f"No *.safetensors or pytorch_model*.bin found in {model_dir}")


def load_qwen2vl_with_hot(model_dir: str):
    """
    Qwen2-VL + HOT
    """
    print("[Qwen2VL][HOT] Loading pretrained model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="cpu",
    )

    print("[Qwen2VL][HOT] Loading state_dict...")
    sd = _load_state_dict_from_dir(model_dir)

    base_sd = {}
    hot_items = []

    for k, v in sd.items():
        if "hot_residual_weight" in k or "hot_residual_bias" in k:
            hot_items.append((k, v))
        else:
            base_sd[k] = v

    print("[Qwen2VL][HOT] Loading base weights into model...")
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print("[Qwen2VL][HOT] missing:", missing)
    print("[Qwen2VL][HOT] unexpected:", unexpected)

    print("[Qwen2VL][HOT] Registering HOT buffers...")
    for name, tensor in hot_items:
        module = model
        parts = name.split(".")
        for p in parts[:-1]:
            if hasattr(module, p):
                module = getattr(module, p)
            else:
                print(f"[HOT][WARN] Can’t locate module for {name}")
                module = None
                break
        if module is None:
            continue
        module.register_buffer(parts[-1], tensor.clone())

    return model


def load_tinyllava_backbone(model_dir: str) -> nn.Module:
    """
    TinyLLaVA loading: only care about language backbone.
    """
    if tinyllava_load_pretrained_model is None:
        raise RuntimeError(
            "tinyllava library not installed, but specified model_type=tinyllava。"
            "Please install tinyllava first, or use model_type=llama/qwen2vl。"
        )

    print(f"[TinyLLaVA] Loading TinyLLaVA model from {model_dir} ...")
    res = tinyllava_load_pretrained_model(model_dir)

    if isinstance(res, nn.Module):
        model = res
    elif isinstance(res, (tuple, list)):
        model = None
        for x in res:
            if isinstance(x, nn.Module):
                model = x
                break
        if model is None:
            raise RuntimeError(
                "tinyllava.load_pretrained_model No nn.Module found in return result nn.Module，"
                "Please check if tinyllava version is compatible."
            )
    else:
        raise RuntimeError(
            f"tinyllava.load_pretrained_model Returned unsupported type: {type(res)}"
        )

    return model


def load_model_with_hot_residual_frozen(model_dir: str, model_type: str = "llama") -> nn.Module:
    """
    Generic HOT loading:
      1) Build empty model with config
      2) Load state_dict
      3) Regular weights load_state_dict
      4) hot_residual_* Register as buffer

    model_type:
      - "llama": Text CausalLM (LLaMA series, AutoModelForCausalLM）
      - "qwen2": Text CausalLM (Qwen2 and Qwen2.5 series, e.g., Qwen/Qwen2.5-7B-Instruct，AutoModelForCausalLM）
      - "qwen2vl": Qwen2VLForConditionalGeneration
    """
    print(f"[HOT] Loading config from {model_dir}")
    config = AutoConfig.from_pretrained(model_dir)

    print(f"[HOT] Building {model_type} model from config (empty weights)...")
    if model_type in ["llama", "qwen2"]:
        model = AutoModelForCausalLM.from_config(config)
    elif model_type == "qwen2vl":
        model = Qwen2VLForConditionalGeneration.from_config(config)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    print(f"[HOT] Loading state_dict from {model_dir}")
    state_dict = _load_state_dict_from_dir(model_dir)

    base_sd = {}
    hot_items = []
    for name, tensor in state_dict.items():
        if "hot_residual_weight" in name or "hot_residual_bias" in name:
            hot_items.append((name, tensor))
        else:
            base_sd[name] = tensor

    print(f"[HOT] Base weights: {len(base_sd)} tensors, HOT residual tensors: {len(hot_items)}")

    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    if missing:
        print(f"[HOT] Missing keys when loading base weights: {len(missing)} (often ok, e.g. lm_head)")
    if unexpected:
        print(f"[HOT] Unexpected keys in base_sd (should be rare): {unexpected}")

    num_reg_ok = 0
    num_reg_fail = 0
    for name, tensor in hot_items:
        parts = name.split(".")
        module = model
        for attr in parts[:-1]:
            if not hasattr(module, attr):
                module = None
                break
            module = getattr(module, attr)
        if module is None:
            print(f"[HOT][Warn] Cannot locate module for {name}, skip.")
            num_reg_fail += 1
            continue

        buf_name = parts[-1]

        if hasattr(module, buf_name):
            buf = getattr(module, buf_name)
            if isinstance(buf, torch.Tensor) and buf.shape == tensor.shape:
                with torch.no_grad():
                    buf.copy_(tensor)
            else:
                module.register_buffer(buf_name, tensor.clone())
        else:
            module.register_buffer(buf_name, tensor.clone())
        num_reg_ok += 1

    print(f"[HOT] Registered HOT buffers: ok={num_reg_ok}, failed={num_reg_fail}")

    return model


def debug_hot_for_qwen2vl(model):
    print("\n================= [Qwen2-VL HOT Debug] =================")

    try:
        test_layer = model.model.layers[0].self_attn.q_proj
    except Exception as e:
        print("[HOT][Debug] Failed to access q_proj:", e)
        return

    print("Layer type:", type(test_layer))

    has_w = hasattr(test_layer, "hot_residual_weight")
    has_b = hasattr(test_layer, "hot_residual_bias")
    patched = hasattr(test_layer, "_original_forward")

    print("Has hot_residual_weight =", has_w)
    print("Has hot_residual_bias   =", has_b)
    print("Forward patched         =", patched)

    if has_w:
        w = test_layer.hot_residual_weight
        print(
            "[HOT][Debug] residual_weight: dtype =",
            w.dtype,
            "max =",
            w.abs().max().item(),
            "mean =",
            w.abs().mean().item(),
        )

    if has_b:
        b = test_layer.hot_residual_bias
        print(
            "[HOT][Debug] residual_bias:   dtype =",
            b.dtype,
            "max =",
            b.abs().max().item(),
            "mean =",
            b.abs().mean().item(),
        )

    if not (has_w and patched):
        print("[HOT][Debug] HOT residual buffer or forward patch missing → HOT NOT active.")
        print("========================================================\n")
        return

    try:
        with torch.no_grad():
            x = torch.randn(
                1,
                test_layer.in_features,
                device=test_layer.weight.device,
                dtype=test_layer.weight.dtype,
            )

            y_base = test_layer._original_forward(x)
            y_hot = test_layer(x)

            diff = (y_hot - y_base).abs().sum().item()
            print("[HOT][Debug] Mean(|F_hot - F_base|) =", diff)

            if diff == 0:
                if w.abs().max().item() < 1e-6:
                    print("[HOT][Debug][WARN] HOT residual is ZERO → no effect.")
                else:
                    print("[HOT][Debug][WARN] residual exists but patch did NOT take effect.")
            else:
                print("[HOT][Debug] HOT residual is ACTIVE → forward changed.")

    except Exception as e:
        print("[HOT][Debug] forward test failed:", e)

    print("========================================================\n")


def load_model_without_hot(model_dir: str, model_type: str = "llama") -> nn.Module:
    """
    For ablation:Ignore hot_residual_*，Only loadregular weights。

    model_type:
      - "llama": Text CausalLM (LLaMA series)
      - "qwen2": Text CausalLM (Qwen2 and Qwen2.5 series, e.g., Qwen/Qwen2.5-7B-Instruct）
      - "qwen2vl": Qwen2-VL
      - "tinyllava": TinyLLaVA
    """
    print(f"[Ablation][No-HOT] Loading base (NO-HOT) model: type={model_type}, dir={model_dir}")

    if model_type in ["llama", "qwen2"]:
        print(f"[Ablation][No-HOT] Loading config from {model_dir}")
        config = AutoConfig.from_pretrained(model_dir)

        print(f"[Ablation][No-HOT] Building causal LM model from config ...")
        model = AutoModelForCausalLM.from_config(config)

        print(f"[Ablation][No-HOT] Loading state_dict but ignoring hot_residual_* ...")
        state_dict = _load_state_dict_from_dir(model_dir)

        base_sd = {
            name: tensor
            for name, tensor in state_dict.items()
            if not ("hot_residual_weight" in name or "hot_residual_bias" in name)
        }

        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        if missing:
            print(f"[Ablation][No-HOT] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Ablation][No-HOT] Unexpected keys: {unexpected}")

        return model

    elif model_type == "qwen2vl":
        print("[Ablation][No-HOT] Loading Qwen2VL without HOT ...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="cpu",
        )
        return model

    elif model_type == "tinyllava":
        print("[Ablation][No-HOT] Loading TinyLLaVA without HOT ...")

        from tinyllava.model import load_pretrained_model as tl_load_pretrained_model

        model, _, _, _ = tl_load_pretrained_model(
            model_dir,
            device_map="cpu",
        )

        print("[Ablation][No-HOT] TinyLLaVA loaded successfully (no HOT residual).")
        return model

    else:
        raise ValueError(f"[Ablation][No-HOT] Unsupported model_type={model_type}")


# ==========================================
# Neuron-select utilities
# ==========================================


def _load_top_indices_dir(top_neuron_dir: str, kinds: list[str], num_layers: int):
    top = {k: [] for k in kinds}
    for kind in kinds:
        for li in range(num_layers):
            path = os.path.join(top_neuron_dir, f"top_neurons_{kind}_layer_{li}.pt")
            if os.path.exists(path):
                idx = torch.load(path, map_location="cpu")
                idx = torch.as_tensor(idx, dtype=torch.long).view(-1)
                top[kind].append(idx)
            else:
                top[kind].append(torch.empty(0, dtype=torch.long))
    return top


def _parse_layer_and_kind_from_name(name: str):
    """
    Parse from Linear name layer_idx, kind
    """
    parts = name.split(".")

    layer_idx = None
    for i, p in enumerate(parts):
        if p in ("layers", "h", "blocks") and i + 1 < len(parts):
            try:
                li = int(parts[i + 1])
                layer_idx = li
                break
            except ValueError:
                continue

    if layer_idx is None:
        return None, None

    comp = parts[-1]
    if comp == "q_proj":
        kind = "Q"
    elif comp == "k_proj":
        kind = "K"
    elif comp == "v_proj":
        kind = "V"
    elif comp in ("o_proj", "out_proj"):
        kind = "O"
    else:
        kind = None

    return layer_idx, kind


# ==========================================
# Fold transport-based residual into weights，get clean model（supports neuron-select）
# ==========================================


@torch.no_grad()
def fold_hot_residual_into_weights(model: nn.Module, hot_neuron_dir: Optional[str] = None):
    """
    Merge HOT residual into weights；supports dense mode & neuron-select mode。
    """
    patched = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_original_forward"):
            patched.append((name, module))

    if not patched:
        print("[Fold] No HOT-patched Linear layers found, nothing to fold.")
        return

    # ---- dense mode： top_neuron_dir，full merge ----
    if hot_neuron_dir is None:
        fused_layers = 0
        for name, module in patched:
            W = module.weight
            b = module.bias
            in_features = W.shape[1]
            dev = W.device
            dt = W.dtype

            zeros = torch.zeros(1, in_features, device=dev, dtype=dt)
            y0 = module(zeros)
            b_eff = y0[0]

            eye = torch.eye(in_features, device=dev, dtype=dt)
            yI = module(eye)
            W_eff = yI.transpose(0, 1) - b_eff.unsqueeze(1)

            module.weight.data.copy_(W_eff)
            if b is not None:
                module.bias.data.copy_(b_eff)

            module.forward = module._original_forward
            delattr(module, "_original_forward")
            if hasattr(module, "hot_residual_weight"):
                delattr(module, "hot_residual_weight")
            if hasattr(module, "hot_residual_bias"):
                delattr(module, "hot_residual_bias")

            fused_layers += 1

        print(f"[Fold] HOT residual fused into {fused_layers} Linear layers (dense mode). Model is now plain HF weights.")
        return

    # ---- neuron-select mode：only merge top_neuron_dir selected neurons in ----
    print(f"[Fold] Neuron-select fold enabled, hot_neuron_dir = {hot_neuron_dir}")

    layer_ids = []
    name_to_layer_kind = {}
    for name, module in patched:
        li, kind = _parse_layer_and_kind_from_name(name)
        if li is not None and kind is not None:
            name_to_layer_kind[name] = (li, kind)
            layer_ids.append(li)

    if not layer_ids:
        print("[Fold][Warn] Could not parse any (layer, kind) from module names, fall back to dense fold.")
        return fold_hot_residual_into_weights(model, hot_neuron_dir=None)

    num_layers = max(layer_ids) + 1
    kinds = ["Q", "K", "V", "O"]
    top = _load_top_indices_dir(hot_neuron_dir, kinds=kinds, num_layers=num_layers)

    fused_layers = 0
    for name, module in patched:
        W_base = module.weight.data.clone()
        b_base = module.bias.data.clone() if module.bias is not None else None
        in_features = W_base.shape[1]
        dev = W_base.device
        dt = W_base.dtype

        zeros = torch.zeros(1, in_features, device=dev, dtype=dt)
        y0 = module(zeros)
        b_fused = y0[0]

        eye = torch.eye(in_features, device=dev, dtype=dt)
        yI = module(eye)
        W_fused = yI.transpose(0, 1) - b_fused.unsqueeze(1)

        if name not in name_to_layer_kind:
            W_eff = W_fused
            b_eff = b_fused
        else:
            layer_idx, kind = name_to_layer_kind[name]
            idx_list = top.get(kind, [])
            if not idx_list or layer_idx >= len(idx_list):
                W_eff = W_fused
                b_eff = b_fused
            else:
                idx = idx_list[layer_idx]
                if not isinstance(idx, torch.Tensor):
                    idx = torch.as_tensor(idx, dtype=torch.long)
                idx = idx.to(device=W_base.device)

                W_eff = W_base.clone()
                b_eff = b_base.clone() if b_base is not None else None

                if kind in ["Q", "K", "V"]:
                    valid = idx[(idx >= 0) & (idx < W_eff.size(0))]
                    if valid.numel() > 0:
                        W_eff[valid] = W_fused[valid]
                        if b_eff is not None:
                            b_eff[valid] = b_fused[valid]
                    print(
                        f"[Fold][NeuronSelect] {name}: kind={kind}, layer={layer_idx}, "
                        f"rows={valid.tolist() if valid.numel() > 0 else []}"
                    )
                elif kind == "O":
                    valid = idx[(idx >= 0) & (idx < W_eff.size(1))]
                    if valid.numel() > 0:
                        W_eff[:, valid] = W_fused[:, valid]
                    print(
                        f"[Fold][NeuronSelect] {name}: kind=O, layer={layer_idx}, "
                        f"cols={valid.tolist() if valid.numel() > 0 else []}"
                    )
                    # O layer usually does not modify bias
                else:
                    W_eff = W_fused
                    b_eff = b_fused

                if module.bias is None:
                    b_eff = b_fused

        module.weight.data.copy_(W_eff)
        if module.bias is not None and b_eff is not None:
            module.bias.data.copy_(b_eff)
        elif module.bias is not None and b_eff is None:
            module.bias.data.copy_(b_fused)

        module.forward = module._original_forward
        delattr(module, "_original_forward")
        if hasattr(module, "hot_residual_weight"):
            delattr(module, "hot_residual_weight")
        if hasattr(module, "hot_residual_bias"):
            delattr(module, "hot_residual_bias")

        fused_layers += 1

    print(
        f"[Fold] HOT residual fused into {fused_layers} Linear layers (neuron-select mode). "
        f"Model is now plain HF weights."
    )


# ==========================================
# freezeset
# ==========================================


def apply_freeze_strategy(model: nn.Module, freeze_strategy: str):
    """
    according tosetparameters requires_grad。

    Args:
        model: model
        freeze_strategy: "frozen_hot" / "frozen_base" / "none"
    """
    if freeze_strategy == "none":
        print("[Freeze] Strategy: none - parameterstraining")
        # sourceparameterstraining
        for param in model.parameters():
            param.requires_grad = True
        # Transport-based residual buffer needs to be parameter for training
        hot_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "hot_residual_weight"):
                    # checkwhether parameter
                    if hasattr(module, "hot_residual_weight_param"):
                        #  parameter，onlyset requires_grad
                        module.hot_residual_weight_param.requires_grad = True
                    else:
                        #  buffer 
                        hot_w = module.hot_residual_weight.clone().detach()
                        #  buffer（e.g.in）
                        if "hot_residual_weight" in module._buffers:
                            del module._buffers["hot_residual_weight"]
                        #  buffer is parameter
                        module.register_parameter("hot_residual_weight_param", nn.Parameter(hot_w))
                        # ，so forward function can access
                        setattr(module, "hot_residual_weight", module.hot_residual_weight_param)
                        #  requires_grad=True
                        module.hot_residual_weight_param.requires_grad = True
                    hot_count += 1
                if hasattr(module, "hot_residual_bias"):
                    if hasattr(module, "hot_residual_bias_param"):
                        module.hot_residual_bias_param.requires_grad = True
                    else:
                        #  buffer 
                        hot_b = module.hot_residual_bias.clone().detach()
                        #  buffer（e.g.in）
                        if "hot_residual_bias" in module._buffers:
                            del module._buffers["hot_residual_bias"]
                        #  buffer is parameter
                        module.register_parameter("hot_residual_bias_param", nn.Parameter(hot_b))
                        setattr(module, "hot_residual_bias", module.hot_residual_bias_param)
                        #  requires_grad=True
                        module.hot_residual_bias_param.requires_grad = True
        print(f"[Freeze] set {hot_count}  HOT residual weightsistraining")
        return

    elif freeze_strategy == "frozen_hot":
        print("[Freeze] Strategy: frozen_transport - transport-based residual frozen, only training source parameters")
        # Transport-based residual buffer does not participate in training (buffer, requires_grad=False)
        hot_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "hot_residual_weight"):
                    # e.g. parameter，need buffer
                    if hasattr(module, "hot_residual_weight_param"):
                        hot_w = module.hot_residual_weight_param.data.clone()
                        module.register_buffer("hot_residual_weight", hot_w)
                        delattr(module, "hot_residual_weight_param")
                    #  buffer and requires_grad=False（buffer  False）
                    if hasattr(module, "hot_residual_weight"):
                        module.hot_residual_weight.requires_grad = False
                    hot_count += 1
                if hasattr(module, "hot_residual_bias"):
                    if hasattr(module, "hot_residual_bias_param"):
                        hot_b = module.hot_residual_bias_param.data.clone()
                        module.register_buffer("hot_residual_bias", hot_b)
                        delattr(module, "hot_residual_bias_param")
                    if hasattr(module, "hot_residual_bias"):
                        module.hot_residual_bias.requires_grad = False
        # sourceparameterstraining
        for param in model.parameters():
            param.requires_grad = True
        print(f"[Freeze]  {hot_count}  HOT residual weights")
        return

    elif freeze_strategy == "frozen_base":
        print("[Freeze] Strategy: frozen_base - source parameters frozen, only training transport-based residual")
        
        # First, collect transport-based residual parameters
        hot_params = set()
        
        # Transport-based residual needs to be parameter for training
        hot_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "hot_residual_weight"):
                    #  buffer is parameter
                    if hasattr(module, "hot_residual_weight_param"):
                        #  parameter，onlyset requires_grad
                        hot_params.add(module.hot_residual_weight_param)
                        module.hot_residual_weight_param.requires_grad = True
                    else:
                        #  buffer （buffer ，can detach）
                        hot_w = module.hot_residual_weight.clone().detach()
                        #  buffer（e.g.in）
                        if "hot_residual_weight" in module._buffers:
                            del module._buffers["hot_residual_weight"]
                        # （e.g.in）
                        if hasattr(module, "hot_residual_weight"):
                            delattr(module, "hot_residual_weight")
                        # registeris parameter（will automatically add to _parameters ）
                        module.register_parameter("hot_residual_weight_param", nn.Parameter(hot_w, requires_grad=True))
                        # ，so forward function can access
                        setattr(module, "hot_residual_weight", module.hot_residual_weight_param)
                        hot_params.add(module.hot_residual_weight_param)
                        #  requires_grad=True
                        module.hot_residual_weight_param.requires_grad = True
                    hot_count += 1
                if hasattr(module, "hot_residual_bias"):
                    if hasattr(module, "hot_residual_bias_param"):
                        hot_params.add(module.hot_residual_bias_param)
                        module.hot_residual_bias_param.requires_grad = True
                    else:
                        #  buffer （buffer ，can detach）
                        hot_b = module.hot_residual_bias.clone().detach()
                        #  buffer（e.g.in）
                        if "hot_residual_bias" in module._buffers:
                            del module._buffers["hot_residual_bias"]
                        # （e.g.in）
                        if hasattr(module, "hot_residual_bias"):
                            delattr(module, "hot_residual_bias")
                        # registeris parameter
                        module.register_parameter("hot_residual_bias_param", nn.Parameter(hot_b, requires_grad=True))
                        # 
                        setattr(module, "hot_residual_bias", module.hot_residual_bias_param)
                        hot_params.add(module.hot_residual_bias_param)
                        #  requires_grad=True
                        module.hot_residual_bias_param.requires_grad = True
        
        # Freeze source parameters, only train transport-based residual
        base_param_count = 0
        for param in model.parameters():
            if param not in hot_params:
                param.requires_grad = False
                base_param_count += 1
        
        # Check transport-based residual parameter training status
        trainable_hot = sum(1 for p in hot_params if p.requires_grad)
        print(f"[Freeze] Frozen {base_param_count} source parameters, training {hot_count} transport-based residual weights ({trainable_hot} trainable)")
        
        # Check whether transport-based residual parameters are registered
        all_params = set(model.parameters())
        missing_params = hot_params - all_params
        if missing_params:
            print(f"[Freeze][WARN] {len(missing_params)} transport-based residual parameters not registered in model parameters")
        
        return

    else:
        raise ValueError(f"Unsupported freeze_strategy={freeze_strategy}")


# ==========================================
# Main training logic
# ==========================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train HOT-residual (or No-HOT) LLaMA / Qwen2 / Qwen2-VL / TinyLLaVA model on "
            "Malaysian-SFT, Geometry3K, Indonesian C4 or Indonesian conversation, "
            "LLaVA-OneVision CLEVR, and Shekswess medical_llama3_instruct_dataset."
        ),
    )

    # model & 
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="directory： HOT model（orregular HF model / TinyLLaVA model）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="trainingdirectory",
    )

    # modeltype
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=["llama", "qwen2", "qwen2vl", "tinyllava"],
        help=(
            "llama: Text CausalLM (LLaMA ）；"
            "qwen2: text Qwen/Qwen2 and Qwen2.5 series CausalLM（Qwen/Qwen2-7B-Instruct, Qwen/Qwen2.5-7B-Instruct ）；"
            "qwen2vl: Qwen/Qwen2-VL-*-Instruct ；"
            "tinyllava: TinyLLaVA "
        ),
    )

    # training
    parser.add_argument(
        "--training_scenario",
        type=str,
        default="hot",
        choices=["hot", "no_hot"],
        help=(
            "hot:  HOT training；"
            "no_hot: ，notUse HOT"
        ),
    )

    parser.add_argument(
        "--freeze_strategy",
        type=str,
        default="frozen_hot",
        choices=["frozen_hot", "frozen_base", "none"],
        help=(
            "frozen_hot:  HOT residual（onlytrainingsourceparameters）；"
            "frozen_base: sourceparameters（onlytraining HOT residual）；"
            "none: not（train both）"
        ),
    )

    parser.add_argument(
        "--hot_neuron_dir",
        type=str,
        default=None,
        help=(
            "set：usedirectoryin top_neurons_{Q,K,V,O}_layer_{i}.pt  HOT neuron-select fold。"
            "（Qwen2.5 can top_neurons_qwen2_5_* directory）"
        ),
    )

    parser.add_argument(
        "--save_untrained_folded",
        action="store_true",
        help="e.g.：intraining“ fold HOT butnottraining”model output_dir/ablation_untrained_hot_fused",
    )

    # type
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="malaysian_sft",
        choices=[
            "malaysian_sft",
            "geometry3k",
            "indonesian",
            "indonesian_conversation",
            "fineweb_thai",
            "onevision_clevr",
            "medical_llama3",  # NEW
            "alpaca",
            "finance",
            "finance_alpaca",
            "cantonese",
            "cantonese_dialogue",
            "cantonese_cot",
            "gsm8k",
        ],
        help=(
            "malaysian_sft: mesolitica/Malaysian-SFT text SFT；"
            "geometry3k: hiyouga/geometry3k；"
            "indonesian: generictext（e.g., Indonesian C4）；"
            "indonesian_conversation: izzulgod/indonesian-conversation multi-turn SFT；"
            "fineweb_thai: ChavyvAkvar/fineweb-2-1M-Sample-Thai ；"
            "onevision_clevr: mvp-lab/LLaVA-OneVision-1.5-Instruct-Data CLEVR in ；"
            "medical_llama3: Shekswess/medical_llama3_instruct_dataset ；"
            "alpaca: tatsu-lab/alpaca ；"
            "finance: Josephgflowers/Finance-Instruct-500k （system/user/assistant ）；"
            "finance_alpaca: gbharti/finance-alpaca （instruction/input/output ）；"
            "cantonese: jed351/cantonese-wikipedia texttraining；"
            "cantonese_dialogue: stvlynn/Cantonese-Dialogue ；"
            "cantonese_cot: indiejoseph/cantonese-cot ；"
            "gsm8k: openai/gsm8k "
        ),
    )

    parser.add_argument(
        "--finance_split",
        type=str,
        default="train",
        help="Finance  split（ train）。：Use finance_alpaca whenuseparameters。",
    )

    parser.add_argument(
        "--geo3k_split",
        type=str,
        default="train",
        help="Use hiyouga/geometry3k  split：train / validation / test",
    )

    # OneVision CLEVR parameters
    parser.add_argument(
        "--onevision_subset",
        type=str,
        default="sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
        help=(
            "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data ，e.g., CLEVR, CLEVR-Math ；"
            "in Data Studio in CLEVR 。"
        ),
    )
    parser.add_argument(
        "--onevision_split",
        type=str,
        default="train",
        help="LLaVA-OneVision  split ，generally train",
    )

    # Malaysian-SFT parameters
    parser.add_argument(
        "--malaysian_sft_split",
        type=str,
        default="google_translate_camel_ai",
        help=(
            "mesolitica/Malaysian-SFT  split ，e.g., google_translate_camel_ai / force_malay ；"
            "or special  random_all：from all splits in"
        ),
    )
    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=2000,
        help="；<=0 or None useentire split",
    )

    parser.add_argument(
        "--alpaca_split",
        type=str,
        default="train",
        help="tatsu-lab/alpaca  split ，e.g., train / test。",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help=" token （only）",
    )

    # Indonesian / generictextparameters
    parser.add_argument(
        "--indonesian_subset",
        type=str,
        default="indonesian",
        help=(
            "load_general_english_texts used subset ："
            "common/wiki/imdb/c4/indonesian/malay/cantonese"
        ),
    )
    parser.add_argument(
        "--indonesian_split",
        type=str,
        default="train",
        help="load_general_english_texts used split",
    )

    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training（ bf16）")

    # LoRA
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help=" LoRA ，Only train LoRA layer",
    )
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default=None,
        help="（ fineweb_thai ）。e.g.，firstusenotfrom Hugging Face 。",
    )
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help=(
            " Linear ，inlayer LoRA，"
            "e.g.: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj"
        ),
    )

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ========= （supportstraining） =========
    # intrainingin，uselocal_ranksetGPU
    # torchrunwill automatically setLOCAL_RANK
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        # setuseGPU
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    print(f"[Device] device={device} (local_rank={os.environ.get('LOCAL_RANK', '0')})")

    # ========= mixed precision（according to freeze_strategy ） =========
    if args.freeze_strategy == "frozen_base":
        # Only transport-based residual training, use FP32 
        print("[MixedPrecision] freeze_strategy=frozen_base -> mixed precisionand GradScaler， FP32 training。")
        os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
        use_fp16 = False
        use_bf16 = False
    else:
        # logic（You can adjust as needed）
        if args.model_type == "qwen2vl":
            # Qwen2-VL Generally recommended bf16，e.g.supports
            print("[MixedPrecision] model_type=qwen2vl -> Use bf16。")
            os.environ.pop("ACCELERATE_MIXED_PRECISION", None)
            use_fp16 = False
            use_bf16 = True
        else:
            if args.fp16:
                print("[MixedPrecision] Use fp16 mixed precisiontraining。")
                os.environ.pop("ACCELERATE_MIXED_PRECISION", None)
                use_fp16 = True
                use_bf16 = False
            else:
                print("[MixedPrecision] Do not usemixed precision（FP32）。")
                os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
                use_fp16 = False
                use_bf16 = False


    print(f"[Config] model_type = {args.model_type}")
    print(f"[Config] training_scenario = {args.training_scenario}")
    print(f"[Config] freeze_strategy = {args.freeze_strategy}")
    print(f"[Config] dataset_type = {args.dataset_type}")
    print(f"[Config] Malaysian-SFT split = {args.malaysian_sft_split}")
    print(f"[Config] Geo3K split = {args.geo3k_split}")
    print(f"[Config] OneVision subset = {args.onevision_subset}")
    print(f"[Config] OneVision split = {args.onevision_split}")
    print(f"[Config] max_samples_per_subset = {args.max_samples_per_subset}")
    print(f"[Config] alpaca_split = {args.alpaca_split}")
    print(f"[Config] use_lora = {args.use_lora}")
    print(f"[Config] indonesian_subset = {args.indonesian_subset}")
    print(f"[Config] indonesian_split = {args.indonesian_split}")
    print(f"[Config] hot_neuron_dir = {args.hot_neuron_dir}")

    # # check
    # if args.model_type in ["llama", "tinyllava", "qwen2"] and args.dataset_type not in [
    #     "malaysian_sft",
    #     "geometry3k",
    #     "indonesian",
    #     "indonesian_conversation",
    #     "onevision_clevr",
    #     "medical_llama3",  # 
    # ]:
    #     raise ValueError(
    #         "onlysupports LLaMA/TinyLLaVA/Qwen2 textmodel + "
    #         "malaysian_sft / geometry3k / indonesian / indonesian_conversation / "
    #         "onevision_clevr / medical_llama3 ，"
    #         "please --dataset_type isin",
    #     )
    if args.model_type == "qwen2vl" and args.dataset_type not in [
        "geometry3k",
        "onevision_clevr",
    ]:
        raise ValueError(
            "onlysupports Qwen2-VL + geometry3k / onevision_clevr ，"
            "please --dataset_type is geometry3k or onevision_clevr"
        )

    # ============= 1. loadingmodel =============

    if args.model_type == "llama":
        if args.training_scenario == "hot":
            print(f"[Model] Loading HOT LLaMA model from {args.model_dir}")
            model = load_model_with_hot_residual_frozen(args.model_dir, model_type="llama")
            print("[Model] Enabling HOT residual forward patch (language only)...")
            enable_hot_residual_for_model(model, use_language_model_only=True, alpha=args.alpha)
            for name, module in model.named_modules():
                if hasattr(module, "hot_residual_weight"):
                    w = module.hot_residual_weight
                    print(name, "hot_residual_weight: mean =", w.abs().mean().item())
        else:
            print(f"[Model][No-HOT] Loading LLaMA model WITHOUT HOT from {args.model_dir}")
            model = load_model_without_hot(args.model_dir, model_type="llama")

    elif args.model_type == "qwen2":
        # ✅ supports Qwen2 and Qwen2.5 textmodel
        if args.training_scenario == "hot":
            print(f"[Model] Loading HOT Qwen2/Qwen2.5 text model from {args.model_dir}")
            model = load_model_with_hot_residual_frozen(args.model_dir, model_type="qwen2")
            print("[Model] Enabling HOT residual forward patch (language only, Qwen2/Qwen2.5 text)...")
            enable_hot_residual_for_model(model, use_language_model_only=True, alpha=args.alpha)
            for name, module in model.named_modules():
                if hasattr(module, "hot_residual_weight"):
                    w = module.hot_residual_weight
                    print(name, "[Qwen2/Qwen2.5] hot_residual_weight: mean =", w.abs().mean().item())
        else:
            print(f"[Model][No-HOT] Loading Qwen2/Qwen2.5 text model WITHOUT HOT from {args.model_dir}")
            model = load_model_without_hot(args.model_dir, model_type="qwen2")

    elif args.model_type == "qwen2vl":
        if args.training_scenario == "hot":
            print("[Model] Loading Qwen2-VL with HOT...")
            model = load_qwen2vl_with_hot(args.model_dir)
            enable_hot_residual_for_model(model, use_language_model_only=False, alpha=args.alpha)
            debug_hot_for_qwen2vl(model)
            for name, module in model.named_modules():
                if hasattr(module, "hot_residual_weight"):
                    w = module.hot_residual_weight
                    print(name, "hot_residual_weight: mean =", w.abs().mean().item())
        else:
            print(f"[Model][No-HOT] Loading Qwen2-VL model WITHOUT HOT from {args.model_dir}")
            model = load_model_without_hot(args.model_dir, model_type="qwen2vl")

    elif args.model_type == "tinyllava":
        if args.training_scenario == "hot":
            print(f"[Model] Loading HOT TinyLLaVA model from {args.model_dir}")
            model = load_tinyllava_backbone(args.model_dir)
            print("[Model] Enabling HOT residual forward patch (language only, TinyLLaVA)...")
            enable_hot_residual_for_model(model, use_language_model_only=True, alpha=args.alpha)
            for name, module in model.named_modules():
                if hasattr(module, "hot_residual_weight"):
                    w = module.hot_residual_weight
                    print(name, "hot_residual_weight: mean =", w.abs().mean().item())
        else:
            print(f"[Model][No-HOT] Loading TinyLLaVA model WITHOUT HOT from {args.model_dir}")
            model = load_tinyllava_backbone(args.model_dir)

    else:
        raise ValueError(f"Unsupported model_type={args.model_type}")

    # =============  =============
    if args.training_scenario == "hot":
        apply_freeze_strategy(model, args.freeze_strategy)
    else:
        print("[Freeze] training_scenario=no_hot，set")

    # =============  B.1： " merge nottraining" model =============

    if args.training_scenario == "hot" and args.save_untrained_folded:
        print("[Ablation][B.1] Saving UNTRAINED but HOT-fused model (fold before training)...")

        if args.model_type == "llama":
            ablation_model = load_model_with_hot_residual_frozen(args.model_dir, model_type="llama")
            enable_hot_residual_for_model(ablation_model, use_language_model_only=True, alpha=args.alpha)
        elif args.model_type == "qwen2":
            ablation_model = load_model_with_hot_residual_frozen(args.model_dir, model_type="qwen2")
            enable_hot_residual_for_model(ablation_model, use_language_model_only=True, alpha=args.alpha)
        elif args.model_type == "qwen2vl":
            ablation_model = load_qwen2vl_with_hot(args.model_dir)
            enable_hot_residual_for_model(ablation_model, use_language_model_only=False, alpha=args.alpha)
        elif args.model_type == "tinyllava":
            ablation_model = load_tinyllava_backbone(args.model_dir)
            enable_hot_residual_for_model(ablation_model, use_language_model_only=True, alpha=args.alpha)
        else:
            raise ValueError(f"[Ablation][B.1] Unsupported model_type={args.model_type}")

        fold_hot_residual_into_weights(ablation_model, hot_neuron_dir=args.hot_neuron_dir)

        ablation_dir = os.path.join(args.output_dir, "ablation_untrained_hot_fused")
        
        # ：e.g. ablation directoryinand，
        # onlyintraining（directorynotinorisempty）when
        ablation_marker = os.path.join(ablation_dir, "_ABLATION_UNTRAINED_CREATED")
        if os.path.exists(ablation_dir) and os.path.exists(ablation_marker):
            print(f"[Ablation][B.1] Skipping: ablation_untrained_hot_fused already exists at {ablation_dir}")
            print(f"[Ablation][B.1] If you want to recreate it, delete the directory first.")
        else:
            import datetime
            os.makedirs(ablation_dir, exist_ok=True)
            print(f"[Ablation][B.1] Saving to {ablation_dir}")
            ablation_model.save_pretrained(ablation_dir)
            # ，trainingnottrainingmodel
            with open(ablation_marker, 'w') as f:
                f.write("This ablation model was created BEFORE training.\n")
                f.write(f"Created at: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Model dir: {args.model_dir}\n")
                f.write(f"Output dir: {args.output_dir}\n")
        if args.model_type in ["llama", "qwen2"]:
            tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
            tok.save_pretrained(ablation_dir)
        elif args.model_type == "tinyllava":
            _, tokenizer_save, _, _ = load_pretrained_model(
                args.model_dir,
                device_map="cpu",
            )
            tokenizer_save.save_pretrained(ablation_dir)
        else:  # qwen2vl
            proc = AutoProcessor.from_pretrained(args.model_dir)
            proc.save_pretrained(ablation_dir)
        del ablation_model
        print("[Ablation][B.1] Done.")

    # ============= 1.1 LoRA =============

    if args.use_lora:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        print(f"[LoRA] Enabling LoRA. target_modules = {target_modules}")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("[LoRA] Disabled.")

    # （training）
    model = model.to(device)

    tokenizer = None
    processor = None

    # ============= 1.2 tokenizer / processor =============

    if args.model_type == "tinyllava":
        print("[TinyLLaVA] Initializing tokenizer and image processor via TinyLLaVA utils...")

        model_tmp, tokenizer, image_processor, _ = load_pretrained_model(
            args.model_dir,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        tinyllava_image_processor = image_processor  # noqa: F841
        del model_tmp

    elif args.model_type == "qwen2vl":
        print("[Qwen2-VL] Initializing AutoProcessor & tokenizer...")
        processor = AutoProcessor.from_pretrained(args.model_dir)
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    else:
        # llama & qwen2/qwen2.5 textmodel
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Qwen2.5 supports apply_chat_template，but still use during trainingbuild LLaMA3 compatible
        # e.g.needUse Qwen2.5  chat template，caninbuildwhenuse

    # ============= 2. build =============

    max_samples = args.max_samples_per_subset
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    if args.dataset_type == "malaysian_sft":
        if tokenizer is None:
            raise RuntimeError("dataset_type=malaysian_sft need tokenizer")
        train_dataset = build_sft_dataset_from_malaysian_sft(
            tokenizer=tokenizer,
            split=args.malaysian_sft_split,
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "medical_llama3":
        if tokenizer is None:
            raise RuntimeError("dataset_type=medical_llama3 need tokenizer（llama/tinyllava/qwen2 textmodel）")

        train_dataset = build_medical_llama3_sft_dataset(
            tokenizer=tokenizer,
            split="train",  # only train one split
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "alpaca":
        if tokenizer is None:
            raise RuntimeError("dataset_type=alpaca need tokenizer")

        train_dataset = build_alpaca_dataset(
            tokenizer=tokenizer,
            split=args.alpaca_split,
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "indonesian":
        if tokenizer is None:
            raise RuntimeError("dataset_type=indonesian need tokenizer")

        effective_max = max_samples if max_samples is not None else 200000
        print(f"[Data][Indonesian] effective_max_samples = {effective_max}")

        train_dataset = build_indonesian_dataset(
            tokenizer=tokenizer,
            subset=args.indonesian_subset,
            split=args.indonesian_split,
            max_samples=effective_max,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "cantonese":
        if tokenizer is None:
            raise RuntimeError("dataset_type=cantonese need tokenizer")

        effective_max = max_samples if max_samples is not None else 200000
        print(f"[Data][Cantonese] Loading jed351/cantonese-wikipedia, effective_max_samples = {effective_max}")

        train_dataset = build_indonesian_dataset(
            tokenizer=tokenizer,
            subset="cantonese",
            split=args.indonesian_split,
            max_samples=effective_max,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "indonesian_conversation":
        if tokenizer is None:
            raise RuntimeError("dataset_type=indonesian_conversation need tokenizer（llama/tinyllava/qwen2 textmodel）")

        train_dataset = build_indoconv_sft_dataset(
            tokenizer=tokenizer,
            split="train",
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "finance":
        if tokenizer is None:
            raise RuntimeError("dataset_type=finance need tokenizer（llama/tinyllava/qwen2 textmodel）")

        print(
            f"[Data][Finance] Loading Finance-Instruct-500k split={args.finance_split}"
        )
        train_dataset = build_finance_instruct_sft_dataset(
            tokenizer=tokenizer,
            split=args.finance_split,
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "finance_alpaca":
        if tokenizer is None:
            raise RuntimeError("dataset_type=finance_alpaca need tokenizer（llama/tinyllava/qwen2 textmodel）")

        print(
            f"[Data][Finance-Alpaca] Loading gbharti/finance-alpaca split={args.finance_split}"
        )
        train_dataset = build_finance_alpaca_sft_dataset(
            tokenizer=tokenizer,
            split=args.finance_split,
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "fineweb_thai":
        if tokenizer is None:
            raise RuntimeError("dataset_type=fineweb_thai need tokenizer")

        train_dataset = build_fineweb_thai_dataset(
            tokenizer=tokenizer,
            split="train",
            max_samples=max_samples,
            max_length=args.block_size,
            local_dataset_path=getattr(args, "local_dataset_path", None),
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "geometry3k":
        if args.model_type == "qwen2vl":
            train_dataset = load_geo3k_raw_dataset(
                split=args.geo3k_split,
                max_samples=max_samples,
            )

            print(train_dataset)

            if processor is None:
                processor = AutoProcessor.from_pretrained(args.model_dir)

            data_collator = QwenGeo3KCollator(
                processor=processor,
                max_length=args.block_size,
            )
        else:
            if tokenizer is None:
                raise RuntimeError("llama/tinyllava/qwen2 + geometry3k need tokenizer")
            train_dataset = build_geo3k_text_sft_dataset(
                tokenizer=tokenizer,
                split=args.geo3k_split,
                max_samples=max_samples,
                max_length=args.block_size,
            )

            print(train_dataset)

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

    elif args.dataset_type == "onevision_clevr":
        # LLaVA-OneVision CLEVR
        if args.model_type == "qwen2vl":
            # ： + conversations
            train_dataset = load_onevision_clevr_raw_dataset(
                subset=args.onevision_subset,
                split=args.onevision_split,
                max_samples=max_samples,
            )

            print(train_dataset)

            if processor is None:
                processor = AutoProcessor.from_pretrained(args.model_dir)

            data_collator = QwenOneVisionCLEVRDataCollator(
                processor=processor,
                max_length=args.block_size,
            )
        else:
            # LLaMA / TinyLLaVA / Qwen2 text：Only use multi-turn text（discard images）
            if tokenizer is None:
                raise RuntimeError("llama/tinyllava/qwen2 + onevision_clevr need tokenizer")
            train_dataset = build_onevision_clevr_text_sft_dataset(
                tokenizer=tokenizer,
                subset=args.onevision_subset,
                split=args.onevision_split,
                max_samples=max_samples,
                max_length=args.block_size,
            )

            print(train_dataset)

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

    elif args.dataset_type == "cantonese_dialogue":
        if tokenizer is None:
            raise RuntimeError("dataset_type=cantonese_dialogue need tokenizer（llama/tinyllava/qwen2 textmodel）")

        train_dataset = build_cantonese_dialogue_dataset(
            tokenizer=tokenizer,
            split="train",
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "cantonese_cot":
        if tokenizer is None:
            raise RuntimeError("dataset_type=cantonese_cot need tokenizer（llama/tinyllava/qwen2 textmodel）")

        train_dataset = build_cantonese_cot_dataset(
            tokenizer=tokenizer,
            split="train",
            max_samples=max_samples,
            max_length=args.block_size,
        )

        print(train_dataset)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    elif args.dataset_type == "gsm8k":
        if tokenizer is None:
            raise RuntimeError("dataset_type=gsm8k need tokenizer")

        print(f"[Data][GSM8K] Loading GSM8K dataset, max_samples={max_samples}")
        texts = load_gsm8k_texts(
            subset="main",
            split="train",
            streaming=False,
            max_samples=max_samples,
        )
        
        print(f"[Data][GSM8K] Loaded {len(texts)} samples")
        
        # textis Dataset
        train_dataset = Dataset.from_dict({"text": texts})
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.block_size,
                padding=False,
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing GSM8K",
        )
        
        print(train_dataset)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

    else:
        raise ValueError(f"Unsupported dataset_type={args.dataset_type}")

    # ============= LLaMA/TinyLLaVA/Qwen2 Transport-Based Residual Debug =============

    if args.model_type in ["llama", "tinyllava", "qwen2"] and args.training_scenario == "hot":
        try:
            base_model = model
            if hasattr(base_model, "base_model"):
                base_model = base_model.base_model

            if hasattr(base_model, "model"):
                base_model = base_model.model

            if hasattr(base_model, "layers"):
                test_layer = base_model.layers[0].self_attn.q_proj
            else:
                raise AttributeError("base_model has no attribute 'layers'")

        except Exception as e:
            print(f"[Debug][Error] Cannot access layers[0].self_attn.q_proj: {e}")
            test_layer = None

        if isinstance(test_layer, nn.Linear):
            has_w = hasattr(test_layer, "hot_residual_weight")
            has_b = hasattr(test_layer, "hot_residual_bias")
            patched = hasattr(test_layer, "_original_forward")

            print("Has hot_residual_weight =", has_w)
            print("Has hot_residual_bias   =", has_b)
            print("Forward patched         =", patched)

            if has_w:
                buf = getattr(test_layer, "hot_residual_weight")
                print("hot_residual_weight.is_buffer =", not any(buf is p for p in model.parameters()))

            if has_w and patched:
                p = next(model.parameters())
                dev = p.device
                dt = p.dtype

                with torch.no_grad():
                    x = torch.randn(1, 8, test_layer.in_features, device=dev, dtype=dt)
                    o_base = test_layer._original_forward(x)
                    o_hot = test_layer(x)
                    diff = (o_hot - o_base).abs().mean().item()
                    print("Mean residual contribution (|F_hot - F_base|) =", diff)
                    if diff == 0.0:
                        print("[Debug][Warn] HOT residual seems inactive (diff=0).")
                    else:
                        print("[Debug] HOT residual is active in forward.")
            else:
                print("[Debug][Warn] HOT residual not fully set up on test_layer.")
        else:
            print("[Debug][Error] test_layer is not nn.Linear, skip HOT check.")
    else:
        print("[Debug] Skip LLaMA/TinyLLaVA/Qwen2-specific HOT debug check (not llama/tinyllava/qwen2 or not hot).")

    print("====================================================\n")

    # ============= 5. Trainer  =============

    # 兼容新版 transformers：eval_strategy、已移除 overwrite_output_dir/save_safetensors
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=False,
        eval_strategy="no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer if args.model_type != "qwen2vl" else None,
        data_collator=data_collator,
    )

    # ============= 6. training =============

    print("[Train] Starting training...")
    trainer.train()

    # ============= 7. After training: merge LoRA, fold transport-based residual =============

    model = trainer.model

    if args.use_lora:
        if hasattr(model, "merge_and_unload"):
            print("[LoRA] Merging LoRA weights into base model...")
            model = model.merge_and_unload()
        else:
            print("[LoRA][Warn] Model has no merge_and_unload(), skip LoRA merge.")
    else:
        print("[LoRA] use_lora = False, skip LoRA merge.")

    print("[Fold] Folding HOT residual into base weights before saving...")
    fold_hot_residual_into_weights(model, hot_neuron_dir=args.hot_neuron_dir)

    trainer.model = model

    print("[Train] Saving final fused model (plain HF, no transport-based residual buffers, no LoRA)...")
    # Note: not the ablation_untrained_transport_fused directory
    ablation_dir = os.path.join(args.output_dir, "ablation_untrained_hot_fused")
    ablation_backup_dir = None
    if os.path.exists(ablation_dir):
        ablation_marker = os.path.join(ablation_dir, "_ABLATION_UNTRAINED_CREATED")
        if os.path.exists(ablation_marker):
            print(f"[Protection] Detected ablation_untrained_hot_fused directory, creating backup...")
            # whenbackupdirectory，
            ablation_backup_dir = tempfile.mkdtemp(prefix="ablation_backup_")
            print(f"[Protection] Backing up ablation model to {ablation_backup_dir}")
            # onlybackupmodelweights，do not backup entire directory
            for f in os.listdir(ablation_dir):
                if f.endswith(('.bin', '.safetensors', '.json')) or f.startswith('config'):
                    src = os.path.join(ablation_dir, f)
                    dst = os.path.join(ablation_backup_dir, f)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
    
    trainer.save_model(args.output_dir)
    
    #  ablation directory（e.g.）
    if ablation_backup_dir and os.path.exists(ablation_backup_dir):
        ablation_model_files = [f for f in os.listdir(ablation_dir) 
                              if f.endswith(('.bin', '.safetensors')) and 'model' in f]
        if not ablation_model_files:
            print(f"[Protection] Warning: ablation_untrained_hot_fused model files may have been overwritten!")
            print(f"[Protection] Restoring from backup...")
            for f in os.listdir(ablation_backup_dir):
                src = os.path.join(ablation_backup_dir, f)
                dst = os.path.join(ablation_dir, f)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            print(f"[Protection] Restoration complete.")
        # backupdirectory
        shutil.rmtree(ablation_backup_dir, ignore_errors=True)

    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)
    if processor is not None:
        processor.save_pretrained(args.output_dir)

    print("[Done] Training finished.")


if __name__ == "__main__":
    main()
