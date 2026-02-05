from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Block handles (attention only; robust naming compatibility: self_attn/attention/attn)
# ============================================================


class BlockHandles:
    """
    Handle only the 4 linear layers for attention: Q/K/V/O
    Compatible with multiple container naming conventions:
      - Attention containers: self_attn / attention / attn / (fallback to current module)
      - Projection linear layers: q_proj/k_proj/v_proj/o_proj/out_proj
    """
    def __init__(self, block: nn.Module):
        sa = getattr(block, "self_attn", None)
        if sa is None:
            sa = getattr(block, "attention", None)
        if sa is None:
            sa = getattr(block, "attn", None)
        if sa is None:
            sa = block

        def pick_linear(obj, *names):
            if not isinstance(obj, nn.Module):
                return None
            for n in names:
                v = getattr(obj, n, None)
                if isinstance(v, nn.Linear):
                    return v
            return None

        self.map: Dict[str, Optional[nn.Linear]] = {
            "Q": pick_linear(sa, "q_proj"),
            "K": pick_linear(sa, "k_proj"),
            "V": pick_linear(sa, "v_proj"),
            "O": pick_linear(sa, "o_proj", "out_proj"),
        }

    def ok(self) -> bool:
        return all(isinstance(self.map[k], nn.Linear) for k in ["Q", "K", "V", "O"])


# =========================
# Utilities
# =========================

def _maybe_language_submodule(model: nn.Module, use_language_model_only: bool) -> nn.Module:
    """
    If only fusing language submodule, prioritize entering text backbone, e.g.:
      - Qwen2.5-VL / Qwen2-VL (multimodal):
          Qwen2_5_VLForConditionalGeneration -> model.language_model
          Qwen2_5_VLModel                  -> language_model
      - TinyLLaVA (multimodal):
          TinyLLaVAForConditionalGeneration -> language_model
      - Qwen2 / Qwen2.5 text models (text-only, non-multimodal): return model directly (no submodule extraction needed)
      - Others: language_model / model / transformer / llm / base_model
    """
    if not use_language_model_only:
        return model

    inner: nn.Module = model

    # 1) Common model.language_model structure on multimodal wrappers
    if hasattr(inner, "model"):
        maybe = getattr(inner, "model")
        if isinstance(maybe, nn.Module) and hasattr(maybe, "language_model"):
            lm = getattr(maybe, "language_model")
            if isinstance(lm, nn.Module):
                return lm
        if isinstance(maybe, nn.Module):
            inner = maybe

    # 2) Single-level structure: directly has language_model
    if hasattr(inner, "language_model"):
        lm = getattr(inner, "language_model")
        if isinstance(lm, nn.Module):
            return lm

    # 3) Fallback to common structures
    for name in ["language_model", "model", "transformer", "llm", "base_model"]:
        if hasattr(inner, name):
            sub = getattr(inner, name)
            if isinstance(sub, nn.Module):
                return sub

    return inner


def _collect_attn_blocks(model: nn.Module) -> List[BlockHandles]:
    """
    Traverse model and collect modules that look like attention containers
    (judged by presence of q/k/v/o), deduplicated by attention container.
    """
    blocks: List[BlockHandles] = []
    seen = set()
    for _, module in model.named_modules():
        sa = getattr(module, "self_attn", None) or getattr(module, "attention", None) or getattr(module, "attn", None) or module
        if not isinstance(sa, nn.Module):
            continue
        ok_any = sum(int(hasattr(sa, n) and isinstance(getattr(sa, n), nn.Linear)) for n in ["q_proj", "v_proj", "k_proj"])
        ok_o   = (hasattr(sa, "o_proj") and isinstance(getattr(sa, "o_proj"), nn.Linear)) or \
                 (hasattr(sa, "out_proj") and isinstance(getattr(sa, "out_proj"), nn.Linear))
        if ok_any >= 2 and ok_o:
            key = id(sa)
            if key in seen:
                continue
            bh = BlockHandles(module)
            if bh.ok():
                blocks.append(bh)
                seen.add(key)
    return blocks

def _load_tensor(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return torch.load(path, map_location="cpu")

def _find_first_param_device(module: nn.Module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    return torch.device("cpu")

def _get_dim_lists(blocksA: List[BlockHandles], blocksB: List[BlockHandles], comp: str):
    dA_out = [int(blocksA[i].map[comp].out_features) for i in range(len(blocksA))]  # type: ignore
    dB_out = [int(blocksB[j].map[comp].out_features) for j in range(len(blocksB))]  # type: ignore
    dA_in  = [int(blocksA[i].map[comp].in_features)  for i in range(len(blocksA))]  # type: ignore
    dB_in  = [int(blocksB[j].map[comp].in_features)  for j in range(len(blocksB))]  # type: ignore
    return dA_out, dB_out, dA_in, dB_in


# ============ Transport-Based Residual Utilities (for fixing transport parameters during training) ============

@torch.no_grad()
def _attach_hot_residual_to_linear(
    lin: nn.Linear,
    W_res: Optional[torch.Tensor],
    b_res: Optional[torch.Tensor],
    alpha: float = 0.01,  # Use alpha here, not scale
):
    if not isinstance(lin, nn.Linear):
        return

    if W_res is not None:
        W_res = W_res.to(lin.weight.device, dtype=lin.weight.dtype)
        if hasattr(lin, "hot_residual_weight"):
            lin.hot_residual_weight.copy_(W_res)
        else:
            lin.register_buffer("hot_residual_weight", W_res)

    if b_res is not None:
        device = lin.bias.device if lin.bias is not None else lin.weight.device
        dtype = lin.bias.dtype if lin.bias is not None else lin.weight.dtype
        b_res = b_res.to(device, dtype=dtype)
        if hasattr(lin, "hot_residual_bias"):
            lin.hot_residual_bias.copy_(b_res)
        else:
            lin.register_buffer("hot_residual_bias", b_res)

    # Store alpha as buffer so it's also "fixed" during training
    alpha_t = torch.tensor(float(alpha), dtype=torch.float32)
    if hasattr(lin, "hot_residual_alpha"):
        lin.hot_residual_alpha.copy_(alpha_t)
    else:
        lin.register_buffer("hot_residual_alpha", alpha_t)

def _patch_linear_for_hot_residual(linear: nn.Linear, _alpha):
    """
    W_eff = (1 - α) * W_base + α * W_hot
    b_eff = (1 - α) * b_base + α * b_hot
    where W_hot / b_hot / α are all buffers and do not participate in training.
    """
    if hasattr(linear, "_original_forward"):
        return

    linear._original_forward = linear.forward

    def forward_with_hot(self, x, *args, **kwargs):
        has_W = hasattr(self, "hot_residual_weight")
        has_b = hasattr(self, "hot_residual_bias")
        if not (has_W or has_b):
            return self._original_forward(x, *args, **kwargs)

        # Default to 1.0 when hot_residual_alpha is not present
        alpha = float(_alpha)

        W_base = self.weight
        b_base = self.bias

        # ---- Weight soft merge ----
        if has_W:
            W_hot = self.hot_residual_weight
            if W_hot.device != W_base.device or W_hot.dtype != W_base.dtype:
                W_hot = W_hot.to(device=W_base.device, dtype=W_base.dtype)
            W_eff = (1.0 - alpha) * W_base + alpha * W_hot
        else:
            W_eff = W_base

        # ---- Bias soft merge ----
        if has_b:
            b_hot = self.hot_residual_bias
            if b_hot.device != W_base.device or b_hot.dtype != W_base.dtype:
                b_hot = b_hot.to(device=W_base.device, dtype=W_base.dtype)
            if b_base is None:
                b_eff = alpha * b_hot
            else:
                b_eff = (1.0 - alpha) * b_base + alpha * b_hot
        else:
            b_eff = b_base

        return torch.nn.functional.linear(x, W_eff, b_eff)

    linear.forward = forward_with_hot.__get__(linear, nn.Linear)


# ============================================================
# Transport-Based Residual Debug
# ============================================================
@torch.no_grad()
def debug_hot(linear: nn.Linear):
    if not hasattr(linear, "_original_forward"):
        print("[Transport Debug] linear not patched, skipping")
        return

    x = torch.randn(2, 3, linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype)

    y_base = linear._original_forward(x)
    y_hot  = linear.forward(x)

    diff = (y_hot - y_base).abs().mean().item()
    # print(f"[Transport Debug] Mean(|y_transport - y_base|) = {diff:.6f}")

def enable_hot_residual_for_model(model: nn.Module, use_language_model_only: bool = True, alpha: float = True):
    base = _maybe_language_submodule(model, use_language_model_only)
    blocks = _collect_attn_blocks(base)
    for bh in blocks:
        for comp in ["Q", "K", "V", "O"]:
            lin = bh.map.get(comp)
            if isinstance(lin, nn.Linear) and (
                hasattr(lin, "hot_residual_weight") or hasattr(lin, "hot_residual_bias")
            ):
                _patch_linear_for_hot_residual(lin, alpha)
                debug_hot(lin)



# ============ Sparsification and Normalization ============

@torch.no_grad()
def _row_renorm_keep_sum(P: torch.Tensor, old_sum: torch.Tensor) -> torch.Tensor:
    s = P.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return P * (old_sum / s)

@torch.no_grad()
def sparsify_P_topk(P: torch.Tensor, k: int = 3, threshold: float = 0.0, keep_row_mass: bool = True) -> torch.Tensor:
    if k is None or k <= 0:
        return P
    P2 = P.clone()
    if threshold > 0.0:
        P2 = torch.where(P2 >= threshold, P2, torch.zeros_like(P2))
    old_sum = P2.sum(dim=1, keepdim=True)
    k_eff = min(k, P2.size(1))
    _, idx = torch.topk(P2, k=k_eff, dim=1)
    mask = torch.zeros_like(P2, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    P2 = torch.where(mask, P2, torch.zeros_like(P2))
    if keep_row_mass:
        P2 = _row_renorm_keep_sum(P2, old_sum.clamp_min(1e-12))
    else:
        P2 = P2 / P2.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return P2

@torch.no_grad()
def _sinkhorn_to_targets(M: torch.Tensor, row_target: float, col_target: float, iters: int = 5) -> torch.Tensor:
    X = M.clamp_min(0).clone()
    rt = torch.tensor(row_target, dtype=X.dtype, device=X.device)
    ct = torch.tensor(col_target, dtype=X.dtype, device=X.device)
    for _ in range(max(1, iters)):
        rs = X.sum(dim=1, keepdim=True).clamp_min(1e-12)
        X = X * (rt / rs)
        cs = X.sum(dim=0, keepdim=True).clamp_min(1e-12)
        X = X * (ct / cs)
    return X

@torch.no_grad()
def sparsify_Q_topr(
    Q: torch.Tensor,
    r_row: int = 128,
    r_col: int = 128,
    threshold: float = 0.0,
    row_target: float = None,
    col_target: float = None,
    sinkhorn_iters: int = 5,
) -> torch.Tensor:
    qa = Q.clone().clamp_min(0)
    nA, nB = qa.shape
    if threshold > 0.0:
        qa = torch.where(qa >= threshold, qa, torch.zeros_like(qa))
    if r_row is not None and r_row > 0 and r_row < nB:
        _, idx = torch.topk(qa, k=r_row, dim=1)
        row_mask = torch.zeros_like(qa, dtype=torch.bool)
        row_mask.scatter_(1, idx, True)
        qa = torch.where(row_mask, qa, torch.zeros_like(qa))
    if r_col is not None and r_col > 0 and r_col < nA:
        _, idx = torch.topk(qa.t(), k=r_col, dim=1)
        col_mask = torch.zeros_like(qa.t(), dtype=torch.bool)
        col_mask.scatter_(1, idx, True)
        qa = torch.where(col_mask.t(), qa, torch.zeros_like(qa))
    if qa.sum() == 0:
        return Q
    if (row_target is not None) and (col_target is not None):
        qa = _sinkhorn_to_targets(qa, row_target=row_target, col_target=col_target, iters=sinkhorn_iters)
    return qa


# ============ Load Transport Plans (supports kept_layer_idx & pre/post) ============

def _init_grid(L_full: int, M_full: int) -> List[List[Optional[torch.Tensor]]]:
    return [[None for _ in range(M_full)] for _ in range(L_full)]

def _assign_q_checked(
    grid: List[List[Optional[torch.Tensor]]],
    i_full: int, j_full: int,
    q: torch.Tensor,
    dA_expect: int, dB_expect: int,
    name: str
):
    q = torch.as_tensor(q)
    if q.shape == (dA_expect, dB_expect):
        grid[i_full][j_full] = q
    elif q.shape == (dB_expect, dA_expect):
        grid[i_full][j_full] = q.t()
    else:
        raise ValueError(f"{name}[{i_full},{j_full}] bad shape {tuple(q.shape)}, expected {(dA_expect,dB_expect)} or {(dB_expect,dA_expect)}")

def _find_hot_file(hot_dir: str, fname_stem: str) -> str:
    print(f"[DEBUG _find_hot_file] hot_dir={hot_dir}, fname_stem={fname_stem}")
    candidates = [
        os.path.join(hot_dir, f"{fname_stem}.pt"),
        os.path.join(hot_dir, f"{fname_stem.upper()}.pt"),
    ]
    print("[DEBUG _find_hot_file] candidates=", candidates)
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Missing transport plan file: {fname_stem}(.pt)")
    print("[DEBUG _find_hot_file] FOUND:", path)
    return path


def _load_hot_bundle_expand(
    hot_dir: str,
    fname_stem: str,
    L_full: int, M_full: int,
    dA_expect_list: List[int],
    dB_expect_list: List[int],
) -> Tuple[torch.Tensor, List[List[Optional[torch.Tensor]]], int, List[int], List[int], str]:
    path = _find_hot_file(hot_dir, fname_stem)
    raw = _load_tensor(path)
    
    if not (isinstance(raw, dict) and ("P" in raw) and ("Q_list" in raw)):
        raise ValueError(f"{fname_stem}: unexpected file format; expect dict with P & Q_list")

    P_small: torch.Tensor = torch.as_tensor(raw["P"], dtype=torch.float32)
    L_eff = int(raw.get("L", P_small.shape[0]))
    M_eff = int(raw.get("M", P_small.shape[1]))

    keptA: List[int] = list(map(int, raw.get("kept_layer_idx", list(range(L_eff)))))
    keptB: List[int] = list(map(int, raw.get("kept_layer_idx_B", keptA)))

    if len(keptA) != L_eff or len(keptB) != M_eff:
        raise ValueError(f"{fname_stem}: kept_layer_idx length mismatch with L/M; got {len(keptA)}/{len(keptB)} vs {L_eff}/{M_eff}")

    P_full = torch.zeros((L_full, M_full), dtype=torch.float32)
    for i_small, i_full in enumerate(keptA):
        for j_small, j_full in enumerate(keptB):
            P_full[i_full, j_full] = P_small[i_small, j_small]

    Qgrid_full = _init_grid(L_full, M_full)

    Q_list_small = raw["Q_list"]
    if not (isinstance(Q_list_small, (list, tuple)) and len(Q_list_small) == L_eff):
        raise ValueError(f"{fname_stem}: Q_list outer len {len(Q_list_small)} != L_eff {L_eff}")
    for i_small in range(L_eff):
        row = Q_list_small[i_small]
        if not (isinstance(row, (list, tuple)) and len(row) == M_eff):
            raise ValueError(f"{fname_stem}: Q_list[{i_small}] len {len(row)} != M_eff {M_eff}")
        i_full = keptA[i_small]
        for j_small in range(M_eff):
            q = row[j_small]
            if q is None:
                continue
            j_full = keptB[j_small]
            dA_expect = dA_expect_list[i_full]
            dB_expect = dB_expect_list[j_full]
            _assign_q_checked(Qgrid_full, i_full, j_full, torch.as_tensor(q), dA_expect, dB_expect, fname_stem)

    # ========== Debug: print RAW basic info ==========
    print(f"\n[HOT DEBUG RAW] loading {fname_stem} from: {path}")
    print(f"[HOT DEBUG RAW] P_small.shape = {P_small.shape}")
    print(f"[HOT DEBUG RAW] Q_list_small outer = {len(Q_list_small)}")
    print(f"[HOT DEBUG RAW] Q_list_small[0] inner = {len(Q_list_small[0])}")

    print("[HOT DEBUG RAW] P_small stats:",
          "min=", float(P_small.min().item()),
          "max=", float(P_small.max().item()),
          "mean=", float(P_small.mean().item()))

    print("[HOT DEBUG RAW] keptA:", keptA)
    print("[HOT DEBUG RAW] keptB:", keptB)

    print("[HOT DEBUG RAW] Checking first few Q entries...")
    for i_small in range(min(3, L_eff)):
        for j_small in range(min(3, M_eff)):
            q = raw["Q_list"][i_small][j_small]
            if q is None:
                print(f"  Q[{i_small},{j_small}] = None")
            else:
                qt = torch.as_tensor(q)
                print(f"  Q[{i_small},{j_small}] shape={tuple(qt.shape)}, ",
                      f"mean={qt.abs().mean().item():.3e}, max={qt.abs().max().item():.3e}")

    return P_full, Qgrid_full, L_eff, keptA, keptB, path


# =========================
# Low-memory utilities: chunked triple matrix multiplication
# =========================

@torch.no_grad()
def _best_chunk_elems(device: torch.device, dtype: torch.dtype, target_mem_mb: int = 800, safety_mb: int = 128) -> int:
    # Logic originally uses 2/4 byte estimation, can be adjusted as needed
    bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    usable_mb = max(128, target_mem_mb - safety_mb)
    elems = (usable_mb * 1024 * 1024) // bytes_per
    return int(elems * 0.6)

@torch.no_grad()
def _chunked_mm(A: torch.Tensor, B: torch.Tensor, out_dtype: torch.dtype = None, max_elems: int = 5_000_000) -> torch.Tensor:
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"_chunked_mm: A={A.shape}, B={B.shape}"
    out = torch.zeros((m, n), device=A.device, dtype=(out_dtype or A.dtype))
    if n == 0:
        return out
    c = max(1, int((max_elems - m * k) // max(1, (k + m))))
    c = min(c, n)
    for c0 in range(0, n, c):
        c1 = min(c0 + c, n)
        out[:, c0:c1] = A @ B[:, c0:c1]
        if A.is_cuda:
            torch.cuda.empty_cache()
    return out

@torch.no_grad()
def _triple_product_chunked(
    Qout: torch.Tensor,   # [A_out, B_out]
    Wb: torch.Tensor,     # [B_out, B_in]
    Qin_t: torch.Tensor,  # [B_in, A_in]
    target_device: torch.device,
    use_autocast: bool = True,
    max_mem_mb: int = 1200,
) -> torch.Tensor:
    A_out, B_out = Qout.shape
    B_out2, B_in = Wb.shape
    B_in2, A_in = Qin_t.shape
    assert B_out == B_out2 and B_in == B_in2, f"shape mismatch: {Qout.shape}, {Wb.shape}, {Qin_t.shape}"
    dev = target_device
    dtype = Qout.dtype

    max_elems = _best_chunk_elems(dev, dtype, target_mem_mb=max_mem_mb)
    out = torch.zeros((A_out, A_in), device=dev, dtype=dtype)

    use_autocast_final = (use_autocast and dev.type == "cuda")
    autocast_dtype = torch.float32  # Can be changed to bfloat16 as needed

    with torch.cuda.amp.autocast(enabled=use_autocast_final, dtype=autocast_dtype):
        if A_out == 0 or B_in == 0:
            c_bo = B_out
        else:
            c_bo = max(1, int(max_elems // max(1, (A_out + B_in + (A_out * B_in) // max(1, A_out)))))
            c_bo = min(c_bo, B_out)

        for s0 in range(0, B_out, c_bo):
            s1 = min(s0 + c_bo, B_out)
            Q_chunk = Qout[:, s0:s1].to(dev, non_blocking=True)
            W_chunk = Wb[s0:s1, :].to(dev, non_blocking=True)

            mid = Q_chunk @ W_chunk             # (A_out, B_in)
            del Q_chunk, W_chunk
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            part = _chunked_mm(mid, Qin_t.to(dev, non_blocking=True), out_dtype=dtype, max_elems=max_elems)  # (A_out, A_in)
            out.add_(part)
            del mid, part
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


# =========================
# Top Neuron index loader (shared by HOT residual selection / TopReplace)
# =========================

def _load_top_indices_dir(top_neuron_dir: str, kinds: list[str], num_layers: int):
    """
    Load neuron indices for each layer and kind from directory:
      Filename: top_neurons_{kind}_layer_{li}.pt
      Content: 1D LongTensor
    """
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


# =========================
# Fusion (attention-only, low-memory support)
# =========================

@torch.no_grad()
def fuse_attention_only_from_hot_dir(
    modelA: nn.Module,
    modelB: nn.Module,
    hot_dir: str,
    alpha: float = 0.35,
    use_language_model_only: bool = True,
    verbose: bool = False,
    # --- Sparsification parameters ---
    p_topk: int = 3,
    p_threshold: float = 0.0,
    q_topr_row: int = 128,
    q_topr_col: int = 128,
    q_threshold: float = 0.0,
    q_sinkhorn_iters: int = 5,
    # --- Low-memory Attention control ---
    attn_device: str = "cuda:0",
    attn_max_mem_mb: int = 1200,
    attn_autocast: bool = True,
    # --- HOT usage mode ---
    use_hot_as_residual: bool = False,
    hot_residual_scale: float = 1.0,
    # --- Neuron Select: apply HOT residual/fusion only to key neurons ---
    hot_neuron_select_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For Q/K/V/O:
      Default (use_hot_as_residual=False):
        W_A ← (1-α)·W_A + α · L_eff · Σ_j P_eff[i,j] · (Qout[i,j] · W_B[j] · Qin[i,j]^T)
        b_A ← (1-α)·b_A + α · L_eff · Σ_j P_eff[i,j] · (Qout[i,j] · b_B[j])

      If use_hot_as_residual=True:
        W_A is no longer directly modified, the above sum accumulates as W_residual, stored as:
          linear.hot_residual_weight
        Used in forward pass as "weight-level addition":
          W_eff = W_A + W_residual
        This residual weight is fixed and does not participate in training gradients (buffer).

      If hot_neuron_select_dir is not None:
        - Read indices from top_neurons_{kind}_layer_{i}.pt
        - For Q/K/V: only fuse these rows: (1-α)*A[row] + α*HOT[row] when row ∈ idx; other rows unchanged
        - For O: only fuse these columns: (1-α)*A[:,col] + α*HOT[:,col] when col ∈ idx; other columns unchanged
    """
    baseA = _maybe_language_submodule(modelA, use_language_model_only)
    baseB = _maybe_language_submodule(modelB, use_language_model_only)

    blocksA = _collect_attn_blocks(baseA)
    blocksB = _collect_attn_blocks(baseB)
    print("==== BlocksA (Model A) layer order ====")
    for idx, bh in enumerate(blocksA):
        print(idx, bh.map["Q"].weight.shape)
    print("==== BlocksB (Model B) layer order ====")
    for idx, bh in enumerate(blocksB):
        print(idx, bh.map["Q"].weight.shape)

    if len(blocksA) == 0 or len(blocksB) == 0:
        raise RuntimeError(f"No attention blocks found. A={len(blocksA)}, B={len(blocksB)}")

    L_full, M_full = len(blocksA), len(blocksB)
    devA = _find_first_param_device(baseA)
    attn_dev = torch.device(attn_device)

    report: Dict[str, Any] = {
        "hot_dir": hot_dir,
        "alpha": alpha,
        "use_language_model_only": use_language_model_only,
        "hot_mode": "residual" if use_hot_as_residual else "fuse",
        "hot_residual_scale": hot_residual_scale,
        "L_full": L_full,
        "M_full": M_full,
        "components": {},
        "sparsify": {
            "p_topk": p_topk, "p_threshold": p_threshold,
            "q_topr_row": q_topr_row, "q_topr_col": q_topr_col,
            "q_threshold": q_threshold, "q_sinkhorn_iters": q_sinkhorn_iters,
        },
        "attn": {"device": attn_device, "max_mem_mb": attn_max_mem_mb, "autocast": attn_autocast},
    }

    # -------- Neuron Select (optional): load key neuron indices --------
    hot_sel = None
    if hot_neuron_select_dir:
        try:
            kinds_sel = ["Q", "K", "V", "O"]
            hot_sel = _load_top_indices_dir(hot_neuron_select_dir, kinds=kinds_sel, num_layers=L_full)
            report["hot_neuron_select"] = {
                "dir": hot_neuron_select_dir,
                "kinds": kinds_sel,
            }
            print(f"[HOT NeuronSelect] Enabled, dir={hot_neuron_select_dir}")
        except Exception as e:
            print(f"[HOT NeuronSelect][Warn] failed to load indices from {hot_neuron_select_dir}: {e}")
            report["hot_neuron_select_error"] = str(e)
            hot_sel = None
    else:
        report["hot_neuron_select"] = {"dir": None, "kinds": [], "enabled": False}

    # Prefetch B weights to CPU (move to attn_device as needed)
    WB: Dict[str, List[torch.Tensor]] = {c: [] for c in ["Q", "K", "V", "O"]}
    bB: Dict[str, List[Optional[torch.Tensor]]] = {c: [] for c in ["Q", "K", "V", "O"]}
    for comp in ["Q", "K", "V", "O"]:
        for bhB in blocksB:
            linB: nn.Linear = bhB.map[comp]  # type: ignore
            WB[comp].append(linB.weight.detach().to("cpu"))
            bB[comp].append(None if linB.bias is None else linB.bias.detach().to("cpu"))

    def _maybe_sparsify_P(P: torch.Tensor) -> torch.Tensor:
        if (p_topk and p_topk > 0) or (p_threshold and p_threshold > 0.0):
            return sparsify_P_topk(P, k=p_topk, threshold=p_threshold, keep_row_mass=True)
        return P

    for comp in ["Q", "K", "V", "O"]:
        if verbose:
            print(f"[Fuse] component = {comp}")

        dA_out, dB_out, dA_in, dB_in = _get_dim_lists(blocksA, blocksB, comp)

        P_post, Qout_full, L_post, keptA_post, keptB_post, path_post = _load_hot_bundle_expand(
            hot_dir, f"hot_{comp}", L_full, M_full, dA_out, dB_out
        )
        P_pre,  Qin_full,  L_pre,  keptA_pre,  keptB_pre,  path_pre  = _load_hot_bundle_expand(
            hot_dir, f"hot_{comp}_pre", L_full, M_full, dA_in, dB_in
        )

        if verbose:
            for ti in range(min(3, L_full)):
                for tj in range(min(3, M_full)):
                    qout = Qout_full[ti][tj]
                    qin  = Qin_full[ti][tj]
                    if qout is not None:
                        print(f"[HOT CHECK] comp={comp}, i={ti}, j={tj}, "
                              f"Qout mean_abs={qout.abs().mean().item():.6e}, max={qout.abs().max().item():.6e}")
                    if qin is not None:
                        print(f"[HOT CHECK] comp={comp}, i={ti}, j={tj}, "
                              f"Qin  mean_abs={qin.abs().mean().item():.6e}, max={qin.abs().max().item():.6e}")


        P_eff = torch.sqrt(P_post.clamp_min(0)) * torch.sqrt(P_pre.clamp_min(0))
        L_eff = int(round(math.sqrt(max(1, L_post * L_pre))))

        comp_rep: Dict[str, Any] = {
            "files": {"post": path_post, "pre": path_pre},
            "L_post": L_post, "L_pre": L_pre, "L_eff": L_eff,
            "kept_layers_A": {"post": keptA_post, "pre": keptA_pre},
            "kept_layers_B": {"post": keptB_post, "pre": keptB_pre},
            "layers": [], "pairs_used": 0, "pairs_skipped": 0,
            "neuron_select_enabled": bool(hot_sel is not None),
        }

        for i, bhA in enumerate(blocksA):
            linA: nn.Linear = bhA.map[comp]  # type: ignore
            WA, bA = linA.weight, linA.bias

            rec_i: Dict[str, Any] = {
                "layer_i": i,
                "A_weight_shape": list(WA.shape),
                "A_bias": (bA is not None),
                "skipped": False,
                "skip_reason": None,
                "pairs": []
            }

            if WA.shape != (dA_out[i], dA_in[i]):
                rec_i["skipped"] = True
                rec_i["skip_reason"] = f"A[{i}].{comp} weight shape {tuple(WA.shape)} != ({dA_out[i]}, {dA_in[i]})"
                comp_rep["layers"].append(rec_i)
                comp_rep["pairs_skipped"] += M_full
                if verbose:
                    print(f"  [Skip] {rec_i['skip_reason']}")
                continue

            W_accum = torch.zeros_like(WA, device=WA.device)
            b_accum = torch.zeros_like(bA, device=WA.device) if bA is not None else None

            debug_layer_sum = 0.0

            for j, bhB in enumerate(blocksB):
                Wb_cpu = WB[comp][j]
                bb_cpu = bB[comp][j]
                pair_rec = {
                    "j": j,
                    "B_weight_shape": list(Wb_cpu.shape) if isinstance(Wb_cpu, torch.Tensor) else None,
                    "Qout_shape": None,
                    "Qin_shape": None,
                    "scale": 0.0,
                    "used": False,
                    "skip_reason": None,
                }

                if not isinstance(Wb_cpu, torch.Tensor) or Wb_cpu.numel() == 0:
                    pair_rec["skip_reason"] = f"B[{j}].{comp} missing"
                    rec_i["pairs"].append(pair_rec)
                    comp_rep["pairs_skipped"] += 1
                    continue
                if Wb_cpu.shape != (dB_out[j], dB_in[j]):
                    pair_rec["skip_reason"] = f"B[{j}].{comp} weight shape {tuple(Wb_cpu.shape)} != ({dB_out[j]}, {dB_in[j]})"
                    rec_i["pairs"].append(pair_rec)
                    comp_rep["pairs_skipped"] += 1
                    if verbose:
                        print(f"    [Skip] {pair_rec['skip_reason']}")
                    continue

                Qout = Qout_full[i][j]
                Qin  = Qin_full[i][j]
                if (Qout is None) or (Qin is None):
                    pair_rec["skip_reason"] = "missing Qout or Qin (likely filtered/empty HOT)"
                    rec_i["pairs"].append(pair_rec)
                    comp_rep["pairs_skipped"] += 1
                    continue

                pair_rec["Qout_shape"] = list(Qout.shape)
                pair_rec["Qin_shape"]  = list(Qin.shape)

                Wb_d = Wb_cpu.to(attn_dev, dtype=WA.dtype, non_blocking=True)
                Qout_d = Qout.to(attn_dev, dtype=WA.dtype, non_blocking=True)
                Qin_t_d = Qin.to(attn_dev, dtype=WA.dtype, non_blocking=True).t()

                W_contrib = _triple_product_chunked(
                    Qout=Qout_d, Wb=Wb_d, Qin_t=Qin_t_d,
                    target_device=attn_dev, use_autocast=attn_autocast, max_mem_mb=attn_max_mem_mb
                )
                if W_contrib.device != WA.device:
                    W_contrib = W_contrib.to(WA.device, non_blocking=True)

                scale = float(L_eff) * float(P_eff[i, j].item())
                pair_rec["scale"] = scale

                if scale != 0.0:
                    W_accum.add_(W_contrib, alpha=scale)

                    contrib_norm = float(W_contrib.abs().mean().item())
                    debug_layer_sum += scale * contrib_norm

                    if verbose:
                        print(
                            f"      [HOT DEBUG] comp={comp}, i={i}, j={j}, "
                            f"scale={scale:.6f}, W_contrib_mean_abs={contrib_norm:.6f}"
                        )

                    if bA is not None and isinstance(bb_cpu, torch.Tensor) and bb_cpu.numel() > 0:
                        bb_d = bb_cpu.to(attn_dev, dtype=bA.dtype, non_blocking=True)
                        use_autocast_bias = (attn_autocast and attn_dev.type == "cuda")
                        with torch.cuda.amp.autocast(enabled=use_autocast_bias, dtype=torch.float32):
                            b_part = Qout_d.to(dtype=bA.dtype) @ bb_d
                        b_part = b_part.to(WA.device, non_blocking=True)
                        b_accum.add_(b_part, alpha=scale)
                        del bb_d, b_part
                    pair_rec["used"] = True
                    comp_rep["pairs_used"] += 1
                else:
                    pair_rec["skip_reason"] = "scale=0 (P_eff=0 or L_eff=0)"
                    comp_rep["pairs_skipped"] += 1

                del Wb_d, Qout_d, Qin_t_d, W_contrib
                if attn_dev.type == "cuda":
                    torch.cuda.empty_cache()

                rec_i["pairs"].append(pair_rec)

            if verbose:
                print(
                    f"  [HOT DEBUG] comp={comp}, layer i={i}, "
                    f"total HOT contrib (scale * mean|W_contrib| sum) = {debug_layer_sum:.6f}"
                )
            if debug_layer_sum == 0.0:
                print(
                    f"  [HOT DEBUG WARNING] comp={comp}, layer {i}: "
                    f"total HOT contribution is 0. "
                    f"This likely means P_eff is all zeros or Qout/Qin are empty."
                )

            # -------- Neuron Select: record indices (actual selection logic done in fusion stage) --------
            if hot_sel is not None:
                all_idx_for_comp = hot_sel.get(comp, [])
                if isinstance(all_idx_for_comp, list) and i < len(all_idx_for_comp):
                    idx_log = all_idx_for_comp[i]
                else:
                    idx_log = torch.empty(0, dtype=torch.long)
                if not isinstance(idx_log, torch.Tensor):
                    idx_log = torch.as_tensor(idx_log, dtype=torch.long)
                rec_i.setdefault("neuron_select", {})[comp] = {
                    "num_selected": int(idx_log.numel()),
                }
                if verbose:
                    if idx_log.numel() <= 16:
                        print(f"  [NeuronSelect] comp={comp}, layer {i}: idx={idx_log.tolist()}")
                    else:
                        print(f"  [NeuronSelect] comp={comp}, layer {i}: num_indices={idx_log.numel()}")

            # -------- Apply to model: residual mode / direct fusion (with neuron-select aligned to codeA behavior) --------
            if use_hot_as_residual:
                # ========== Residual Mode ==========
                if hot_sel is not None:
                    # Only write HOT residual for selected neurons, other neurons residual=0 (equivalent to no change)
                    all_idx_for_comp = hot_sel.get(comp, [])
                    if isinstance(all_idx_for_comp, list) and i < len(all_idx_for_comp):
                        idx = all_idx_for_comp[i]
                    else:
                        idx = torch.empty(0, dtype=torch.long)
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.as_tensor(idx, dtype=torch.long)
                    idx = idx.to(WA.device)

                    if idx.numel() == 0:
                        # No neurons need residual, skip directly
                        if verbose:
                            print(f"  [Residual-NeuronSelect] comp={comp}, layer {i}: no indices, skip residual.")
                    else:
                        # Key change:
                        # W_res directly copies from current W_base,
                        # so unselected neurons have W_hot = W_base,
                        # substituting W_eff = (1-α) W_base + α W_hot => W_eff = W_base (no change at all)
                        W_res = WA.detach().clone()

                        # Same for bias: default to original bias, overwrite for selected neurons
                        if bA is not None and b_accum is not None:
                            b_res = bA.detach().clone()
                        else:
                            b_res = None

                        if comp in ["Q", "K", "V"]:
                            # Row selection: only overwrite HOT for selected rows
                            valid = idx[(idx >= 0) & (idx < W_res.size(0))]
                            if valid.numel() > 0:
                                W_res[valid] = W_accum[valid]
                                if b_res is not None:
                                    b_res[valid] = b_accum[valid]
                        else:  # O, column selection
                            valid = idx[(idx >= 0) & (idx < W_res.size(1))]
                            if valid.numel() > 0:
                                W_res[:, valid] = W_accum[:, valid]
                                # O usually has no bias, skip bias handling

                        _attach_hot_residual_to_linear(
                            linA,
                            W_res=W_res,
                            b_res=b_res,
                            alpha=alpha,
                        )
                        #_patch_linear_for_hot_residual(linA, alpha)
                        if verbose:
                            print(
                                f"  [Residual-NeuronSelect] comp={comp}, layer {i}: "
                                f"residual attached on selected neurons (non-selected use base params)."
                            )

                else:
                    # Dense residual: all neurons use HOT
                    _attach_hot_residual_to_linear(
                        linA,
                        W_res=W_accum,
                        b_res=b_accum,
                        alpha=alpha,
                    )
                    #_patch_linear_for_hot_residual(linA, alpha)
                    if verbose:
                        print(f"  [Residual] comp={comp}, layer {i}: dense residual attached.")
            else:
                # ========== Directly modify weights (non-residual) ==========
                if hot_sel is not None:
                    # CodeA-style: only do (1-α)*A + α*HOT for selected neurons, other neurons completely unchanged
                    all_idx_for_comp = hot_sel.get(comp, [])
                    if isinstance(all_idx_for_comp, list) and i < len(all_idx_for_comp):
                        idx = all_idx_for_comp[i]
                    else:
                        idx = torch.empty(0, dtype=torch.long)
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.as_tensor(idx, dtype=torch.long)
                    idx = idx.to(WA.device)

                    if idx.numel() == 0:
                        # No selected neurons, layer remains unchanged
                        if verbose:
                            print(f"  [Fuse-NeuronSelect] comp={comp}, layer {i}: no indices, skip fusion.")
                    else:
                        if comp in ["Q", "K", "V"]:
                            # Row selection (output dimension)
                            valid = idx[(idx >= 0) & (idx < WA.size(0))]
                            if valid.numel() > 0:
                                mask = torch.zeros(WA.size(0), dtype=torch.bool, device=WA.device)
                                mask[valid] = True

                                WA_old = WA.data.clone()
                                WA.data[mask] = (1.0 - alpha) * WA_old[mask] + alpha * W_accum[mask]

                                if bA is not None and b_accum is not None:
                                    b_old = bA.data.clone()
                                    bA.data[mask] = (1.0 - alpha) * b_old[mask] + alpha * b_accum[mask]

                                if verbose:
                                    print(f"  [Fuse-NeuronSelect] comp={comp}, layer {i}: fused {int(valid.numel())} rows.")
                        else:
                            # O: column selection (input dimension)
                            valid = idx[(idx >= 0) & (idx < WA.size(1))]
                            if valid.numel() > 0:
                                mask = torch.zeros(WA.size(1), dtype=torch.bool, device=WA.device)
                                mask[valid] = True

                                WA_old = WA.data.clone()
                                WA.data[:, mask] = (1.0 - alpha) * WA_old[:, mask] + alpha * W_accum[:, mask]
                                # O usually has no bias; if present can broadcast by column, ignored here

                                if verbose:
                                    print(f"  [Fuse-NeuronSelect] comp={comp}, layer {i}: fused {int(valid.numel())} cols.")
                else:
                    # Dense fusion: all neurons do (1-α)*A + α*HOT (original logic)
                    WA.mul_(1.0 - alpha).add_(W_accum, alpha=alpha)
                    if bA is not None and b_accum is not None:
                        bA.mul_(1.0 - alpha).add_(b_accum, alpha=alpha)
                    if verbose:
                        print(f"  [Fuse-Dense] comp={comp}, layer {i}: dense fusion applied.")

            comp_rep["layers"].append(rec_i)

        if verbose:
            print(f"[Fuse] component {comp} completed. L_eff={L_eff}")

        report["components"][comp] = comp_rep

    print("[Done] Attention-only HOT fusion completed (low-memory).")

    return report


# ============================================================
# Top Neuron Replacement (replace "non-transferred TOP" slices into clean basemodelA)
# ============================================================

def _maybe_language_submodule_for_top(model: nn.Module) -> nn.Module:
    return _maybe_language_submodule(model, use_language_model_only=True)

def _get_decoder_layers_list(model_root: nn.Module):
    import types
    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", types.SimpleNamespace()), "layers", None)),
        ("layers",       lambda m: getattr(m, "layers", None)),
        ("transformer.h",lambda m: getattr(getattr(m, "transformer", types.SimpleNamespace()), "h", None)),
        ("blocks",       lambda m: getattr(m, "blocks", None)),
    ]
    for _, getter in candidates:
        layers = getter(model_root)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return model_root, list(layers)
    for mod in model_root.modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            if hasattr(first, "self_attn") or hasattr(first, "mlp"):
                return model_root, list(mod)
    raise AttributeError("Could not find decoder layer list (layers/h/blocks) in current model structure.")

def _get_attn_mlp_linears(layer: nn.Module) -> dict:
    sa = getattr(layer, "self_attn", layer)
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        mlp = getattr(layer, "feed_forward", getattr(layer, "ffn", None))

    def pick(obj, *names):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if isinstance(v, nn.Linear):
                    return v
        return None

    q = pick(sa, "q_proj")
    k = pick(sa, "k_proj")
    v = pick(sa, "v_proj")
    o = pick(sa, "o_proj", "out_proj")

    up = gate = down = None
    if mlp is not None:
        up   = pick(mlp, "up_proj", "w1")
        gate = pick(mlp, "gate_proj", "w3")
        down = pick(mlp, "down_proj", "w2")

    return {"Q": q, "K": k, "V": v, "O": o, "up": up, "gate": gate, "down": down}

@torch.no_grad()
def _copy_rows(dst: nn.Linear, src: nn.Linear, rows: torch.Tensor):
    if not isinstance(dst, nn.Linear) or not isinstance(src, nn.Linear):
        return
    if rows.numel() == 0:
        return
    rows = rows[(rows >= 0) & (rows < dst.out_features)]
    if rows.numel() == 0:
        return
    dst.weight.data[rows] = src.weight.data[rows].to(dst.weight.dtype, non_blocking=True)
    if (dst.bias is not None) and (src.bias is not None):
        dst.bias.data[rows] = src.bias.data[rows].to(dst.bias.dtype, non_blocking=True)

@torch.no_grad()
def _copy_cols(dst: nn.Linear, src: nn.Linear, cols: torch.Tensor):
    if not isinstance(dst, nn.Linear) or not isinstance(src, nn.Linear):
        return
    if cols.numel() == 0:
        return
    cols = cols[(cols >= 0) & (cols < dst.in_features)]
    if cols.numel() == 0:
        return
    dst.weight.data[:, cols] = src.weight.data[:, cols].to(dst.weight.dtype, non_blocking=True)

@torch.no_grad()
def replace_top_neurons_between_models(
    src_model: nn.Module,
    dst_model: nn.Module,
    top_neuron_dir: str,
    kinds: list[str],
    use_language_model_only: bool = True,
    verify: bool = True,
    return_report: bool = False,
    atol: float = 1e-6,
    max_print: int = 20,
):
    src_base = _maybe_language_submodule_for_top(src_model) if use_language_model_only else src_model
    dst_base = _maybe_language_submodule_for_top(dst_model) if use_language_model_only else dst_model

    _, src_layers = _get_decoder_layers_list(src_base)
    _, dst_layers = _get_decoder_layers_list(dst_base)
    if len(src_layers) != len(dst_layers):
        raise ValueError(f"Layer count mismatch: src={len(src_layers)} vs dst={len(dst_layers)}")

    L = len(dst_layers)
    top = _load_top_indices_dir(top_neuron_dir, kinds=kinds, num_layers=L)

    report = {
        "layers": L,
        "changed": 0,
        "checked": 0,
        "mismatches": 0,
        "skipped": 0,
        "prints": [],
        "per_kind": {k: {"changed": 0, "checked": 0, "mismatches": 0, "skipped": 0} for k in kinds},
    }

    def _log(msg):
        if len(report["prints"]) < max_print:
            print(msg)
        report["prints"].append(msg)

    for li in range(L):
        s_lin = _get_attn_mlp_linears(src_layers[li])
        d_lin = _get_attn_mlp_linears(dst_layers[li])

        # Q/K/V row replacement
        for kind in ["Q", "K", "V"]:
            idx = top.get(kind, [torch.empty(0)])[li]
            if idx.numel() == 0 or d_lin[kind] is None or s_lin[kind] is None:
                if idx.numel() > 0:
                    report["skipped"] += idx.numel()
                    report["per_kind"][kind]["skipped"] += idx.numel()
                continue
            rows = idx[(idx >= 0) & (idx < d_lin[kind].out_features)]
            skipped_here = idx.numel() - rows.numel()
            if skipped_here > 0:
                report["skipped"] += skipped_here
                report["per_kind"][kind]["skipped"] += skipped_here
            if rows.numel() == 0:
                continue

            w_base_rows = d_lin[kind].weight.data[rows].clone()
            b_base_rows = d_lin[kind].bias.data[rows].clone() if (d_lin[kind].bias is not None) else None

            _copy_rows(d_lin[kind], s_lin[kind], rows)

            if verify:
                w_upd_rows = d_lin[kind].weight.data[rows]
                b_upd_rows = d_lin[kind].bias.data[rows] if (d_lin[kind].bias is not None) else None
                w_src_rows = s_lin[kind].weight.data[rows]
                b_src_rows = s_lin[kind].bias.data[rows] if (s_lin[kind].bias is not None) else None

                ok_w = torch.allclose(w_upd_rows, w_src_rows, atol=atol)
                changed_w = not torch.allclose(w_upd_rows, w_base_rows, atol=atol)
                if not ok_w:
                    report["mismatches"] += rows.numel()
                    report["per_kind"][kind]["mismatches"] += rows.numel()
                    _log(f"[MM] layer {li:02d} {kind} rows {rows.tolist()} weight mismatch vs src")
                if changed_w:
                    report["changed"] += rows.numel()
                    report["per_kind"][kind]["changed"] += rows.numel()
                    _log(f"[Changed] layer {li:02d} {kind} rows {rows.tolist()}")
                report["checked"] += rows.numel()
                report["per_kind"][kind]["checked"] += rows.numel()

                if (b_upd_rows is not None) and (b_src_rows is not None) and (b_base_rows is not None):
                    ok_b = torch.allclose(b_upd_rows, b_src_rows, atol=atol)
                    changed_b = not torch.allclose(b_upd_rows, b_base_rows, atol=atol)
                    if not ok_b:
                        report["mismatches"] += rows.numel()
                        report["per_kind"][kind]["mismatches"] += rows.numel()
                        _log(f"[MM] layer {li:02d} {kind} rows {rows.tolist()} bias mismatch vs src")
                    if changed_b:
                        _log(f"[ChangedBias] layer {li:02d} {kind} rows {rows.tolist()}")

        # O column replacement
        kind = "O"
        idx = top.get(kind, [torch.empty(0)])[li]
        if idx.numel() > 0 and d_lin["O"] is not None and s_lin["O"] is not None:
            cols = idx[(idx >= 0) & (idx < d_lin["O"].in_features)]
            skipped_here = idx.numel() - cols.numel()
            if skipped_here > 0:
                report["skipped"] += skipped_here
                report["per_kind"][kind]["skipped"] += skipped_here
            if cols.numel() > 0:
                w_base_cols = d_lin["O"].weight.data[:, cols].clone()
                _copy_cols(d_lin["O"], s_lin["O"], cols)

                if verify:
                    w_upd_cols = d_lin["O"].weight.data[:, cols]
                    w_src_cols = s_lin["O"].weight.data[:, cols]
                    ok_w = torch.allclose(w_upd_cols, w_src_cols, atol=atol)
                    changed_w = not torch.allclose(w_upd_cols, w_base_cols, atol=atol)
                    if not ok_w:
                        report["mismatches"] += cols.numel()
                        report["per_kind"][kind]["mismatches"] += cols.numel()
                        _log(f"[MM] layer {li:02d} O cols {cols.tolist()} weight mismatch vs src")
                    if changed_w:
                        report["changed"] += cols.numel()
                        report["per_kind"][kind]["changed"] += cols.numel()
                        _log(f"[Changed] layer {li:02d} O cols {cols.tolist()}")
                    report["checked"] += cols.numel()
                    report["per_kind"][kind]["checked"] += cols.numel()
        elif idx.numel() > 0:
            report["skipped"] += idx.numel()
            report["per_kind"][kind]["skipped"] += idx.numel()

        # fwd_up (up/gate row replacement)
        kind = "fwd_up"
        idx = top.get(kind, [torch.empty(0)])[li]
        for key in ["up", "gate"]:
            if idx.numel() == 0 or d_lin[key] is None or s_lin[key] is None:
                if idx.numel() > 0:
                    report["skipped"] += idx.numel()
                    report["per_kind"][kind]["skipped"] += idx.numel()
                continue
            rows = idx[(idx >= 0) & (idx < d_lin[key].out_features)]
            skipped_here = idx.numel() - rows.numel()
            if skipped_here > 0:
                report["skipped"] += skipped_here
                report["per_kind"][kind]["skipped"] += skipped_here
            if rows.numel() == 0:
                continue

            w_base_rows = d_lin[key].weight.data[rows].clone()
            b_base_rows = d_lin[key].bias.data[rows].clone() if (d_lin[key].bias is not None) else None

            _copy_rows(d_lin[key], s_lin[key], rows)

            if verify:
                w_upd_rows = d_lin[key].weight.data[rows]
                b_upd_rows = d_lin[key].bias.data[rows] if (d_lin[key].bias is not None) else None
                w_src_rows = s_lin[key].weight.data[rows]
                b_src_rows = s_lin[key].bias.data[rows] if (s_lin[key].bias is not None) else None

                ok_w = torch.allclose(w_upd_rows, w_src_rows, atol=atol)
                changed_w = not torch.allclose(w_upd_rows, w_base_rows, atol=atol)
                if not ok_w:
                    report["mismatches"] += rows.numel()
                    report["per_kind"][kind]["mismatches"] += rows.numel()
                    _log(f"[MM] layer {li:02d} {key} rows {rows.tolist()} weight mismatch vs src")
                if changed_w:
                    report["changed"] += rows.numel()
                    report["per_kind"][kind]["changed"] += rows.numel()
                    _log(f"[Changed] layer {li:02d} {key} rows {rows.tolist()}")

                report["checked"] += rows.numel()
                report["per_kind"][kind]["checked"] += rows.numel()

                if (b_upd_rows is not None) and (b_src_rows is not None) and (b_base_rows is not None):
                    ok_b = torch.allclose(b_upd_rows, b_src_rows, atol=atol)
                    changed_b = not torch.allclose(b_upd_rows, b_base_rows, atol=atol)
                    if not ok_b:
                        report["mismatches"] += rows.numel()
                        report["per_kind"][kind]["mismatches"] += rows.numel()
                        _log(f"[MM] layer {li:02d} {key} rows {rows.tolist()} bias mismatch vs src")
                    if changed_b:
                        _log(f"[ChangedBias] layer {li:02d} {key} rows {rows.tolist()}")

        # fwd_down (down column replacement)
        kind = "fwd_down"
        idx = top.get(kind, [torch.empty(0)])[li]
        if idx.numel() > 0 and d_lin["down"] is not None and s_lin["down"] is not None:
            cols = idx[(idx >= 0) & (idx < d_lin["down"].in_features)]
            skipped_here = idx.numel() - cols.numel()
            if skipped_here > 0:
                report["skipped"] += skipped_here
                report["per_kind"][kind]["skipped"] += skipped_here
            if cols.numel() > 0:
                w_base_cols = d_lin["down"].weight.data[:, cols].clone()
                _copy_cols(d_lin["down"], s_lin["down"], cols)

                if verify:
                    w_upd_cols = d_lin["down"].weight.data[:, cols]
                    w_src_cols = s_lin["down"].weight.data[:, cols]
                    ok_w = torch.allclose(w_upd_cols, w_src_cols, atol=atol)
                    changed_w = not torch.allclose(w_upd_cols, w_base_cols, atol=atol)
                    if not ok_w:
                        report["mismatches"] += cols.numel()
                        report["per_kind"][kind]["mismatches"] += cols.numel()
                        _log(f"[MM] layer {li:02d} down cols {cols.tolist()} weight mismatch vs src")
                    if changed_w:
                        report["changed"] += cols.numel()
                        report["per_kind"][kind]["changed"] += cols.numel()
                        _log(f"[Changed] layer {li:02d} down cols {cols.tolist()}")
                    report["checked"] += cols.numel()
                    report["per_kind"][kind]["checked"] += cols.numel()
        elif idx.numel() > 0:
            report["skipped"] += idx.numel()
            report["per_kind"][kind]["skipped"] += idx.numel()

    if verify:
        print(f"[TopReplace] Verify: changed={report['changed']} checked={report['checked']} mismatches={report['mismatches']} skipped={report['skipped']}")
        if report["mismatches"] == 0 and report["changed"] > 0:
            print("[TopReplace] ✅ Replacement verified (updated == HOT slices, and != baseline on changed parts)")
        elif report["changed"] == 0:
            print("[TopReplace]⚠️ No neuron indices applied (all empty or out-of-range).")
        else:
            print("[TopReplace] ❌ Found mismatches, please inspect 'prints' details.")

    return (dst_model, report) if return_report else dst_model


# ============ CLI ============

if __name__ == "__main__":
    import argparse
    import sys
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    # Qwen2.5-VL / Qwen2-VL multimodal models need dedicated classes or Vision2Seq auto class
    try:
        from transformers import (
            AutoModelForVision2Seq,
            Qwen2VLForConditionalGeneration,
        )  # type: ignore
    except Exception:
        AutoModelForVision2Seq = None  # type: ignore
        Qwen2VLForConditionalGeneration = None  # type: ignore

    # TinyLLaVA loading function (Factory / Bench are exported from tinyllava.model)
    try:
        from tinyllava.model import load_pretrained_model as tinyllava_load_pretrained_model  # type: ignore
    except Exception:
        tinyllava_load_pretrained_model = None  # type: ignore

    parser = argparse.ArgumentParser(
        description="Fuse attention-only weights using HOT (post & pre) with low-memory chunked matmul; and also save a top-neuron-replaced baseline."
    )
    parser.add_argument(
        "--modelA_id",
        type=str,
        default="<path_to_model_a>",
    )
    parser.add_argument(
        "--modelB_id",
        type=str,
        default="<path_to_model_b>",
    )
    parser.add_argument(
        "--hot_dir",
        type=str,
        default="<path_to_hot_dir>",
        help="Directory containing hot_{Q,K,V,O}.pt and hot_{Q,K,V,O}_pre.pt",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lm_only", action="store_true")
    parser.add_argument("--no_lm_only", dest="lm_only", action="store_false")
    parser.set_defaults(lm_only=True)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="<path_to_output_dir>",
    )

    # Sparsification
    parser.add_argument("--p_topk", type=int, default=3)
    parser.add_argument("--p_threshold", type=float, default=0.0)
    parser.add_argument("--q_topr_row", type=int, default=128)
    parser.add_argument("--q_topr_col", type=int, default=128)
    parser.add_argument("--q_threshold", type=float, default=0.0)
    parser.add_argument("--q_sinkhorn_iters", type=int, default=5)

    # Low-memory Attention control
    parser.add_argument(
        "--attn_device",
        type=str,
        default="cuda:0",
        help="'cuda:0' or 'cpu'",
    )
    parser.add_argument(
        "--attn_max_mem_mb",
        type=int,
        default=1200,
        help="Target memory (MB) for attention chunked matmul.",
    )
    parser.add_argument(
        "--attn_no_autocast",
        action="store_true",
        help="Disable CUDA autocast(bf16) for attention.",
    )

    # HOT residual mode
    parser.add_argument(
        "--use_hot_residual",
        default=True,
        help="Use HOT contribution as fixed residual (buffer) instead of directly modifying modelA weights.",
    )
    parser.add_argument(
        "--hot_residual_scale",
        type=float,
        default=1.0,
        help="Global scale for HOT residual when --use_hot_residual is set.",
    )

    # New: HOT residual neuron-select mode
    parser.add_argument(
        "--hot_neuron_dir",
        type=str,
        default=None,
        help=(
            "If set, only neurons in this directory's index files will receive HOT residual/fusion. "
            "File pattern: top_neurons_{Q,K,V,O}_layer_{i}.pt, each containing 1D LongTensor of indices. "
            "For Q/K/V indices are output rows; for O indices are input columns."
        ),
    )

    # Additional: Top Neuron Replacement
    parser.add_argument(
        "--top_neuron_dir",
        type=str,
        default=None,
        help="Directory containing top_neurons_{kind}_layer_{i}.pt. If set, will replace TOP neurons into a clean baseline copy.",
    )
    parser.add_argument(
        "--top_replace_output_dir",
        type=str,
        default=None,
        help="Where to save the baseline-with-replaced-top-neurons model. If None, will default to {output_dir}_topneurons",
    )
    parser.add_argument("--top_verify_atol", type=float, default=1e-6)
    parser.add_argument("--top_max_print", type=int, default=20)

    args = parser.parse_args()

    # ---- Generic loading function: supports Qwen2 / Qwen2.5 / Qwen2-VL / Qwen2.5-VL / TinyLLaVA / regular CausalLM ----
    from transformers import AutoConfig, AutoModelForCausalLM

    def _load_backbone(model_id: str, device_map=None):
        """
        Unified entry point:
        - If path/name contains 'TinyLLaVA', prioritize tinyllava.load_pretrained_model
        - Otherwise use HF's AutoConfig + AutoModelForCausalLM / Qwen2VL / Vision2Seq
        - Qwen2 and Qwen2.5 text models (e.g., Qwen/Qwen2.5-7B-Instruct) will be loaded via AutoModelForCausalLM
        """
        # 1) TinyLLaVA: determine by name
        base_name = os.path.basename(str(model_id))
        if ("TinyLLaVA" in base_name) or ("TinyLLaVA" in str(model_id)):
            if tinyllava_load_pretrained_model is None:
                raise RuntimeError(
                    f"modelA_id appears to be TinyLLaVA ({model_id}), but tinyllava is not properly installed."
                )
            print("[Backbone] Loading TinyLLaVA via tinyllava.load_pretrained_model ...")

            res = tinyllava_load_pretrained_model(model_id)

            if isinstance(res, tuple):
                model = None
                for x in res:
                    if isinstance(x, nn.Module):
                        model = x
                        break
                if model is None:
                    raise RuntimeError(
                        "tinyllava.load_pretrained_model returned no nn.Module, does not match TinyLLaVA expectation."
                    )
            elif isinstance(res, nn.Module):
                model = res
            else:
                raise RuntimeError(
                    f"tinyllava.load_pretrained_model returned unexpected type: {type(res)}"
                )

            return model

        # 2) Other models: use HF AutoConfig to determine type
        cfg = AutoConfig.from_pretrained(model_id)
        model_type = getattr(cfg, "model_type", None)
        print("[Backbone] model_type =", model_type)

        # Qwen2-VL series
        if model_type == "qwen2_vl":
            if Qwen2VLForConditionalGeneration is None:
                raise RuntimeError(
                    "Detected Qwen2-VL model, but current transformers does not have Qwen2VLForConditionalGeneration, please upgrade transformers>=4.40+."
                )
            print(
                "[Backbone] Loading Qwen2-VL via Qwen2VLForConditionalGeneration..."
            )
            return Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map=device_map,
            )

        # Multimodal models with vision_config, prioritize Vision2Seq
        if hasattr(cfg, "vision_config") and AutoModelForVision2Seq is not None:
            try:
                print("[Backbone] Loading VLM via AutoModelForVision2Seq...")
                return AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map=device_map,
                )
            except Exception as e:
                print("[Backbone][WARN] AutoModelForVision2Seq failed:", e)

        # Default: pure language models (including Qwen2 / Qwen2.5 text models)
        print("[Backbone] Loading regular CausalLM (supports Qwen2/Qwen2.5 text models)...")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
        )

    # 1) Load A
    print(f"[Load] Model A: {args.modelA_id}")
    modelA = _load_backbone(args.modelA_id, device_map=None)
    if torch.cuda.is_available():
        try:
            modelA = modelA.to("cuda:0")
        except Exception:
            pass
    modelA = modelA.eval()

    # 2) Load B
    print(f"[Load] Model B: {args.modelB_id}")
    modelB = _load_backbone(args.modelB_id, device_map=None).eval()

    # 3) Fusion
    print(
        f"[Fuse] HOT dir = {args.hot_dir} | alpha = {args.alpha} | lm_only = {args.lm_only}"
    )
    print(
        f"[Fuse] Low-memory attention: device={args.attn_device}, max_mem_mb={args.attn_max_mem_mb}, autocast={not args.attn_no_autocast}"
    )
    print(
        f"[Fuse] HOT mode = {'residual' if args.use_hot_residual else 'fuse-direct'}, "
        f"hot_residual_scale={args.hot_residual_scale}"
    )
    if args.hot_neuron_dir:
        print(f"[Fuse] HOT neuron-select enabled, hot_neuron_dir={args.hot_neuron_dir}")
    else:
        print("[Fuse] HOT neuron-select disabled (dense HOT on all neurons).")

    try:
        with torch.no_grad():
            report = fuse_attention_only_from_hot_dir(
                modelA=modelA,
                modelB=modelB,
                hot_dir=args.hot_dir,
                alpha=args.alpha,
                use_language_model_only=args.lm_only,
                verbose=args.verbose,
                p_topk=args.p_topk,
                p_threshold=args.p_threshold,
                q_topr_row=args.q_topr_row,
                q_topr_col=args.q_topr_col,
                q_threshold=args.q_threshold,
                q_sinkhorn_iters=args.q_sinkhorn_iters,
                attn_device=args.attn_device,
                attn_max_mem_mb=args.attn_max_mem_mb,
                attn_autocast=(not args.attn_no_autocast),
                use_hot_as_residual=args.use_hot_residual,
                hot_residual_scale=args.hot_residual_scale,
                hot_neuron_select_dir=args.hot_neuron_dir,
            )
    except Exception as e:
        print(f"[Error] Fusion failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Save HOT-fused A (full parameter version)
    print(f"[Save] fused model -> {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        modelA.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"[Warn] save_pretrained(modelA) failed: {e}", file=sys.stderr)

    # Copy tokenizer files from modelA to output directory
    try:
        print(f"[Save] copying tokenizer from {args.modelA_id} -> {args.output_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.modelA_id, use_fast=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[Save] tokenizer saved successfully")
    except Exception as e:
        print(f"[Warn] Failed to copy tokenizer: {e}", file=sys.stderr)
    rep_path = os.path.join(args.output_dir, "fusion_report.json")
    try:
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Report] Saved fusion report -> {rep_path}")
    except Exception as e:
        print(f"[Warn] save fusion_report failed: {e}", file=sys.stderr)

    # 5) Save "high-activation neuron" version (replace on clean basemodelA)
    if args.top_neuron_dir:
        try:
            print(
                "[TopReplace] Replacing non-transferred TOP neurons in baseline with HOT-transferred slices..."
            )
            # Load a "clean basemodelA" (consistent with before HOT)
            basemodelA = _load_backbone(args.modelA_id, device_map="auto")
            if torch.cuda.is_available():
                try:
                    basemodelA = basemodelA.to("cuda:0")
                except Exception:
                    pass
            basemodelA = basemodelA.eval()

            # Convention: TOP index kinds
            kinds = ["Q", "K", "V", "O", "fwd_up", "fwd_down"]

            updated_model, toprep = replace_top_neurons_between_models(
                src_model=modelA,  # From HOT-fused A
                dst_model=basemodelA,  # Clean A (replace here)
                top_neuron_dir=args.top_neuron_dir,
                kinds=kinds,
                use_language_model_only=True,
                verify=True,
                return_report=True,
                atol=args.top_verify_atol,
                max_print=args.top_max_print,
            )

            # Save replacement report
            top_rep_path = os.path.join(args.output_dir, "top_replace_report.json")
            with open(top_rep_path, "w", encoding="utf-8") as f:
                json.dump(toprep, f, ensure_ascii=False, indent=2)
            print(f"[TopReplace] Saved report -> {top_rep_path}")

            # Save replaced basemodelA -- always save (even if output directory not explicitly specified)
            top_out_dir = (
                args.top_replace_output_dir
                or (args.output_dir.rstrip("/") + "_topneurons")
            )
            os.makedirs(top_out_dir, exist_ok=True)
            updated_model.save_pretrained(top_out_dir)
            print(
                f"[TopReplace] Updated model with replaced TOP neurons saved to {top_out_dir}"
            )
            # Copy tokenizer files
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.modelA_id, use_fast=True)
                tokenizer.save_pretrained(top_out_dir)
                print(f"[TopReplace] tokenizer saved successfully")
            except Exception as e:
                print(f"[TopReplace][Warn] Failed to copy tokenizer: {e}", file=sys.stderr)

        except Exception as e:
            print(f"[TopReplace][Error] {e}", file=sys.stderr)
    else:
        print("[TopReplace] Skipped: no --top_neuron_dir provided.")

    # Console summary
    total_used = sum(c["pairs_used"] for c in report["components"].values())
    total_skip = sum(c["pairs_skipped"] for c in report["components"].values())
    print(f"[Summary] pairs_used={total_used}, pairs_skipped={total_skip}")
