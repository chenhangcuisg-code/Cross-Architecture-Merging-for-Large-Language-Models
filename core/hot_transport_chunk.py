# ============================================================
# hot_transport_chunk_corr.py
# ============================================================
# 分块稳健 HOT（Hybrid Optimal Transport）
# - 内层：相关距离 + Sinkhorn（分块流式）
# - 外层：稳定 Sinkhorn（log-domain，自适应 eps）
# - 保留旧版数值安全函数与接口兼容性
# ============================================================

from typing import List, Tuple, Optional, Dict, Union
import torch


# -------------------- dtype 适配 --------------------
_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
    torch.bfloat16: torch.bfloat16,
}
def _as_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    return _DTYPE_MAP.get(dtype, torch.float32)


# -------------------- 安全统计函数 --------------------
@torch.no_grad()
def _safe_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x)

@torch.no_grad()
def _safe_min(x: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    m = _safe_mask(x)
    if m.any():
        return torch.min(x[m])
    return torch.tensor(default, dtype=x.dtype, device=x.device)

@torch.no_grad()
def _safe_mean(x: torch.Tensor, default: float = 1.0) -> torch.Tensor:
    m = _safe_mask(x)
    if m.any():
        return torch.mean(x[m])
    return torch.tensor(default, dtype=x.dtype, device=x.device)

@torch.no_grad()
def _safe_median(x: torch.Tensor, default: float = 1.0) -> torch.Tensor:
    m = _safe_mask(x)
    if m.any():
        return torch.median(x[m])
    return torch.tensor(default, dtype=x.dtype, device=x.device)


# ============================================================
# 内层：相关距离 + Sinkhorn（分块）
# ============================================================

@torch.no_grad()
def _zscore_along_time(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按时间维度做 z-score 标准化，用于相关距离。"""
    if X.device.type == "cpu" and X.dtype in (torch.float16, torch.bfloat16):
        X = X.to(torch.float32)
    mean = X.mean(dim=0, keepdim=True)
    var = ((X - mean) ** 2).sum(dim=0, keepdim=True) / max(X.shape[0] - 1, 1)
    std = torch.sqrt(var + eps)
    return (X - mean) / std


@torch.no_grad()
def corr_distance_matrix(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算相关距离矩阵：d_corr = 1 - corr(x,y)。"""
    T = min(X.shape[0], Y.shape[0])
    X = X[:T]
    Y = Y[:T]
    ZX = _zscore_along_time(X, eps)
    ZY = _zscore_along_time(Y, eps)
    rho = (ZX.T @ ZY) / max(T - 1, 1)
    rho = torch.clamp(rho, -1.0, 1.0)
    return 1.0 - rho


@torch.no_grad()
def sinkhorn_uniform_streaming(
    C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, reg: float,
    n_iters: int = 200, tol: float = 1e-6,
    block_cols: int = 2048, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """节省内存的 Sinkhorn for uniform marginals（分块 log-domain 实现）。"""
    n, m = C.shape
    logu = torch.zeros(n, device=C.device, dtype=torch.float32)
    logv = torch.zeros(m, device=C.device, dtype=torch.float32)
    loga = a.clamp_min(1e-30).log()
    logb = b.clamp_min(1e-30).log()
    C_scaled = (C / reg).to(torch.float32)

    def log_Kv(lv):
        out = torch.full((n,), -float("inf"), device=C.device, dtype=torch.float32)
        for j0 in range(0, m, block_cols):
            j1 = min(j0 + block_cols, m)
            block = (-C_scaled[:, j0:j1]) + lv[j0:j1]
            out = torch.logaddexp(out, torch.logsumexp(block, dim=1))
        return out

    def log_KTu(lu):
        out = torch.full((m,), -float("inf"), device=C.device, dtype=torch.float32)
        for i0 in range(0, n, block_cols):
            i1 = min(i0 + block_cols, n)
            block = (-C_scaled[i0:i1, :]) + lu[i0:i1].unsqueeze(1)
            out = torch.logaddexp(out, torch.logsumexp(block, dim=0))
        return out

    prev_err = 1e9
    for t in range(n_iters):
        logu = loga - log_Kv(logv)
        logv = logb - log_KTu(logu)
        if (t % 10 == 0) or (t == n_iters - 1):
            lKv = log_Kv(logv)
            row_marg = (logu + lKv).exp()
            err = (row_marg - a).abs().sum().item()
            if err < tol or abs(err - prev_err) < tol * 0.1:
                break
            prev_err = err

    Q = torch.empty((n, m), device=C.device, dtype=dtype)
    for i0 in range(0, n, block_cols):
        i1 = min(i0 + block_cols, n)
        block = (-C_scaled[i0:i1, :]) + logu[i0:i1].unsqueeze(1) + logv.unsqueeze(0)
        Q[i0:i1, :] = block.exp().to(dtype)
    Q /= Q.sum()
    return Q.to(torch.float32)


@torch.no_grad()
def compute_Q_and_layer_costs(
    X_list: List[torch.Tensor],
    Y_list: List[torch.Tensor],
    reg_inner: float = 0.1,
    device: str = "cuda",
    dtype: Union[str, torch.dtype] = torch.float32,
    verbose: bool = True,
) -> Tuple[List[List[torch.Tensor]], torch.Tensor, Dict]:
    """
    对每一对 (ℓ,m)：
      1. 计算相关距离矩阵 C_inner
      2. Sinkhorn 得到 Q_{ℓm}
      3. cost = ⟨C_inner, Q_{ℓm}⟩
    """
    L = len(X_list)
    M = len(Y_list)
    Q_list = [[None for _ in range(M)] for _ in range(L)]
    C_layer = torch.empty((L, M), device=device, dtype=torch.float32)

    for i in range(L):
        for j in range(M):
            X = X_list[i].to(device, dtype=torch.float32)
            Y = Y_list[j].to(device, dtype=torch.float32)
            C_inner = corr_distance_matrix(X, Y)
            C_inner = torch.nan_to_num(C_inner, nan=1.0, posinf=2.0, neginf=0.0)

            a = torch.full((C_inner.shape[0],), 1.0 / C_inner.shape[0], device=device)
            b = torch.full((C_inner.shape[1],), 1.0 / C_inner.shape[1], device=device)

            Q = sinkhorn_uniform_streaming(C_inner, a, b, reg=reg_inner)
            cost_val = (C_inner * Q).sum()
            Q_list[i][j] = Q.detach().cpu()
            C_layer[i, j] = cost_val

            if verbose and (i == 0 or j == 0):
                print(f"[Layer {i},{j}] cost={cost_val.item():.4f}")

    aux = {
        "L": L, "M": M, "reg_inner": reg_inner, "device": device,
        "dtype": str(dtype).replace("torch.", "")
    }
    return Q_list, C_layer, aux


# ============================================================
# 外层：稳定 Sinkhorn（log-domain + 自适应 eps）
# ============================================================

@torch.no_grad()
def normalize_cost(
    C: torch.Tensor,
    method: str = "median",
    scale_hint: float = 1.0,
) -> torch.Tensor:
    """数值稳化但不改变相对大小。"""
    C = C.to(torch.float64)
    if method == "median":
        C_min = _safe_min(C, default=0.0)
        C_shift = C - C_min
        med = _safe_median(C_shift, default=1.0)
        med = torch.clamp(med, min=1e-12)
        return (C_shift / med * scale_hint).to(torch.float32)
    elif method == "per_token_feature":
        mean = torch.clamp(_safe_mean(C, default=1.0), min=1e-12)
        return (C / mean * scale_hint).to(torch.float32)
    elif method == "min_shift":
        C_min = _safe_min(C, default=0.0)
        return (C - C_min).to(torch.float32)
    else:
        return C.to(torch.float32)


@torch.no_grad()
def compute_P_stable(
    C_layer: torch.Tensor,
    reg_outer: float = 0.5,
    max_iters: int = 1000,
    tol: float = 1e-7,
    normalize: str = "median",
    max_eps_grow: int = 6,
    clip_large: Optional[float] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """外层稳定 Sinkhorn（log-domain，自适应 ε）。"""
    L, M = C_layer.shape
    C = C_layer.clone()
    C[~torch.isfinite(C)] = torch.tensor(float("inf"), dtype=C.dtype, device=C.device)
    if clip_large is not None:
        C = torch.clamp(C, max=clip_large)
    if normalize and normalize.lower() != "none":
        C = normalize_cost(C, method=normalize)

    r = torch.full((L,), 1.0 / L, dtype=torch.float64, device=C.device)
    c = torch.full((M,), 1.0 / M, dtype=torch.float64, device=C.device)
    logr = torch.log(r)
    logc = torch.log(c)

    def sinkhorn_logspace(eps: float) -> Optional[torch.Tensor]:
        logK = (-C.to(torch.float64)) / max(eps, 1e-12)
        logu = torch.zeros_like(logr)
        logv = torch.zeros_like(logc)

        for it in range(max_iters):
            A = logK + logv.unsqueeze(0)
            logu_new = logr - torch.logsumexp(A, dim=1)
            B = logK.T + logu_new.unsqueeze(0)
            logv_new = logc - torch.logsumexp(B, dim=1)
            if torch.max(torch.abs(logu_new - logu)).item() < tol:
                break
            logu, logv = logu_new, logv_new

        P_log = logK + logu.unsqueeze(1) + logv.unsqueeze(0)
        P = torch.exp(P_log)
        if torch.isfinite(P).all():
            return P.to(torch.float32)
        return None

    eps = max(float(reg_outer), 1e-12)
    for grow in range(max_eps_grow + 1):
        P_try = sinkhorn_logspace(eps)
        if P_try is not None:
            P_try /= P_try.sum()
            # 修正边际
            for _ in range(3):
                P_try = P_try * (1.0 / L) / (P_try.sum(dim=1, keepdim=True) + 1e-30)
                P_try = P_try * (1.0 / M) / (P_try.sum(dim=0, keepdim=True) + 1e-30)
            if verbose:
                print(f"[HOT] P ok, eps={eps:.3g}")
            return P_try
        eps *= 2.0
        if verbose:
            print(f"[HOT] Retry Sinkhorn with larger eps={eps:.3g}")
    return torch.zeros((L, M), dtype=torch.float32, device=C.device)


# ============================================================
# 重建阶段
# ============================================================

@torch.no_grad()
def reconstruct_X(
    P: torch.Tensor,
    Q_list: List[List[torch.Tensor]],
    Y_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """用 P 和 Q 重构 \hat X。"""
    L, M = P.shape
    device = P.device
    Xhat_list: List[torch.Tensor] = []
    for li in range(L):
        accum: Optional[torch.Tensor] = None
        for mj in range(M):
            Y_m = Y_list[mj]
            Q_lm = Q_list[li][mj]
            term = P[li, mj] * (Y_m @ Q_lm.T)
            accum = term if accum is None else (accum + term)
        Xhat = accum
        Xhat_list.append(Xhat.to(device))
    return Xhat_list
