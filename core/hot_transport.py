# Transport plan computation module
import math
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn.functional as F


def _zscore_along_time(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute z-score along time/stimulus dimension (ddof=1) for correlation computation.
    Automatically handles CPU half/bfloat16 -> float32 conversion.
    
    Args:
        X: [T, N] tensor
        eps: small epsilon for numerical stability
        
    Returns:
        Z-scored tensor [T, N]
    """
    # On CPU, float16/bfloat16 don't have sqrt operations
    if X.device.type == "cpu" and X.dtype in (torch.float16, torch.bfloat16):
        X = X.to(torch.float32)

    T = X.size(0)
    mean = X.mean(dim=0, keepdim=True)
    var = ((X - mean) ** 2).sum(dim=0, keepdim=True) / max(T - 1, 1)

    eps_t = torch.tensor(eps, dtype=X.dtype, device=X.device)
    std = torch.sqrt(var + eps_t)
    return (X - mean) / std


def corr_distance_matrix(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Correlation distance matrix C_inner[lm]: [n_x, n_y], where X:[T,n_x], Y:[T,n_y]
    d_corr(x,y) = 1 - rho(x,y), where rho is Pearson correlation (ddof=1)
    
    Args:
        X: [T, n_x] tensor
        Y: [T, n_y] tensor
        eps: small epsilon for numerical stability
        
    Returns:
        Correlation distance matrix [n_x, n_y]
    """
    assert X.dim() == 2 and Y.dim() == 2 and X.size(0) == Y.size(0), \
        "X, Y must be [T, N] with same T"

    # CPU half/bfloat16 -> float32
    def _safe_cast(Z: torch.Tensor) -> torch.Tensor:
        return Z.to(torch.float32) if (Z.device.type == "cpu" and Z.dtype in (torch.float16, torch.bfloat16)) else Z

    X = _safe_cast(X)
    Y = _safe_cast(Y)

    T = X.size(0)
    ZX = _zscore_along_time(X, eps=eps)  # [T, n_x]
    ZY = _zscore_along_time(Y, eps=eps)  # [T, n_y]

    rho = (ZX.transpose(0, 1) @ ZY) / max(T - 1, 1)   # [n_x, n_y]
    rho = torch.clamp(rho, -1.0, 1.0)
    return 1.0 - rho


import torch
from torch import Tensor

@torch.no_grad()
def sinkhorn_uniform_streaming(
    C: Tensor, a: Tensor, b: Tensor, reg: float,
    n_iters: int = 200, tol: float = 1e-6,
    block_cols: int = 2048, block_rows: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """
    Memory-efficient Sinkhorn for uniform marginals.
    
    Args:
        C: [n, m] cost matrix (on device)
        a: [n] marginal (sum=1)
        b: [m] marginal (sum=1)
        reg: regularization parameter
        n_iters: maximum iterations
        tol: tolerance for convergence
        block_cols: block size for column-wise operations
        block_rows: block size for row-wise operations
        dtype: dtype for storing Q
        
    Returns:
        Q [n, m] transport plan without materializing K = exp(-C/reg)
    """
    assert C.dim() == 2
    n, m = C.shape
    if block_rows is None:
        block_rows = block_cols

    # Fallback if bfloat16 is not supported
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16

    # log-domain variables
    logu = torch.zeros(n, device=C.device, dtype=torch.float32)
    logv = torch.zeros(m, device=C.device, dtype=torch.float32)
    loga = a.to(torch.float32).clamp_min(1e-30).log()
    logb = b.to(torch.float32).clamp_min(1e-30).log()

    # Use fp32 for computation, low precision for storage
    C_scaled = (C.to(dtype) / reg).to(torch.float32)

    def log_Kv(logv_: Tensor) -> Tensor:
        """log((K @ v))[i] = logsum_j exp(-C[i,j]/reg + logv[j])"""
        out = torch.full((n,), -float('inf'), device=C.device, dtype=torch.float32)
        for j0 in range(0, m, block_cols):
            j1 = min(j0 + block_cols, m)
            block = (-C_scaled[:, j0:j1]) + logv_[j0:j1]  # [n, j]
            out = torch.logaddexp(out, torch.logsumexp(block, dim=1))
        return out

    def log_KTu(logu_: Tensor) -> Tensor:
        """log((K^T @ u))[j] = logsum_i exp(-C[i,j]/reg + logu[i])"""
        out = torch.full((m,), -float('inf'), device=C.device, dtype=torch.float32)
        for i0 in range(0, n, block_rows):
            i1 = min(i0 + block_rows, n)
            block = (-C_scaled[i0:i1, :]) + logu_[i0:i1].unsqueeze(1)  # [i, m]
            out = torch.logaddexp(out, torch.logsumexp(block, dim=0))
        return out

    prev_err = None
    for t in range(n_iters):
        # u = a / (K @ v)
        logu = loga - log_Kv(logv)
        # v = b / (K^T @ u)
        logv = logb - log_KTu(logu)

        if (t % 5 == 0) or (t + 1 == n_iters):
            # Monitor row marginal error
            lKv = log_Kv(logv)
            row_marg = (logu + lKv).exp()
            err = (row_marg - a).abs().sum().item()
            if prev_err is not None and err < tol:
                break
            prev_err = err

    # Generate Q (block-wise) to avoid materializing K at once
    Q = torch.empty((n, m), device=C.device, dtype=dtype)
    for i0 in range(0, n, block_rows):
        i1 = min(i0 + block_rows, n)
        block = (-C_scaled[i0:i1, :]) + logu[i0:i1].unsqueeze(1) + logv.unsqueeze(0)
        Q[i0:i1, :] = block.exp().to(dtype)

    Q /= Q.sum()
    return Q


def sinkhorn_uniform(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-9,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Entropy-regularized OT: argmin_Q <C,Q> + reg * KL(Q || a b^T)
    Sinkhorn-Knopp algorithm.
    
    Args:
        C: [n, m] cost matrix
        a: [n] marginal (sum=1)
        b: [m] marginal (sum=1)
        reg: regularization parameter
        max_iter: maximum iterations
        tol: tolerance for convergence
        eps: small epsilon for numerical stability
        
    Returns:
        Q [n, m] transport plan
    """
    assert C.dim() == 2 and a.dim() == 1 and b.dim() == 1
    n, m = C.shape
    assert a.numel() == n and b.numel() == m
    assert abs(a.sum().item() - 1.0) < 1e-6 and abs(b.sum().item() - 1.0) < 1e-6

    device = C.device
    C = C - C.min()
    K = torch.exp(-C / max(reg, 1e-6)).to(device) + eps

    u = torch.ones_like(a) / n
    v = torch.ones_like(b) / m

    for _ in range(max_iter):
        Kv = K @ v + eps
        u = a / Kv
        KTu = K.transpose(0, 1) @ u + eps
        v = b / KTu

        if _ % 10 == 0:
            Q = (u.unsqueeze(1) * K) * v.unsqueeze(0)
            err = (Q.sum(dim=1) - a).abs().mean() + (Q.sum(dim=0) - b).abs().mean()
            if err.item() < tol:
                break

        if not torch.isfinite(u).all() or not torch.isfinite(v).all():
            raise FloatingPointError("Sinkhorn numerical divergence; increase reg or normalize cost matrix.")

    Q = (u.unsqueeze(1) * K) * v.unsqueeze(0)
    return Q


@torch.no_grad()
def compute_Q_and_layer_costs(
    X_list: List[torch.Tensor],
    Y_list: List[torch.Tensor],
    reg_inner: float = 0.1,
    device: str = "cuda",
    sinkhorn_max_iter: int = 1000,
    sinkhorn_tol: float = 1e-9,
) -> Tuple[List[List[torch.Tensor]], torch.Tensor, List[List[torch.Tensor]]]:
    """
    Compute for all layer pairs (l,m):
      Q_{lm}, C_layer[l,m], C_inner_{lm}
    
    Args:
        X_list: List of [T, n_x] activation tensors for model A layers
        Y_list: List of [T, n_y] activation tensors for model B layers
        reg_inner: regularization for inner-level OT
        device: device to use
        sinkhorn_max_iter: maximum iterations for Sinkhorn
        sinkhorn_tol: tolerance for Sinkhorn convergence
        
    Returns:
        Q_list: List of lists of transport plans Q_{lm}
        C_layer: [L, M] layer-level cost matrix
        C_inner_list: List of lists of inner cost matrices
    """
    L, M = len(X_list), len(Y_list)
    X_list = [x.to(device, dtype=torch.float32) for x in X_list]
    Y_list = [y.to(device, dtype=torch.float32) for y in Y_list]
    Q_list = [[None for _ in range(M)] for _ in range(L)]
    C_inner_list = [[None for _ in range(M)] for _ in range(L)]
    C_layer = torch.empty((L, M), device=device)

    for li, X in enumerate(X_list):
        T, n_x = X.shape
        a = torch.full((n_x,), 1.0 / n_x, device=device)
        for mj, Y in enumerate(Y_list):
            _, n_y = Y.shape
            b = torch.full((n_y,), 1.0 / n_y, device=device)

            # 1) Inner-level correlation distance
            C_inner = corr_distance_matrix(X, Y)
            C_inner = torch.nan_to_num(C_inner, nan=1.0, posinf=2.0, neginf=0.0)

            # 2) Sinkhorn to get Q_{lm}
            Q_lm = sinkhorn_uniform_streaming(
                C_inner, a, b, reg=reg_inner,
                n_iters=200, tol=1e-6, block_cols=2048, dtype=torch.bfloat16
            )

            # 3) Cost
            cost_val = (C_inner * Q_lm).sum()

            Q_list[li][mj] = Q_lm
            C_inner_list[li][mj] = C_inner
            C_layer[li, mj] = cost_val

    return Q_list, C_layer, C_inner_list


@torch.no_grad()
def compute_P(
    C_layer: torch.Tensor,
    reg_outer: float = 0.1,
    sinkhorn_max_iter: int = 1000,
    sinkhorn_tol: float = 1e-9,
) -> torch.Tensor:
    """
    Solve outer-level OT to get layer coupling matrix P.
    
    Args:
        C_layer: [L, M] layer-level cost matrix
        reg_outer: regularization for outer-level OT
        sinkhorn_max_iter: maximum iterations
        sinkhorn_tol: tolerance for convergence
        
    Returns:
        P [L, M] layer coupling matrix
    """
    device = C_layer.device
    L, M = C_layer.shape
    a = torch.full((L,), 1.0 / L, device=device)
    b = torch.full((M,), 1.0 / M, device=device)

    P = sinkhorn_uniform(C_layer, a, b, reg=reg_outer,
                         max_iter=sinkhorn_max_iter, tol=sinkhorn_tol)
    return P


@torch.no_grad()
def reconstruct_X(
    P: torch.Tensor,
    Q_list: List[List[torch.Tensor]],
    Y_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Reconstruct \hat X using P and Q.
    
    Args:
        P: [L, M] layer coupling matrix
        Q_list: List of lists of transport plans Q_{lm}
        Y_list: List of [T, n_y] activation tensors for model B layers
        
    Returns:
        Xhat_list: List of reconstructed [T, n_x] activation tensors
    """
    L, M = P.shape
    device = P.device
    Xhat_list: List[torch.Tensor] = []
    for li in range(L):
        accum: Optional[torch.Tensor] = None
        for mj in range(M):
            Y_m = Y_list[mj]
            Q_lm = Q_list[li][mj]
            term = P[li, mj] * (Y_m @ Q_lm.transpose(0, 1))
            accum = term if accum is None else (accum + term)
        Xhat = L * accum
        Xhat_list.append(Xhat.to(device))
    return Xhat_list


# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T = 256
    X_list = [torch.randn(T, 64, device=device, dtype=torch.float32),
              torch.randn(T, 96, device=device, dtype=torch.float32),
              torch.randn(T, 128, device=device, dtype=torch.float32)]
    Y_list = [torch.randn(T, 80, device=device, dtype=torch.float32),
              torch.randn(T, 96, device=device, dtype=torch.float32)]

    Q_list, C_layer, C_inner_list = compute_Q_and_layer_costs(X_list, Y_list, reg_inner=0.1)
    P = compute_P(C_layer, reg_outer=0.1)
    Xhat_list = reconstruct_X(P, Q_list, Y_list)

    print("C_layer shape:", C_layer.shape)
    print("P row sums:", P.sum(dim=1))
    print("P col sums:", P.sum(dim=0))
