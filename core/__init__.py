# Core transport plan computation implementation
from .hot_transport import (
    corr_distance_matrix,
    sinkhorn_uniform,
    sinkhorn_uniform_streaming,
    compute_Q_and_layer_costs,
    compute_P,
    reconstruct_X,
)

__all__ = [
    'corr_distance_matrix',
    'sinkhorn_uniform',
    'sinkhorn_uniform_streaming',
    'compute_Q_and_layer_costs',
    'compute_P',
    'reconstruct_X',
]
