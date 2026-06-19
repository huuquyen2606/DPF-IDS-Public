"""Stage-2 server prototype aggregation."""

from __future__ import annotations

import torch

def stage2_global_aggregation(
    benign_cluster_protos,
    safe_counts, 
):
    """
    Tính prototype toàn cục ở tầng 2 từ các prototype cụm benign.

    Mỗi phần tử trong `benign_cluster_protos` chính là p_k trong công thức:
        mu_global = arg min_mu sum_k beta_k * ||p_k - mu||_1

    Trong đó beta_k không cần truyền từ ngoài vào.
    Server sẽ tự tính beta_k từ `safe_counts`.
    """
    if not benign_cluster_protos:
        raise ValueError("benign_cluster_protos must not be empty.")

    if safe_counts is None:
        raise ValueError("safe_counts must not be None.") # tính beta_k

    if len(benign_cluster_protos) != len(safe_counts):
        raise ValueError("safe_counts must have the same length as benign_cluster_protos.")

    def weighted_median_1d(values, weights): # tính weighted median cho 1 chiều
        order = torch.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]

        total_weight = float(sorted_weights.sum().item())
        if total_weight <= 0.0:
            raise ValueError("weights must have positive total sum.")

        cumulative = torch.cumsum(sorted_weights, dim=0)
        threshold = 0.5 * total_weight

        idx = int(torch.searchsorted(
            cumulative,
            torch.tensor(threshold, dtype=sorted_weights.dtype)
        ).item())
        idx = min(idx, sorted_values.numel() - 1)

        return float(sorted_values[idx].item())

    prototype_matrix = torch.stack(
        [proto.detach().cpu().view(-1).float() for proto in benign_cluster_protos],
        dim=0,
    )

    # Tính beta_k từ safe_counts: beta_k = safe_counts[k] / sum(safe_counts)
    beta_tensor = torch.tensor(safe_counts, dtype=prototype_matrix.dtype)
    if float(beta_tensor.sum().item()) <= 0.0:
        raise ValueError("safe_counts must have positive total sum.")

    beta_tensor = beta_tensor / beta_tensor.sum()

    # lấy arg min theo L1 distance với trọng số beta_k chính là weighted median trên từng chiều
    mu_coordinates = [
        weighted_median_1d(prototype_matrix[:, dim], beta_tensor)
        for dim in range(prototype_matrix.shape[1])
    ]
    mu_global = torch.tensor(mu_coordinates, dtype=prototype_matrix.dtype)

    return mu_global
