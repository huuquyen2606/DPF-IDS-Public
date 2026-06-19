"""Prototype math and robust thresholds for DPF-IDS."""

from __future__ import annotations

import numpy as np
import torch

def extract_prototype_limited(model, dataloader, device, max_batches=None):
    """
    Extract mean hidden representation from up to max_batches.
    If max_batches is None, use the full dataloader.
    """
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, _ = batch
            inputs = inputs.to(device)
            hidden_output, _ = model(inputs)
            all_hidden.append(hidden_output.detach())
            if max_batches is not None and (batch_idx + 1) >= int(max_batches):
                break

    if not all_hidden:
        raise RuntimeError("Cannot extract prototype from an empty dataloader.")

    return torch.mean(torch.cat(all_hidden, dim=0), dim=0).detach().cpu()


def _as_proto_vec(proto):
    """Convert a prototype-like tensor to a detached 1-D CPU float tensor."""
    if proto is None:
        raise ValueError("Prototype must not be None")
    return proto.detach().cpu().view(-1).float()


def stack_prototypes(client_prototypes, client_ids=None):
    """Stack prototypes into a matrix with shape [num_clients, proto_dim]."""
    if client_ids is None:
        client_ids = sorted(client_prototypes.keys())
    vectors = [_as_proto_vec(client_prototypes[cid]) for cid in client_ids]
    if not vectors:
        raise ValueError("No prototypes to stack")
    return torch.stack(vectors, dim=0)


def coordinate_wise_median(prototypes, client_ids=None):
    """Robust reference prototype: median per coordinate.

    Accepts either:
    - dict {client_id: prototype_tensor}
    - list/tuple of prototype tensors
    """
    if isinstance(prototypes, dict):
        matrix = stack_prototypes(prototypes, client_ids=client_ids)
    else:
        vectors = [_as_proto_vec(p) for p in prototypes]
        if not vectors:
            raise ValueError("No prototypes provided")
        matrix = torch.stack(vectors, dim=0)
    return torch.median(matrix, dim=0).values


def compute_distance(proto1, proto2, metric="euclidean", eps=1e-12):
    """Distance between two prototype vectors.

    Supported metrics:
    - euclidean: magnitude-sensitive L2 distance
    - cosine: 1 - cosine similarity, direction-sensitive
    - normalized_euclidean: L2 distance after unit normalization
    """
    p1 = _as_proto_vec(proto1)
    p2 = _as_proto_vec(proto2)

    if metric == "euclidean":
        return torch.norm(p1 - p2, p=2).item()

    if metric == "cosine":
        n1 = torch.norm(p1, p=2)
        n2 = torch.norm(p2, p=2)
        denom = n1 * n2
        if denom.item() < eps:
            return 0.0
        cos_sim = torch.dot(p1, p2) / (denom + eps)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        return float(1.0 - cos_sim.item())

    if metric == "normalized_euclidean":
        n1 = torch.norm(p1, p=2)
        n2 = torch.norm(p2, p=2)
        if n1.item() < eps or n2.item() < eps:
            return 0.0
        p1 = p1 / (n1 + eps)
        p2 = p2 / (n2 + eps)
        return torch.norm(p1 - p2, p=2).item()

    raise ValueError("metric must be 'euclidean', 'cosine', or 'normalized_euclidean'")


def robust_upper_threshold(values, tau=3.0, eps=1e-12):
    """One-sided robust upper threshold: median + tau * 1.4826 * MAD.

    If MAD is zero, fall back to IQR. If IQR is also zero, the threshold
    equals the maximum observed value, so no point is flagged only because
    all values are identical.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("inf"), {
            "median": np.nan,
            "mad": np.nan,
            "scale": np.nan,
            "method": "empty",
            "tau": float(tau),
        }

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    method = "mad"

    if scale < eps:
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = float(q3 - q1)
        if iqr >= eps:
            scale = iqr / 1.349
            method = "iqr"
        else:
            threshold = float(np.max(arr))
            return threshold, {
                "median": med,
                "mad": mad,
                "scale": 0.0,
                "method": "constant",
                "tau": float(tau),
            }

    threshold = med + float(tau) * scale
    return float(threshold), {
        "median": med,
        "mad": mad,
        "scale": float(scale),
        "method": method,
        "tau": float(tau),
    }


def robust_z_score(value, stats, eps=1e-12):
    """Robust z-score using the stats returned by robust_upper_threshold."""
    scale = float(stats.get("scale", 0.0) or 0.0)
    if scale < eps:
        return 0.0
    return float((float(value) - float(stats.get("median", 0.0))) / (scale + eps))


def decayed_tau(round_idx, total_rounds, tau_start=3.0, tau_min=1.0, beta=2.0):
    """Round-dependent tau used to make detection stricter over time.

    tau starts near tau_start and decays toward tau_min. The threshold itself
    is still data-adaptive, but the multiplier becomes smaller across rounds.
    """
    if total_rounds <= 1:
        progress = 1.0
    else:
        progress = float(round_idx) / max(float(total_rounds - 1), 1.0)

    tau_t = float(tau_min) + (float(tau_start) - float(tau_min)) * np.exp(-float(beta) * progress)
    return float(tau_t)


def compute_decayed_threshold_from_scores(
    values,
    metric_name,
    round_idx,
    total_rounds,
    tau_start=3.0,
    tau_min=1.0,
    beta=2.0,
    eps=1e-12,
):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        stats = {
            "method": "empty",
            "median": 0.0,
            "mad": 0.0,
            "tau": 0.0,
            "threshold": float("inf"),
        }
        return float("inf"), stats

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))

    if total_rounds <= 1:
        progress = 1.0
    else:
        progress = round_idx / max(total_rounds - 1, 1)

    tau_t = tau_min + (tau_start - tau_min) * np.exp(-beta * progress)

    threshold = med + tau_t * max(mad, eps)

    stats = {
        "method": "decayed_median_mad",
        "median": float(med),
        "mad": float(mad),
        "tau": float(tau_t),
        "threshold": float(threshold),
    }

    return float(threshold), stats


def select_closest_prototype(client_prototypes, reference_prototype, metric="euclidean"):
    """Select the client whose prototype is closest to a reference prototype."""
    if not client_prototypes:
        raise ValueError("client_prototypes is empty")

    scores = {
        cid: compute_distance(proto, reference_prototype, metric=metric)
        for cid, proto in client_prototypes.items()
    }
    best_client_id = min(scores, key=scores.get)
    return best_client_id


def compute_lambda_g_robust(client_prototypes, metric="cosine", tau=2.5, min_lambda=1e-6, verbose=True):
    """Compute lambda_g every round from nearest-neighbor prototype distances.

    This replaces the old max-min round-1-only lambda_g. The goal is to
    avoid path dependence: if round 1 produces a bad lambda_g, later rounds
    are not forced to reuse it.
    """
    if not client_prototypes or len(client_prototypes) < 2:
        raise ValueError("Need at least 2 client prototypes to compute lambda_g")

    client_ids = sorted(client_prototypes.keys())
    nn_distances = {}

    for cid in client_ids:
        distances = []
        for other_id in client_ids:
            if other_id == cid:
                continue
            distances.append(
                compute_distance(
                    client_prototypes[cid],
                    client_prototypes[other_id],
                    metric=metric,
                )
            )
        nn_distances[cid] = float(min(distances))

    values = list(nn_distances.values())
    lambda_g, stats = robust_upper_threshold(values, tau=tau)

    # A zero lambda_g makes clustering too brittle. Keep a small positive floor.
    if not np.isfinite(lambda_g) or lambda_g <= 0.0:
        lambda_g = float(max(np.quantile(values, 0.75), min_lambda))
    else:
        lambda_g = float(max(lambda_g, min_lambda))

    if verbose:
        print(
            f"lambda_g robust ({metric}) = {lambda_g:.6f} | "
            f"median NN={stats['median']:.6f}, scale={stats['scale']:.6f}, method={stats['method']}"
        )

    return lambda_g, {
        "nearest_neighbor_distances": nn_distances,
        "stats": stats,
        "metric": metric,
    }


def compute_lambda_g(client_prototypes, metric="euclidean", k=None):
    """
    Old max-min lambda_g. Kept only for comparison/ablation.
    Prefer compute_lambda_g_robust(...) in the refactored framework.
    """
    if not client_prototypes or len(client_prototypes) < 2:
        raise ValueError("Need at least 2 client prototypes")

    client_ids = list(client_prototypes.keys())
    n_clients = len(client_ids)

    if k is None:
        k = n_clients
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, n_clients)

    mean_proto = torch.mean(stack_prototypes(client_prototypes, client_ids), dim=0)
    T_vectors = [mean_proto]
    remaining = set(client_ids)
    selected_order = []
    lambda_g = 0.0

    for step in range(1, k + 1):
        chosen_id = None
        chosen_min_dist = float("-inf")

        for cid in remaining:
            client_vec = _as_proto_vec(client_prototypes[cid])
            min_dist_to_T = min(
                compute_distance(client_vec, t_vec, metric=metric)
                for t_vec in T_vectors
            )
            if min_dist_to_T > chosen_min_dist:
                chosen_min_dist = min_dist_to_T
                chosen_id = cid

        selected_order.append(chosen_id)
        T_vectors.append(_as_proto_vec(client_prototypes[chosen_id]))
        remaining.remove(chosen_id)
        lambda_g = chosen_min_dist

    print(f"[Ablation] Old Lambda_g at step K={k}: {lambda_g:.4f}")
    return lambda_g, selected_order
