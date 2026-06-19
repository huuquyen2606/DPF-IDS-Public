"""Prototype-only poisoned client detection."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from .prototype import (
    _as_proto_vec,
    compute_distance,
    coordinate_wise_median,
    decayed_tau,
    robust_upper_threshold,
)

def _safe_cosine_distance(vec1, vec2, eps=1e-12):
    v1 = _as_proto_vec(vec1)
    v2 = _as_proto_vec(vec2)
    n1 = torch.norm(v1, p=2)
    n2 = torch.norm(v2, p=2)
    denom = n1 * n2
    if denom.item() < eps:
        return 0.0
    cos_sim = torch.dot(v1, v2) / (denom + eps)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    return float(1.0 - cos_sim.item())


def _build_cluster_lookup(clusters):
    lookup = {}
    for cluster_id, members in clusters.items():
        for cid in members:
            lookup[cid] = cluster_id
    return lookup


def _decayed_robust_threshold(values, metric_name, round_idx, total_rounds, tau_start, tau_min, beta):
    """Data-adaptive threshold with decayed tau, without monotone hard capping.

    This keeps the detector stricter across rounds through tau(t), but avoids the
    earlier failure mode where a near-zero lambda in an early round permanently
    capped later thresholds.
    """
    tau_t = decayed_tau(
        round_idx=round_idx,
        total_rounds=total_rounds,
        tau_start=tau_start,
        tau_min=tau_min,
        beta=beta,
    )
    threshold, stats = robust_upper_threshold(values, tau=tau_t)
    stats = dict(stats)
    stats["metric"] = metric_name
    stats["threshold"] = float(threshold)
    stats["raw_threshold"] = float(threshold)
    stats["tau"] = float(tau_t)
    return float(threshold), stats


def _robust_z_from_stats(value, stats, eps=1e-12):
    """Robust z-score compatible with both MAD/IQR stats and simple stats."""
    scale = float(stats.get("scale", 0.0) or 0.0)
    if scale < eps:
        mad = float(stats.get("mad", 0.0) or 0.0)
        scale = 1.4826 * mad
    if scale < eps:
        return 0.0
    return float((float(value) - float(stats.get("median", 0.0))) / (scale + eps))


def detect_poisoned_clients_prototype_only(
    client_prototypes,
    clusters,
    prev_client_prototypes=None,
    round_idx=0,
    total_rounds=1,
    threshold_state=None,  # kept for checkpoint compatibility; not used for monotone capping anymore
    tau_E_start=3.0,
    tau_C_start=3.0,
    tau_D_start=3.0,
    tau_N_start=3.0,
    tau_E_min=1.0,
    tau_C_min=1.0,
    tau_D_min=1.0,
    tau_N_min=1.0,
    lambda_decay_beta=2.0,
    lambda_min_ratio=0.20,  # kept for API compatibility; not used by decayed thresholds
    vote_threshold=2,
    min_cluster_size_for_local_threshold=4,  # kept for API compatibility
    cluster_contamination_ratio=0.5,
    enable_cluster_centroid_filter=True,
    eps=1e-12,
    verbose=True,
):
    """Detect poisoned clients and poisoned clusters using prototype-only signals.

    Server-side security constraint:
        The Global Server receives only prototypes. It does not use client loss,
        gradients, model parameters, or raw data.

    Two anomaly views are computed by the Global Server:
        1. Local anomaly:  client prototype vs. its own cluster reference.
        2. Global anomaly: client prototype vs. whole-population robust reference.

    In addition, the server can remove an entire cluster if the cluster centroid
    / robust cluster reference is globally abnormal:
        cluster_ref_k too far from global_ref => all clients in cluster k are removed.

    Practical decision rule:
        - dE/dN are used as the main decision signals for PGA with fixed gamma,
          because previous diagnostics showed dC/dD were noisy.
        - dC/dD are still logged for debugging but do not drive the main decision.
        - A client is marked poisoned if local/global magnitude evidence is strong.
        - If a cluster centroid exceeds lambda_cluster_E_global, every member is
          marked poisoned for the current round only.

    There is no accumulated risk score. A client flagged in one round is checked
    again from scratch in the next round.
    """
    if not client_prototypes:
        raise ValueError("client_prototypes is empty")
    if not clusters:
        raise ValueError("clusters is empty")
    if threshold_state is None:
        threshold_state = {}

    prev_client_prototypes = prev_client_prototypes or {}
    client_ids = sorted(client_prototypes.keys())
    cluster_lookup = _build_cluster_lookup(clusters)

    # Whole-population robust reference used for global anomaly.
    global_ref = coordinate_wise_median(client_prototypes)
    global_norms = [torch.norm(_as_proto_vec(client_prototypes[cid]), p=2).item() for cid in client_ids]
    global_norm_ref = float(np.median(global_norms)) if global_norms else 0.0

    # Cluster robust references used for local anomaly.
    cluster_refs = {}
    cluster_norm_refs = {}
    for cluster_id, members in clusters.items():
        valid_members = [cid for cid in members if cid in client_prototypes]
        if not valid_members:
            continue
        cluster_refs[cluster_id] = coordinate_wise_median(client_prototypes, client_ids=valid_members)
        norms = [torch.norm(_as_proto_vec(client_prototypes[cid]), p=2).item() for cid in valid_members]
        cluster_norm_refs[cluster_id] = float(np.median(norms)) if norms else 0.0

    # Prototype drift references are logged only. They are not used for the main decision by default.
    deltas = {}
    for cid in client_ids:
        if cid in prev_client_prototypes:
            deltas[cid] = _as_proto_vec(client_prototypes[cid]) - _as_proto_vec(prev_client_prototypes[cid])

    global_delta_ref = coordinate_wise_median(list(deltas.values())) if deltas else None
    cluster_delta_refs = {}
    for cluster_id, members in clusters.items():
        valid_delta_members = [cid for cid in members if cid in deltas]
        if valid_delta_members:
            cluster_delta_refs[cluster_id] = coordinate_wise_median([deltas[cid] for cid in valid_delta_members])

    # Build client-level raw signal table.
    per_client = {}
    for cid in client_ids:
        cluster_id = cluster_lookup.get(cid)
        cluster_ref = cluster_refs.get(cluster_id, global_ref)
        proto = client_prototypes[cid]

        # Local anomaly: client vs its own cluster.
        dE_local = compute_distance(proto, cluster_ref, metric="euclidean")
        dC_local = compute_distance(proto, cluster_ref, metric="cosine")

        # Global anomaly: client vs whole population.
        dE_global = compute_distance(proto, global_ref, metric="euclidean")
        dC_global = compute_distance(proto, global_ref, metric="cosine")

        # Drift anomaly, logged for debugging only.
        if cid in deltas:
            delta_ref = cluster_delta_refs.get(cluster_id, global_delta_ref)
            if delta_ref is None:
                dD = 0.0
            else:
                delta_ref_norm = torch.norm(_as_proto_vec(delta_ref), p=2).item()
                delta_i_norm = torch.norm(_as_proto_vec(deltas[cid]), p=2).item()
                if delta_ref_norm < 1e-8 or delta_i_norm < 1e-8:
                    dD = 0.0
                else:
                    dD = _safe_cosine_distance(deltas[cid], delta_ref)
        else:
            dD = 0.0

        norm_i = torch.norm(_as_proto_vec(proto), p=2).item()
        norm_ref_local = cluster_norm_refs.get(cluster_id, norm_i)

        dN_local = abs(math.log((norm_i + eps) / (norm_ref_local + eps)))
        dN_global = abs(math.log((norm_i + eps) / (global_norm_ref + eps)))

        per_client[cid] = {
            "client_id": int(cid),
            "cluster_id": int(cluster_id) if cluster_id is not None else -1,
            "dE_local": float(dE_local),
            "dC_local": float(dC_local),
            "dN_local": float(dN_local),
            "dE_global": float(dE_global),
            "dC_global": float(dC_global),
            "dN_global": float(dN_global),
            "dD": float(dD),
            "norm": float(norm_i),
        }

        # Backward-compatible aliases for older analysis scripts.
        per_client[cid]["dE"] = float(dE_local)
        per_client[cid]["dC"] = float(dC_local)
        per_client[cid]["dN"] = float(dN_local)

    # Decision signals. Keep cosine/drift out of main decision because they were noisy in diagnostics.
    decision_signal_keys = ["dE_local", "dN_local", "dE_global", "dN_global"]
    debug_signal_keys = ["dC_local", "dC_global", "dD"]
    all_signal_keys = decision_signal_keys + debug_signal_keys

    tau_start_map = {
        "dE_local": tau_E_start,
        "dE_global": tau_E_start,
        "cluster_dE_global": tau_E_start,
        "dC_local": tau_C_start,
        "dC_global": tau_C_start,
        "dD": tau_D_start,
        "dN_local": tau_N_start,
        "dN_global": tau_N_start,
        "cluster_dN_global": tau_N_start,
    }
    tau_min_map = {
        "dE_local": tau_E_min,
        "dE_global": tau_E_min,
        "cluster_dE_global": tau_E_min,
        "dC_local": tau_C_min,
        "dC_global": tau_C_min,
        "dD": tau_D_min,
        "dN_local": tau_N_min,
        "dN_global": tau_N_min,
        "cluster_dN_global": tau_N_min,
    }

    # Client-level thresholds are global population thresholds for each score type.
    # This avoids unstable cluster-specific lambdas when topology changes.
    global_thresholds = {}
    for key in all_signal_keys:
        values = [row[key] for row in per_client.values()]
        thr, stats = _decayed_robust_threshold(
            values=values,
            metric_name=key,
            round_idx=round_idx,
            total_rounds=total_rounds,
            tau_start=tau_start_map[key],
            tau_min=tau_min_map[key],
            beta=lambda_decay_beta,
        )
        global_thresholds[key] = {"threshold": thr, "stats": stats}

    # Cluster-level global anomaly: server may remove the entire cluster.
    cluster_level_rows = {}
    cluster_threshold_values = {"cluster_dE_global": [], "cluster_dN_global": []}
    for cluster_id, cluster_ref in cluster_refs.items():
        cluster_dE_global = compute_distance(cluster_ref, global_ref, metric="euclidean")
        cluster_dC_global = compute_distance(cluster_ref, global_ref, metric="cosine")
        cluster_norm = torch.norm(_as_proto_vec(cluster_ref), p=2).item()
        cluster_dN_global = abs(math.log((cluster_norm + eps) / (global_norm_ref + eps)))

        cluster_level_rows[cluster_id] = {
            "cluster_dE_global": float(cluster_dE_global),
            "cluster_dC_global": float(cluster_dC_global),
            "cluster_dN_global": float(cluster_dN_global),
            "cluster_norm": float(cluster_norm),
        }
        cluster_threshold_values["cluster_dE_global"].append(float(cluster_dE_global))
        cluster_threshold_values["cluster_dN_global"].append(float(cluster_dN_global))

    cluster_global_thresholds = {}
    for key, values in cluster_threshold_values.items():
        thr, stats = _decayed_robust_threshold(
            values=values,
            metric_name=key,
            round_idx=round_idx,
            total_rounds=total_rounds,
            tau_start=tau_start_map[key],
            tau_min=tau_min_map[key],
            beta=lambda_decay_beta,
        )
        cluster_global_thresholds[key] = {"threshold": thr, "stats": stats}

    cluster_forced_poisoned = set()
    for cluster_id, row in cluster_level_rows.items():
        lambda_cluster_E = cluster_global_thresholds["cluster_dE_global"]["threshold"]
        lambda_cluster_N = cluster_global_thresholds["cluster_dN_global"]["threshold"]

        flag_cluster_E = bool(row["cluster_dE_global"] > lambda_cluster_E)
        flag_cluster_N = bool(row["cluster_dN_global"] > lambda_cluster_N)

        # Main cluster-level rule requested by the user:
        # If the centroid/cluster reference is too far from the global reference,
        # the Global Server can remove the whole cluster.
        cluster_poisoned_by_centroid = bool(enable_cluster_centroid_filter and flag_cluster_E)

        row["lambda_cluster_dE_global"] = float(lambda_cluster_E)
        row["lambda_cluster_dN_global"] = float(lambda_cluster_N)
        row["flag_cluster_dE_global"] = int(flag_cluster_E)
        row["flag_cluster_dN_global"] = int(flag_cluster_N)
        row["cluster_poisoned_by_centroid"] = int(cluster_poisoned_by_centroid)

        if cluster_poisoned_by_centroid:
            cluster_forced_poisoned.add(cluster_id)

    # Threshold record for every cluster: used for reporting only.
    cluster_thresholds = {}
    for cluster_id, members in clusters.items():
        cluster_thresholds[cluster_id] = {}
        for key in all_signal_keys:
            cluster_thresholds[cluster_id][key] = {
                "threshold": global_thresholds[key]["threshold"],
                "stats": global_thresholds[key]["stats"],
                "scope": "global_population_threshold",
            }
        for key in cluster_global_thresholds:
            cluster_thresholds[cluster_id][key] = {
                "threshold": cluster_global_thresholds[key]["threshold"],
                "stats": cluster_global_thresholds[key]["stats"],
                "scope": "cluster_centroid_global_threshold",
            }

    poisoned_client_ids = []
    benign_client_ids = []

    for cid, row in per_client.items():
        cluster_id = row["cluster_id"]
        votes = 0
        score = 0.0

        for key in all_signal_keys:
            threshold_pack = global_thresholds[key]
            threshold = float(threshold_pack["threshold"])
            stats = threshold_pack["stats"]
            flag = bool(row[key] > threshold)
            z = _robust_z_from_stats(row[key], stats)

            row[f"lambda_{key}"] = threshold
            row[f"raw_lambda_{key}"] = float(stats.get("raw_threshold", threshold))
            row[f"tau_{key}"] = float(stats.get("tau", np.nan))
            row[f"z_{key}"] = z
            row[f"flag_{key}"] = int(flag)
            row[f"threshold_scope_{key}"] = "global_population_threshold"

            if key in decision_signal_keys:
                votes += int(flag)
                score += max(0.0, z)

        flag_E_local = bool(row["flag_dE_local"])
        flag_N_local = bool(row["flag_dN_local"])
        flag_E_global = bool(row["flag_dE_global"])
        flag_N_global = bool(row["flag_dN_global"])

        local_votes = int(flag_E_local) + int(flag_N_local)
        global_votes = int(flag_E_global) + int(flag_N_global)
        magnitude_votes = local_votes + global_votes

        row["local_votes"] = int(local_votes)
        row["global_votes"] = int(global_votes)
        row["magnitude_votes"] = int(magnitude_votes)
        row["direction_votes"] = int(row.get("flag_dC_local", 0)) + int(row.get("flag_dC_global", 0)) + int(row.get("flag_dD", 0))
        row["suspicious_votes"] = int(votes)
        row["suspicious_score"] = float(score)
        row["cluster_forced_poisoned"] = int(cluster_id in cluster_forced_poisoned)

        # Main client-level decision rule:
        # - global magnitude agreement: different from whole population in both distance and norm;
        # - cross-view distance agreement: abnormal vs both cluster and global reference;
        # - cross-view norm agreement: abnormal norm vs both cluster and global reference;
        # - or, as a fallback, enough magnitude signals vote abnormal.
        client_poisoned_by_score = bool(
            (flag_E_global and flag_N_global)
            or (flag_E_global and flag_E_local)
            or (flag_N_global and flag_N_local)
            or (votes >= int(vote_threshold) and global_votes >= 1)
        )

        row["client_poisoned_by_score"] = int(client_poisoned_by_score)
        row["is_poisoned"] = int(row["cluster_forced_poisoned"] or client_poisoned_by_score)

        if row["is_poisoned"]:
            poisoned_client_ids.append(cid)
        else:
            benign_client_ids.append(cid)

    # Cluster summary: now both diagnostic and actionable.
    cluster_summary = {}
    poisoned_set = set(poisoned_client_ids)
    for cluster_id, members in clusters.items():
        valid_members = [cid for cid in members if cid in per_client]
        poisoned_members = [cid for cid in valid_members if cid in poisoned_set]
        benign_members = [cid for cid in valid_members if cid not in poisoned_set]
        ratio = len(poisoned_members) / max(len(valid_members), 1)

        cluster_row = cluster_level_rows.get(cluster_id, {})
        cluster_status = "POISONED_BY_CENTROID" if cluster_id in cluster_forced_poisoned else (
            "CONTAMINATED" if ratio >= cluster_contamination_ratio else "MOSTLY_BENIGN"
        )

        cluster_summary[cluster_id] = {
            "cluster_id": int(cluster_id),
            "members": valid_members,
            "poisoned_members": poisoned_members,
            "benign_members": benign_members,
            "num_members": int(len(valid_members)),
            "num_poisoned": int(len(poisoned_members)),
            "suspicious_ratio": float(ratio),
            "cluster_status": cluster_status,
            "cluster_forced_poisoned": int(cluster_id in cluster_forced_poisoned),
            **cluster_row,
        }

    if verbose:
        print("\n=== Local + global prototype-only poison detection ===")
        print(
            "Client lambdas: "
            + ", ".join(
                f"{key}={global_thresholds[key]['threshold']:.6f}"
                for key in decision_signal_keys
            )
        )
        print(
            "Cluster lambdas: "
            + ", ".join(
                f"{key}={cluster_global_thresholds[key]['threshold']:.6f}"
                for key in cluster_global_thresholds
            )
        )
        print(f"Clusters forced poisoned by centroid: {sorted(cluster_forced_poisoned)}")
        print(f"Detected poisoned clients: {sorted(poisoned_client_ids)}")
        print(f"Detected benign clients: {len(benign_client_ids)}/{len(client_ids)}")
        for cluster_id, summary in sorted(cluster_summary.items()):
            print(
                f"Cluster {cluster_id}: size={summary['num_members']}, "
                f"poisoned={summary['num_poisoned']}, "
                f"ratio={summary['suspicious_ratio']:.3f}, "
                f"centroid_flag={summary['cluster_forced_poisoned']}, "
                f"status={summary['cluster_status']}"
            )

    return {
        "poisoned_client_ids": sorted(poisoned_client_ids),
        "benign_client_ids": sorted(benign_client_ids),
        "per_client": per_client,
        "cluster_summary": cluster_summary,
        "global_thresholds": global_thresholds,
        "cluster_global_thresholds": cluster_global_thresholds,
        "cluster_thresholds": cluster_thresholds,
        "threshold_state": threshold_state,
        "vote_threshold": int(vote_threshold),
        "decision_signal_keys": decision_signal_keys,
        "debug_signal_keys": debug_signal_keys,
    }


def build_benign_cluster_prototypes(clusters, client_prototypes, poisoned_client_ids):
    """Build robust cluster prototypes using only benign clients inside each cluster."""
    poisoned_set = set(poisoned_client_ids or [])
    benign_cluster_protos = []
    benign_safe_counts = []
    benign_cluster_members = {}

    for cluster_id, members in clusters.items():
        benign_members = [cid for cid in members if cid in client_prototypes and cid not in poisoned_set]
        benign_cluster_members[cluster_id] = benign_members

        if not benign_members:
            continue

        # Robust within-cluster aggregation after removing suspicious clients.
        cluster_proto = coordinate_wise_median(client_prototypes, client_ids=benign_members)
        benign_cluster_protos.append(cluster_proto)
        benign_safe_counts.append(len(benign_members))

    return benign_cluster_protos, benign_safe_counts, benign_cluster_members


def save_detection_report(detection_result, round_num, output_dir):
    """Save detailed client-level and cluster-level detection reports."""
    os.makedirs(output_dir, exist_ok=True)
    current_round = int(round_num) + 1

    client_rows = []
    for cid, row in sorted(detection_result["per_client"].items()):
        saved = dict(row)
        saved["round"] = current_round
        client_rows.append(saved)

    client_path = os.path.join(output_dir, f"prototype_detection_clients_round_{current_round}.csv")
    pd.DataFrame(client_rows).to_csv(client_path, index=False)

    cluster_rows = []
    for cluster_id, summary in sorted(detection_result["cluster_summary"].items()):
        row = {
            "round": current_round,
            "cluster_id": int(cluster_id),
            "num_members": summary["num_members"],
            "num_poisoned": summary["num_poisoned"],
            "suspicious_ratio": summary["suspicious_ratio"],
            "cluster_status": summary["cluster_status"],
            "cluster_forced_poisoned": summary.get("cluster_forced_poisoned", 0),
            "cluster_dE_global": summary.get("cluster_dE_global", np.nan),
            "cluster_dC_global": summary.get("cluster_dC_global", np.nan),
            "cluster_dN_global": summary.get("cluster_dN_global", np.nan),
            "lambda_cluster_dE_global": summary.get("lambda_cluster_dE_global", np.nan),
            "lambda_cluster_dN_global": summary.get("lambda_cluster_dN_global", np.nan),
            "flag_cluster_dE_global": summary.get("flag_cluster_dE_global", 0),
            "flag_cluster_dN_global": summary.get("flag_cluster_dN_global", 0),
            "members": ";".join(map(str, summary["members"])),
            "poisoned_members": ";".join(map(str, summary["poisoned_members"])),
            "benign_members": ";".join(map(str, summary["benign_members"])),
        }
        cluster_rows.append(row)

    cluster_path = os.path.join(output_dir, f"prototype_detection_clusters_round_{current_round}.csv")
    pd.DataFrame(cluster_rows).to_csv(cluster_path, index=False)

    # Save lambda values for debugging threshold decay.
    threshold_rows = []
    for scope, threshold_dict in [
        ("client", detection_result.get("global_thresholds", {}) or {}),
        ("cluster", detection_result.get("cluster_global_thresholds", {}) or {}),
    ]:
        for metric_name, pack in threshold_dict.items():
            stats = dict(pack.get("stats", {}) or {})
            threshold_rows.append({
                "round": current_round,
                "scope": scope,
                "metric": metric_name,
                "threshold": pack.get("threshold", np.nan),
                **stats,
            })

    threshold_path = os.path.join(output_dir, f"prototype_detection_lambdas_round_{current_round}.csv")
    if threshold_rows:
        pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)

    print(f"Saved detection client report: {client_path}")
    print(f"Saved detection cluster report: {cluster_path}")
    if threshold_rows:
        print(f"Saved detection lambda report: {threshold_path}")
    return client_path, cluster_path
