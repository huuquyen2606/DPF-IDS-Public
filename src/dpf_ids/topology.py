"""Prototype topology construction and optimization."""

from __future__ import annotations

from .prototype import compute_distance, coordinate_wise_median

def clustering(client_prototypes, best_client_id, clusters=None, lambda_g=0.5, metric="cosine"):
    """
    Prototype-based clustering only.

    Rule:
    - if nearest centroid distance < lambda_g: assign client to that cluster
    - otherwise: create a new cluster with the client as its own centroid

    No poison decision is made here.
    """
    if best_client_id not in client_prototypes:
        raise ValueError("best_client_id not found in client_prototypes")
    if lambda_g is None:
        raise ValueError("lambda_g must not be None")

    if clusters is None:
        clusters = {best_client_id: [best_client_id]}
    else:
        # Keep previous centroids only if the caller explicitly passes previous clusters.
        # The refactored main loop defaults to rebuilding from scratch each round.
        clusters = {
            centroid_id: [centroid_id]
            for centroid_id in clusters.keys()
            if centroid_id in client_prototypes
        }
        if not clusters:
            clusters = {best_client_id: [best_client_id]}

    for client_id, client_proto in client_prototypes.items():
        if client_id in clusters:
            continue

        distances_to_centroids = {}
        for centroid_id in clusters:
            distances_to_centroids[centroid_id] = compute_distance(
                client_proto,
                client_prototypes[centroid_id],
                metric=metric,
            )

        nearest_centroid_id = min(distances_to_centroids, key=distances_to_centroids.get)
        nearest_dist = distances_to_centroids[nearest_centroid_id]

        if nearest_dist < lambda_g:
            clusters[nearest_centroid_id].append(client_id)
        else:
            clusters[client_id] = [client_id]

    return clusters


def update_centroids(clusters, client_prototypes, metric="cosine"):
    """Select the medoid client in each cluster as the new centroid."""
    updated_clusters = {}
    new_centroids = {}

    for old_centroid_id, client_ids in clusters.items():
        valid_client_ids = [cid for cid in client_ids if cid in client_prototypes]
        if not valid_client_ids:
            continue

        best_client_id = None
        best_total_dist = float("inf")

        for candidate_id in valid_client_ids:
            total_dist = 0.0
            for other_id in valid_client_ids:
                if other_id == candidate_id:
                    continue
                total_dist += compute_distance(
                    client_prototypes[candidate_id],
                    client_prototypes[other_id],
                    metric=metric,
                )

            if total_dist < best_total_dist:
                best_total_dist = total_dist
                best_client_id = candidate_id

        if best_client_id is None:
            continue

        updated_clusters[best_client_id] = valid_client_ids
        new_centroids[best_client_id] = client_prototypes[best_client_id]

    return updated_clusters, new_centroids


def select_global_centroid(new_centroids, metric="cosine"):
    """Select the centroid closest to the robust median of all centroids."""
    if not new_centroids:
        raise ValueError("new_centroids is empty")

    centroid_ids = sorted(new_centroids.keys())
    robust_centroid_ref = coordinate_wise_median(new_centroids, client_ids=centroid_ids)

    scores = {}
    for cid in centroid_ids:
        scores[cid] = compute_distance(
            new_centroids[cid],
            robust_centroid_ref,
            metric=metric,
        )

    global_centroid_id = min(scores, key=scores.get)
    return global_centroid_id, new_centroids[global_centroid_id], scores


def optimize_topology(
    client_prototypes,
    best_client_id,
    clusters=None,
    lambda_g=None,
    metric="cosine",
    max_iters=20,
    stable_rounds=2,
    verbose=True,
):
    """
    Optimize topology by alternating:
    1) prototype clustering
    2) medoid centroid update

    This function does not detect poison. Poison detection is performed after
    topology construction by detect_poisoned_clients_prototype_only(...).
    """
    if not client_prototypes:
        raise ValueError("client_prototypes is empty")
    if best_client_id not in client_prototypes:
        raise ValueError("best_client_id not found in client_prototypes")
    if lambda_g is None:
        raise ValueError("lambda_g must not be None")

    current_best = best_client_id
    current_clusters = clusters
    previous_signature = None
    stable_count = 0
    history = []
    new_centroids = {}

    for it in range(1, max_iters + 1):
        clustered = clustering(
            client_prototypes=client_prototypes,
            best_client_id=current_best,
            clusters=current_clusters,
            lambda_g=lambda_g,
            metric=metric,
        )

        updated_clusters, new_centroids = update_centroids(
            clusters=clustered,
            client_prototypes=client_prototypes,
            metric=metric,
        )

        if not new_centroids:
            raise RuntimeError("No centroids produced during topology optimization")

        signature = tuple(
            sorted((cid, tuple(sorted(members))) for cid, members in updated_clusters.items())
        )

        history.append({"iter": it, "num_clusters": len(updated_clusters)})

        if verbose:
            print(f"[Iter {it}] clusters={len(updated_clusters)}")

        if signature == previous_signature:
            stable_count += 1
        else:
            stable_count = 0

        previous_signature = signature
        current_clusters = updated_clusters

        if stable_count >= stable_rounds:
            if verbose:
                print(f"Topology converged at iter {it}.")
            break

    global_centroid_id, global_centroid_proto, scores = select_global_centroid(
        new_centroids=new_centroids,
        metric=metric,
    )

    return {
        "clusters": current_clusters,
        "centroids": new_centroids,
        "global_centroid_id": global_centroid_id,
        "global_centroid_proto": global_centroid_proto,
        "global_scores": scores,
        "history": history,
    }
