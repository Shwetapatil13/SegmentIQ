"""
Clustering models: K-Means, Agglomerative Hierarchical, and DBSCAN.
Includes parameter validation, runtime benchmarking, and centroid tracking.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.neighbors import NearestNeighbors


def run_kmeans(
    X_scaled: np.ndarray,
    k: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute K-Means clustering with validation and runtime recording.
    """
    n_samples = X_scaled.shape[0]
    if k < 2 or k >= n_samples:
        raise ValueError(f"k must satisfy 2 <= k < {n_samples}, got k={k}.")

    start_time = time.perf_counter()
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    runtime = time.perf_counter() - start_time

    return {
        "model": model,
        "labels": labels,
        "n_clusters": k,
        "centroids_scaled": model.cluster_centers_,
        "inertia": float(model.inertia_),
        "runtime_sec": float(runtime),
        "algorithm": "K-Means",
        "params": {"k": k, "init": "k-means++", "random_state": random_state}
    }


def compute_kmeans_parameter_curves(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Dict[str, list]:
    """
    Compute inertia (Elbow) and silhouette scores for range of k.
    """
    from src.evaluation import evaluate_clustering

    k_values = list(range(k_min, min(k_max + 1, X_scaled.shape[0])))
    inertias = []
    silhouettes = []

    for k in k_values:
        res = run_kmeans(X_scaled, k=k, random_state=random_state)
        inertias.append(res["inertia"])
        metrics = evaluate_clustering(X_scaled, res["labels"])
        silhouettes.append(metrics["silhouette_score"] if metrics["is_valid"] else np.nan)

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes
    }


def run_agglomerative(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    linkage_type: str = "ward",
    metric: str = "euclidean"
) -> Dict[str, Any]:
    """
    Execute Agglomerative Hierarchical clustering.
    Enforces that Ward linkage strictly requires Euclidean distance.
    """
    n_samples = X_scaled.shape[0]
    if n_clusters < 2 or n_clusters >= n_samples:
        raise ValueError(f"n_clusters must satisfy 2 <= n_clusters < {n_samples}, got {n_clusters}.")

    valid_linkages = ["ward", "complete", "average", "single"]
    if linkage_type not in valid_linkages:
        raise ValueError(f"Invalid linkage '{linkage_type}'. Must be one of {valid_linkages}.")

    if linkage_type == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage strictly requires 'euclidean' distance metric.")

    start_time = time.perf_counter()
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric if linkage_type != "ward" else "euclidean",
        linkage=linkage_type
    )
    labels = model.fit_predict(X_scaled)
    runtime = time.perf_counter() - start_time

    return {
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "runtime_sec": float(runtime),
        "algorithm": "Agglomerative",
        "params": {"n_clusters": n_clusters, "linkage": linkage_type, "metric": metric}
    }


def compute_dendrogram_data(
    X_scaled: np.ndarray,
    method: str = "ward",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute linkage matrix for dendrogram visualization, optionally subsampling for readability.
    """
    if sample_size and sample_size < len(X_scaled):
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(X_scaled), size=sample_size, replace=False)
        X_sample = X_scaled[sample_indices]
        is_subsampled = True
    else:
        X_sample = X_scaled
        is_subsampled = False

    Z = linkage(X_sample, method=method)
    return {
        "linkage_matrix": Z,
        "sample_size": len(X_sample),
        "is_subsampled": is_subsampled,
        "method": method
    }


def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5
) -> Dict[str, Any]:
    """
    Execute DBSCAN density-based clustering with noise tracking and degenerate outcome handling.
    """
    if eps <= 0:
        raise ValueError(f"eps must be strictly positive, got {eps}.")
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1, got {min_samples}.")

    start_time = time.perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    runtime = time.perf_counter() - start_time

    unique_labels = np.unique(labels)
    n_noise = int(np.sum(labels == -1))
    noise_pct = (n_noise / len(labels)) * 100
    clusters_found = len([lbl for lbl in unique_labels if lbl != -1])

    # Check degenerate outcomes
    is_all_noise = (clusters_found == 0)
    is_single_cluster = (clusters_found == 1 and n_noise == 0)
    
    degenerate_msg = None
    if is_all_noise:
        degenerate_msg = (
            f"Degenerate Outcome: All {len(labels)} observations were classified as noise (eps={eps} is too small, "
            f"or min_samples={min_samples} is too large)."
        )
    elif is_single_cluster:
        degenerate_msg = (
            f"Degenerate Outcome: All observations were merged into a single cluster (eps={eps} is too large)."
        )

    return {
        "model": model,
        "labels": labels,
        "n_clusters": clusters_found,
        "n_noise": n_noise,
        "noise_pct": round(float(noise_pct), 2),
        "runtime_sec": float(runtime),
        "is_all_noise": is_all_noise,
        "is_single_cluster": is_single_cluster,
        "degenerate_message": degenerate_msg,
        "algorithm": "DBSCAN",
        "params": {"eps": eps, "min_samples": min_samples}
    }


def compute_k_distance_graph(
    X_scaled: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, float]:
    """
    Compute sorted k-nearest neighbor distances to guide epsilon selection.
    Returns:
        (sorted_k_distances, recommended_knee_estimate)
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    # Distance to k-th neighbor (0-indexed k-1)
    k_dists = np.sort(distances[:, k - 1])
    # Knee estimate: median + 1.5 * IQR of distances
    q1 = np.percentile(k_dists, 25)
    q3 = np.percentile(k_dists, 75)
    knee_est = float(np.median(k_dists) + 1.0 * (q3 - q1))
    return k_dists, round(knee_est, 2)
