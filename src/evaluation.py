"""
Evaluation metrics module with mathematical validity guards and comparative summary tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clustering(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    runtime_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics with strict validity checks.
    For density-based clustering, noise points (label -1) are excluded before metric calculation.
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_noise = int(np.sum(labels == -1))
    noise_pct = (n_noise / n_samples) * 100 if n_samples > 0 else 0.0

    # Filter out noise points
    non_noise_mask = (labels != -1)
    clean_labels = labels[non_noise_mask]
    clean_X = X_scaled[non_noise_mask]
    n_clean_samples = len(clean_labels)
    n_valid_clusters = len(np.unique(clean_labels))

    is_valid = True
    invalid_reason = None

    if n_valid_clusters < 2:
        is_valid = False
        if n_valid_clusters == 0:
            invalid_reason = "All observations were classified as noise (-1)."
        else:
            invalid_reason = f"Only 1 valid cluster was formed. Clustering evaluation requires at least 2 clusters."
    elif n_valid_clusters >= n_clean_samples:
        is_valid = False
        invalid_reason = f"Every data point is its own cluster (n_clusters={n_valid_clusters} >= n_samples={n_clean_samples})."

    if is_valid:
        try:
            sil = float(silhouette_score(clean_X, clean_labels, random_state=42))
            db = float(davies_bouldin_score(clean_X, clean_labels))
            ch = float(calinski_harabasz_score(clean_X, clean_labels))
        except Exception as e:
            is_valid = False
            invalid_reason = f"Metric calculation failed: {str(e)}"
            sil, db, ch = None, None, None
    else:
        sil, db, ch = None, None, None

    return {
        "is_valid": is_valid,
        "invalid_reason": invalid_reason,
        "n_clusters": n_valid_clusters,
        "n_noise": n_noise,
        "noise_pct": round(float(noise_pct), 2),
        "silhouette_score": round(sil, 4) if sil is not None else None,
        "davies_bouldin_index": round(db, 4) if db is not None else None,
        "calinski_harabasz_index": round(ch, 2) if ch is not None else None,
        "runtime_sec": round(float(runtime_sec), 4) if runtime_sec is not None else None,
        "evaluation_note": (
            "Metrics computed on non-noise points. Silhouette, Davies-Bouldin, and Calinski-Harabasz "
            "inherently favor compact, spherical clusters (such as those produced by K-Means and Ward linkage) "
            "and do not capture non-convex density geometries favored by DBSCAN."
        )
    }


def build_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Construct a standardized comparative evaluation table across multiple algorithms.
    """
    rows = []
    for res in results_list:
        algo = res.get("algorithm", "Unknown")
        params_str = ", ".join([f"{k}={v}" for k, v in res.get("params", {}).items() if k != "random_state"])
        eval_metrics = res.get("metrics", {})
        
        n_clusters = eval_metrics.get("n_clusters", res.get("n_clusters", "N/A"))
        n_noise = eval_metrics.get("n_noise", res.get("n_noise", 0))
        noise_pct = eval_metrics.get("noise_pct", res.get("noise_pct", 0.0))
        noise_str = f"{n_noise} ({noise_pct:.1f}%)" if n_noise > 0 else "0 (0.0%)"
        
        if eval_metrics.get("is_valid", True):
            sil = eval_metrics.get("silhouette_score")
            sil_str = f"{sil:.4f}" if sil is not None else "N/A"
            db = eval_metrics.get("davies_bouldin_index")
            db_str = f"{db:.4f}" if db is not None else "N/A"
            ch = eval_metrics.get("calinski_harabasz_index")
            ch_str = f"{ch:.2f}" if ch is not None else "N/A"
        else:
            reason = eval_metrics.get("invalid_reason", "Degenerate partition")
            sil_str = f"N/A ({reason})"
            db_str = "N/A"
            ch_str = "N/A"

        runtime = eval_metrics.get("runtime_sec", res.get("runtime_sec", 0.0))
        runtime_str = f"{runtime:.4f}" if runtime is not None else "N/A"

        rows.append({
            "Algorithm": algo,
            "Parameters": params_str,
            "Clusters Found (excl. noise)": n_clusters,
            "Noise Count & %": noise_str,
            "Silhouette Score": sil_str,
            "Davies-Bouldin Index": db_str,
            "Calinski-Harabasz Index": ch_str,
            "Runtime (s)": runtime_str
        })

    return pd.DataFrame(rows)
