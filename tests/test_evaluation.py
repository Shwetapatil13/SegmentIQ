"""
Unit tests for evaluation metrics and mathematical validity guards.
"""

import pytest
import numpy as np
from src.evaluation import evaluate_clustering, build_comparison_table


@pytest.fixture
def clean_clusters():
    np.random.seed(42)
    c1 = np.random.randn(20, 2) + np.array([0, 0])
    c2 = np.random.randn(20, 2) + np.array([10, 10])
    X = np.vstack([c1, c2])
    labels = np.array([0] * 20 + [1] * 20)
    return X, labels


def test_evaluate_clustering_valid_metrics(clean_clusters):
    X, labels = clean_clusters
    res = evaluate_clustering(X, labels, runtime_sec=0.05)
    assert res["is_valid"] is True
    assert res["invalid_reason"] is None
    assert res["silhouette_score"] > 0.6  # distinct clusters
    assert res["davies_bouldin_index"] > 0.0
    assert res["calinski_harabasz_index"] > 0.0
    assert res["n_clusters"] == 2
    assert res["n_noise"] == 0
    assert res["runtime_sec"] == 0.05


def test_evaluate_clustering_rejects_single_cluster():
    X = np.random.randn(30, 2)
    labels = np.array([0] * 30)
    res = evaluate_clustering(X, labels)
    assert res["is_valid"] is False
    assert "Only 1 valid cluster" in res["invalid_reason"]
    assert res["silhouette_score"] is None
    assert res["davies_bouldin_index"] is None


def test_evaluate_clustering_rejects_all_noise():
    X = np.random.randn(30, 2)
    labels = np.array([-1] * 30)
    res = evaluate_clustering(X, labels)
    assert res["is_valid"] is False
    assert "All observations were classified as noise" in res["invalid_reason"]
    assert res["n_clusters"] == 0
    assert res["n_noise"] == 30
    assert res["noise_pct"] == 100.0


def test_evaluate_clustering_rejects_k_equals_n():
    X = np.random.randn(10, 2)
    labels = np.arange(10)  # Each point is its own cluster
    res = evaluate_clustering(X, labels)
    assert res["is_valid"] is False
    assert "Every data point is its own cluster" in res["invalid_reason"]


def test_evaluate_clustering_excludes_dbscan_noise():
    X = np.random.randn(50, 2)
    # 20 in cluster 0, 20 in cluster 1, 10 noise points (-1)
    labels = np.array([0] * 20 + [1] * 20 + [-1] * 10)
    res = evaluate_clustering(X, labels)
    assert res["is_valid"] is True
    assert res["n_clusters"] == 2
    assert res["n_noise"] == 10
    assert res["noise_pct"] == 20.0
    assert res["silhouette_score"] is not None


def test_build_comparison_table():
    sample_res = [
        {
            "algorithm": "K-Means",
            "params": {"k": 3},
            "metrics": {
                "is_valid": True,
                "n_clusters": 3,
                "n_noise": 0,
                "noise_pct": 0.0,
                "silhouette_score": 0.55,
                "davies_bouldin_index": 0.75,
                "calinski_harabasz_index": 120.5,
                "runtime_sec": 0.0123
            }
        },
        {
            "algorithm": "DBSCAN",
            "params": {"eps": 0.5, "min_samples": 5},
            "metrics": {
                "is_valid": False,
                "invalid_reason": "Only 1 valid cluster was formed",
                "n_clusters": 1,
                "n_noise": 10,
                "noise_pct": 10.0,
                "runtime_sec": 0.005
            }
        }
    ]
    df_table = build_comparison_table(sample_res)
    assert len(df_table) == 2
    assert "Clusters Found (excl. noise)" in df_table.columns
    assert df_table.loc[0, "Silhouette Score"] == "0.5500"
    assert "N/A" in df_table.loc[1, "Silhouette Score"]
