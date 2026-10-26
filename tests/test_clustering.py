"""
Unit tests for clustering algorithms (K-Means, Agglomerative, DBSCAN) and seed reproducibility.
"""

import pytest
import numpy as np
from src.clustering import run_kmeans, run_agglomerative, run_dbscan, compute_kmeans_parameter_curves


@pytest.fixture
def sample_data():
    np.random.seed(42)
    # 3 distinct 2D Gaussian blobs, total 60 points
    c1 = np.random.randn(20, 2) + np.array([0, 0])
    c2 = np.random.randn(20, 2) + np.array([5, 5])
    c3 = np.random.randn(20, 2) + np.array([10, 0])
    return np.vstack([c1, c2, c3])


def test_kmeans_label_length_and_clusters(sample_data):
    res = run_kmeans(sample_data, k=3, random_state=42)
    assert len(res["labels"]) == len(sample_data)
    assert res["n_clusters"] == 3
    assert len(np.unique(res["labels"])) == 3
    assert res["centroids_scaled"].shape == (3, 2)
    assert res["runtime_sec"] >= 0.0


def test_kmeans_reproducibility_fixed_seed(sample_data):
    res1 = run_kmeans(sample_data, k=3, random_state=42)
    res2 = run_kmeans(sample_data, k=3, random_state=42)
    np.testing.assert_array_equal(res1["labels"], res2["labels"])
    np.testing.assert_allclose(res1["centroids_scaled"], res2["centroids_scaled"])
    assert res1["inertia"] == res2["inertia"]


def test_kmeans_invalid_k(sample_data):
    with pytest.raises(ValueError):
        run_kmeans(sample_data, k=1)
    with pytest.raises(ValueError):
        run_kmeans(sample_data, k=len(sample_data) + 5)


def test_agglomerative_ward_enforces_euclidean(sample_data):
    # Valid call
    res = run_agglomerative(sample_data, n_clusters=3, linkage_type="ward", metric="euclidean")
    assert len(res["labels"]) == len(sample_data)
    assert len(np.unique(res["labels"])) == 3

    # Invalid metric with Ward
    with pytest.raises(ValueError, match="Ward linkage strictly requires 'euclidean'"):
        run_agglomerative(sample_data, n_clusters=3, linkage_type="ward", metric="manhattan")


def test_agglomerative_linkage_options(sample_data):
    res = run_agglomerative(sample_data, n_clusters=3, linkage_type="complete", metric="manhattan")
    assert len(res["labels"]) == len(sample_data)


def test_dbscan_normal_execution(sample_data):
    res = run_dbscan(sample_data, eps=1.5, min_samples=3)
    assert len(res["labels"]) == len(sample_data)
    assert res["n_clusters"] >= 1
    assert res["n_noise"] >= 0


def test_dbscan_all_noise_handled_gracefully(sample_data):
    # Very tiny epsilon produces all noise
    res = run_dbscan(sample_data, eps=0.0001, min_samples=10)
    assert res["is_all_noise"] is True
    assert res["n_clusters"] == 0
    assert res["n_noise"] == len(sample_data)
    assert res["degenerate_message"] is not None


def test_dbscan_single_cluster_handled_gracefully(sample_data):
    # Enormous epsilon merges everything
    res = run_dbscan(sample_data, eps=100.0, min_samples=2)
    assert res["is_single_cluster"] is True
    assert res["n_clusters"] == 1
    assert res["n_noise"] == 0
    assert res["degenerate_message"] is not None
