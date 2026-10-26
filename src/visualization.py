"""
Visualization module providing Plotly helpers for Streamlit and Matplotlib/Seaborn helpers for reports.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from typing import Dict, Any, List, Optional, Tuple


# Set static plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12


# -------------------------------------------------------------
# Matplotlib / Seaborn Static Helpers (for reports & notebook)
# -------------------------------------------------------------
def plot_eda_distributions(df: pd.DataFrame, numeric_cols: List[str], save_path: Optional[str] = None) -> plt.Figure:
    """Plot histograms with KDE and boxplots for selected numeric features."""
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, col in enumerate(numeric_cols):
        # Histogram with KDE
        sns.histplot(df[col], kde=True, ax=axes[0, i], color="#2563eb", bins=20)
        axes[0, i].set_title(f"Distribution of {col}")
        axes[0, i].set_ylabel("Count")

        # Boxplot
        sns.boxplot(y=df[col], ax=axes[1, i], color="#60a5fa")
        axes[1, i].set_title(f"Boxplot of {col}")

    plt.suptitle("Figure 1: Exploratory Feature Distributions and Outlier Spread", weight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_elbow_and_silhouette(
    k_values: List[int],
    inertias: List[float],
    silhouettes: List[float],
    best_k: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot K-Means Elbow curve (Inertia SSE) and Silhouette coefficient curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    axes[0].plot(k_values, inertias, marker="o", linewidth=2.5, color="#1e3a8a")
    axes[0].axvline(x=best_k, color="crimson", linestyle="--", label=f"Selected Elbow (k={best_k})")
    axes[0].set_title("K-Means Elbow Method (Inertia SSE vs. k)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)")
    axes[0].set_xticks(k_values)
    axes[0].legend()

    # Silhouette
    axes[1].plot(k_values, silhouettes, marker="s", linewidth=2.5, color="#059669")
    axes[1].axvline(x=best_k, color="crimson", linestyle="--", label=f"Optimal k={best_k}")
    axes[1].set_title("K-Means Silhouette Score vs. k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Mean Silhouette Coefficient")
    axes[1].set_xticks(k_values)
    axes[1].legend()

    plt.suptitle("Figure 2: K-Means Parameter Optimization Diagnostics", weight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_dendrogram_static(linkage_matrix: np.ndarray, cut_distance: float = 6.0, save_path: Optional[str] = None) -> plt.Figure:
    """Plot hierarchical clustering dendrogram using Ward linkage."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    dendrogram(linkage_matrix, p=30, truncate_mode="lastp", ax=ax, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
    ax.axhline(y=cut_distance, color="crimson", linestyle="--", label=f"Cut Distance Threshold (height={cut_distance})")
    ax.set_title("Figure 3: Hierarchical Agglomerative Dendrogram (Ward Linkage)", weight="bold")
    ax.set_xlabel("Cluster Node Index (or sample count in node)")
    ax.set_ylabel("Ward Distance / Dissimilarity")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_k_distance_static(k_dists: np.ndarray, knee_value: float, save_path: Optional[str] = None) -> plt.Figure:
    """Plot sorted k-distance curve for DBSCAN epsilon selection."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_dists, color="#7c3aed", linewidth=2.5)
    ax.axhline(y=knee_value, color="crimson", linestyle="--", label=f"Knee Threshold (eps ≈ {knee_value})")
    ax.set_title("Figure 4: DBSCAN k-Distance Graph (k=5)", weight="bold")
    ax.set_xlabel("Points Sorted Ascending by Distance to 5th Nearest Neighbor")
    ax.set_ylabel("5-NN Distance")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_clustering_comparison_scatter(
    X_plot: np.ndarray,
    results_dict: Dict[str, np.ndarray],
    feature_labels: Tuple[str, str],
    centroids_dict: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot side-by-side scatter plots of clustering partitions."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]

    palette = sns.color_palette("tab10")

    for i, (title, labels) in enumerate(results_dict.items()):
        ax = axes[i]
        unique_labels = np.unique(labels)
        
        for u in unique_labels:
            mask = (labels == u)
            if u == -1:
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c="black", marker="x", s=40, label="Noise (-1)", alpha=0.7)
            else:
                color = palette[u % len(palette)]
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], s=45, color=color, alpha=0.8, label=f"Cluster {u}")

        # Centroids if available
        if centroids_dict and title in centroids_dict:
            cents = centroids_dict[title]
            ax.scatter(cents[:, 0], cents[:, 1], c="red", marker="*", s=220, edgecolor="black", label="Centroids")

        ax.set_title(f"{title}", weight="bold")
        ax.set_xlabel(feature_labels[0])
        ax.set_ylabel(feature_labels[1])
        ax.legend(loc="upper right", markerscale=1.2, fontsize=8)

    plt.suptitle("Figure 5: 2D Spatial Comparison of Clustering Partitions", weight="bold", y=1.03)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_metric_bars_static(comparison_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot bar charts for Silhouette, Davies-Bouldin, and Calinski-Harabasz."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    algos = comparison_df["Algorithm"].tolist()

    # Parse numeric or nan
    def to_float(val):
        try:
            return float(str(val).split()[0])
        except (ValueError, IndexError):
            return np.nan

    sil_vals = [to_float(v) for v in comparison_df.get("Silhouette Score", comparison_df.get("Silhouette Score (↑)", []))]
    db_vals = [to_float(v) for v in comparison_df.get("Davies-Bouldin Index", comparison_df.get("Davies-Bouldin Index (↓)", []))]
    ch_vals = [to_float(v) for v in comparison_df.get("Calinski-Harabasz Index", comparison_df.get("Calinski-Harabasz Index (↑)", []))]
    rt_vals = [to_float(v) for v in comparison_df.get("Runtime (s)", [])]

    colors = ["#1e3a8a", "#059669", "#7c3aed", "#d97706"][:len(algos)]

    # Silhouette
    axes[0].bar(algos, sil_vals, color=colors)
    axes[0].set_title("Silhouette Score (Higher = Better)")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=25)

    # Davies-Bouldin
    axes[1].bar(algos, db_vals, color=colors)
    axes[1].set_title("Davies-Bouldin Index (Lower = Better)")
    axes[1].set_ylabel("Index")
    axes[1].tick_params(axis="x", rotation=25)

    # Calinski-Harabasz
    axes[2].bar(algos, ch_vals, color=colors)
    axes[2].set_title("Calinski-Harabasz Index (Higher = Better)")
    axes[2].set_ylabel("Index")
    axes[2].tick_params(axis="x", rotation=25)

    # Runtime
    axes[3].bar(algos, rt_vals, color=colors)
    axes[3].set_title("Runtime in Seconds (Lower = Better)")
    axes[3].set_ylabel("Seconds")
    axes[3].tick_params(axis="x", rotation=25)

    plt.suptitle("Figure 6: Quantitative Clustering Benchmark Across Evaluation Metrics", weight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cluster_profiles_heatmap(profile_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot standardized attribute deviation heatmap across clusters."""
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_cols = [
        c for c in profile_df.columns
        if pd.api.types.is_numeric_dtype(profile_df[c])
        and c not in ["Cluster", "Count", "Percentage"]
        and not c.endswith("_median")
    ]
    
    # Standardize means across clusters for heatmap visualization
    num_df = profile_df.set_index("Cluster")[feature_cols]
    z_scores = (num_df - num_df.mean()) / num_df.std()

    sns.heatmap(z_scores.T, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Standardized Deviation (Z-score)"})
    ax.set_title("Figure 7: Cluster Behavioral Deviation Profile Heatmap", weight="bold")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Customer Feature")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
