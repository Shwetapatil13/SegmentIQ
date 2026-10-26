"""
SegmentIQ: Shared Utility Module for Clustering, Dimensionality Reduction & Customer Profiling
TAGLINE: Comparative Study of K-Means, Hierarchical and DBSCAN Clustering for Credit Card Customer Segmentation
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_dataset(filepath: str = "CC_GENERAL.csv") -> pd.DataFrame:
    """Load credit card dataset using relative paths with fallback."""
    candidate_paths = [
        filepath,
        os.path.join("data", filepath),
        os.path.join(".", filepath),
        "CC_GENERAL.csv",
        os.path.join("data", "CC_GENERAL.csv")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Dataset could not be found at candidate paths: {candidate_paths}")


def get_variable_types(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize variables with their measurement scale and classification."""
    records = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        missing = int(df[col].isnull().sum())
        pct_missing = round((missing / len(df)) * 100, 2)
        
        if col == "CUST_ID":
            v_type = "Nominal (Categorical Identifier)"
            scale = "Discrete"
        elif n_unique <= 20:
            v_type = "Discrete Numeric (Count / Scale)"
            scale = "Discrete"
        else:
            v_type = "Continuous Numeric (Monetary / Frequency)"
            scale = "Continuous"
            
        records.append({
            "Feature": col,
            "Data Type": dtype,
            "Unique Values": n_unique,
            "Missing Count": missing,
            "Missing %": pct_missing,
            "Classification": v_type,
            "Measurement Scale": scale
        })
    return pd.DataFrame(records)


def compute_iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate IQR bounds and count outliers per numeric feature using Tukey's rule."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    records = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        pct_outliers = round((outliers / len(df)) * 100, 2)
        records.append({
            "Feature": col,
            "Skewness": round(float(df[col].skew()), 2),
            "Q1 (25%)": round(float(q1), 2),
            "Median (50%)": round(float(df[col].median()), 2),
            "Q3 (75%)": round(float(q3), 2),
            "IQR": round(float(iqr), 2),
            "Lower Bound": round(float(lower_bound), 2),
            "Upper Bound": round(float(upper_bound), 2),
            "Outlier Count": outliers,
            "Outlier %": pct_outliers
        })
    return pd.DataFrame(records)


def preprocess_data(df: pd.DataFrame, drop_id: bool = True):
    """
    Impute missing values using median (robust to extreme positive skewness),
    drop ID column, and fit StandardScaler.
    Returns: df_clean, X_scaled, feature_names, scaler, medians
    """
    df_clean = df.copy()
    if drop_id and "CUST_ID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["CUST_ID"])
        
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    # Store medians for real-time inference pipeline
    medians = {col: float(df_clean[col].median()) for col in numeric_cols}
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(medians[col])
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[numeric_cols])
    
    return df_clean, X_scaled, numeric_cols, scaler, medians


def run_pca(X_scaled: np.ndarray, n_components: int = 3, random_state: int = 42):
    """Compute PCA projection up to 3 components and return explained variance ratios."""
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_
    cum_var_2d = float(np.sum(var_ratio[:2]))
    cum_var_3d = float(np.sum(var_ratio[:3]))
    return pca, coords, var_ratio, cum_var_2d, cum_var_3d


def train_clustering(algorithm: str, X_scaled: np.ndarray, params: dict = None, random_state: int = 42):
    """
    Train specified clustering algorithm and measure high-precision execution runtime.
    Supported: 'K-Means', 'Hierarchical (Agglomerative)', 'DBSCAN', 'Gaussian Mixture (GMM)'
    """
    if params is None:
        params = {}
        
    start_time = time.perf_counter()
    
    if algorithm == "K-Means":
        k = params.get("k", 4)
        init = params.get("init", "k-means++")
        max_iter = params.get("max_iter", 300)
        model = KMeans(n_clusters=k, init=init, max_iter=max_iter, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "Hierarchical (Agglomerative)":
        k = params.get("k", 4)
        linkage_type = params.get("linkage", "ward")
        metric = "euclidean" if linkage_type == "ward" else params.get("metric", "euclidean")
        model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage_type)
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "DBSCAN":
        eps = params.get("eps", 2.0)
        min_samples = params.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X_scaled)
        
    elif algorithm == "Gaussian Mixture (GMM)":
        n_components = params.get("k", 4)
        covariance_type = params.get("covariance_type", "full")
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        labels = model.fit_predict(X_scaled)
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    runtime = time.perf_counter() - start_time
    return model, labels, runtime


def evaluate_clusters(X_scaled: np.ndarray, labels: np.ndarray, runtime: float = 0.0) -> dict:
    """
    Compute cluster metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz.
    Explicitly filters out noise points (-1) for density algorithms like DBSCAN.
    """
    unique_labels = np.unique(labels)
    noise_count = int(np.sum(labels == -1)) if -1 in unique_labels else 0
    non_noise_mask = labels != -1
    clean_labels = labels[non_noise_mask]
    clean_X = X_scaled[non_noise_mask]
    num_valid_clusters = len(np.unique(clean_labels))
    
    if num_valid_clusters >= 2 and len(clean_labels) > num_valid_clusters:
        sil = float(silhouette_score(clean_X, clean_labels, random_state=42))
        db = float(davies_bouldin_score(clean_X, clean_labels))
        ch = float(calinski_harabasz_score(clean_X, clean_labels))
    else:
        sil, db, ch = None, None, None
        
    return {
        "n_clusters": num_valid_clusters,
        "n_noise": noise_count,
        "silhouette": round(sil, 4) if sil is not None else -1.0,
        "davies_bouldin": round(db, 4) if db is not None else 999.0,
        "calinski_harabasz": round(ch, 2) if ch is not None else 0.0,
        "runtime_sec": round(runtime, 4)
    }


def get_cluster_profiles(df_clean: pd.DataFrame, labels: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Compute per-cluster average feature values (excluding noise points)."""
    df_prof = df_clean.copy()
    df_prof["Cluster"] = labels
    df_prof = df_prof[df_prof["Cluster"] != -1]
    profile = df_prof.groupby("Cluster")[feature_names].mean().reset_index()
    return profile


def define_personas(profile_df: pd.DataFrame) -> dict:
    """
    Inspect cluster means and assign realistic banking persona names and marketing actions.
    Segment names match the required specification:
    - Big Spenders (Premium Transactors)
    - Cash-Advance Users (Credit Revolvers)
    - Installment Buyers (Budget Shoppers)
    - Low-Activity Customers (Dormant Savers)
    """
    personas = {}
    for _, row in profile_df.iterrows():
        c = int(row["Cluster"])
        bal = row.get("BALANCE", 0)
        purch = row.get("PURCHASES", 0)
        cash_adv = row.get("CASH_ADVANCE", 0)
        cred_lim = row.get("CREDIT_LIMIT", 0)
        installments = row.get("INSTALLMENTS_PURCHASES", 0)
        
        if purch > 2500 or (purch > 1500 and cred_lim > 6000):
            name = "Big Spenders (Premium Transactors)"
            badge = "💎 VIP Elite"
            desc = "High-income transactors making frequent, large-ticket purchases with high credit limits and healthy repayments."
            action = "Offer exclusive platinum/black card tier upgrades, 3x travel reward multipliers, concierge perks, and luxury merchant discounts."
        elif cash_adv > 2000 or (cash_adv > purch * 2 and cash_adv > 800):
            name = "Cash-Advance Users (Credit Revolvers)"
            badge = "💳 Credit Revolvers"
            desc = "Cardholders borrowing substantial cash on credit; generates heavy interest income but carries heightened credit risk."
            action = "Promote structured debt-consolidation personal loans, 0% balance transfer offers, and automated repayment alerts to reduce default risk."
        elif installments > 700 or (installments > purch * 0.45 and purch > 800):
            name = "Installment Buyers (Budget Shoppers)"
            badge = "🛍️ Smart Budgeters"
            desc = "Financially conscious shoppers who actively leverage monthly installment plans rather than lump-sum payments."
            action = "Partner with major electronics and e-commerce platforms for zero-interest EMI financing and seasonal retail cashbacks."
        else:
            name = "Low-Activity Customers (Dormant Savers)"
            badge = "💤 Inactive / Dormant"
            desc = "Cardholders maintaining minimal balances and negligible purchase activity; conservative card usage."
            action = "Launch automated re-activation campaigns, zero annual fee waivers, and introductory $25 cashback on first $100 spent."
            
        personas[c] = {
            "name": name,
            "badge": badge,
            "description": desc,
            "marketing_action": action
        }
    return personas


def predict_customer_segment(customer_dict: dict, scaler: StandardScaler, kmeans_model: KMeans, medians: dict, feature_names: list):
    """
    Given raw customer dictionary, scale and predict K-Means cluster.
    Computes a distance-based confidence score (0-100%) indicating proximity to the assigned cluster centroid.
    """
    row = []
    for feat in feature_names:
        val = customer_dict.get(feat, medians.get(feat, 0.0))
        row.append(float(val))
        
    X_input = np.array([row])
    X_scaled_input = scaler.transform(X_input)
    
    # Distance to all centroids
    centroids = kmeans_model.cluster_centers_
    distances = np.linalg.norm(centroids - X_scaled_input, axis=1)
    
    predicted_cluster = int(np.argmin(distances))
    min_dist = distances[predicted_cluster]
    
    # Softmax / inverse distance confidence
    temperature = 2.0
    exp_neg_dist = np.exp(-distances / temperature)
    confidence_pct = float((exp_neg_dist[predicted_cluster] / np.sum(exp_neg_dist)) * 100)
    
    return predicted_cluster, round(confidence_pct, 1), round(float(min_dist), 2), X_scaled_input, distances
