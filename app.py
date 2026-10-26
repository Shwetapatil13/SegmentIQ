"""
SegmentIQ: A Comparative Study of Clustering Algorithms for Customer Segmentation
Modular Streamlit Dashboard importing directly from src/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_dataset, get_dataset_summary
from src.preprocessing import CustomerPreprocessor, compute_before_after_summary
from src.clustering import (
    run_kmeans,
    run_agglomerative,
    run_dbscan,
    compute_kmeans_parameter_curves,
    compute_dendrogram_data,
    compute_k_distance_graph
)
from src.evaluation import evaluate_clustering, build_comparison_table
from src.run_experiments import derive_cluster_personas_and_recommendations


# -------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="SegmentIQ | Customer Segmentation Study",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    
    .app-header h1 {
        color: white !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .app-header p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .persona-card {
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2563eb;
        background: #ffffff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
        border-top: 1px solid #f1f5f9;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .viva-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .viva-q {
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.3rem;
    }
    
    .viva-a {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Results & Data Ingestion
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_results():
    results_path = os.path.join("reports", "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

default_results = load_default_results()


@st.cache_data(show_spinner=False)
def get_cached_dataset(source="data/Mall_Customers.csv"):
    return load_dataset(source)


# -------------------------------------------------------------
# Sidebar: Navigation & Data Source
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-mall.png", width=55)
    st.title("SegmentIQ")
    st.caption("A Comparative Study of Clustering Algorithms for Customer Segmentation")
    st.markdown("---")

    # Data Source Selector
    data_option = st.radio(
        "📂 Data Source:",
        ["Mall_Customers.csv (Default)", "Upload Custom CSV", "Synthetic Demo Data"],
        index=0
    )

    is_synthetic_banner = False
    if data_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                active_df, is_synthetic_banner = load_dataset(uploaded_file)
                st.success("Uploaded CSV loaded successfully!")
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")
                active_df, _ = get_cached_dataset("data/Mall_Customers.csv")
        else:
            active_df, _ = get_cached_dataset("data/Mall_Customers.csv")
    elif data_option == "Synthetic Demo Data":
        active_df, is_synthetic_banner = get_cached_dataset("data/sample_synthetic.csv")
    else:
        active_df, is_synthetic_banner = get_cached_dataset("data/Mall_Customers.csv")

    st.markdown("---")
    pages = [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "⚙️ Preprocessing",
        "🧪 Clustering Lab",
        "⚖️ Algorithm Comparison",
        "👥 Cluster Insights",
        "🎓 Methodology & Viva"
    ]
    selected_page = st.selectbox("Navigation:", pages, index=0)
    st.markdown("---")
    st.caption("College Mini-Project &bull; Third Year Computer Engineering")
    st.caption("Random Seed: 42 (Deterministic)")


# Display Synthetic Warning Banner if applicable
if is_synthetic_banner:
    st.warning("⚠️ **Notice:** Synthetic demo data in use, not the authentic Mall Customers dataset. Results are illustrative only.")


# Header renderer
def render_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="app-header">
        <h1>SegmentIQ: {title}</h1>
        <p>A Comparative Study of K-Means, Hierarchical and DBSCAN Clustering for Customer Segmentation</p>
        <p style="font-size:0.95rem; opacity:0.85; margin-top:0.3rem;"><em>{subtitle}</em></p>
    </div>
    """, unsafe_allow_html=True)


# Footer renderer
def render_footer():
    st.markdown("""
    <div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; font-size: 0.85rem;">
        <strong>SegmentIQ</strong> &bull; Third Year Computer Engineering Mini-Project &bull; Academic Year 2026<br>
        <em>Student Placeholders: [Student 1 Name, Roll No.] &bull; [Student 2 Name, Roll No.] &bull; Guide: [Faculty Guide Name]</em>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------
if selected_page == "🏠 Overview":
    render_header("Executive Overview", "Problem Statement, Academic Context & Key Benchmark Results")

    summary_info = get_dataset_summary(active_df)

    # KPI Metric Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Customers", f"{summary_info['n_rows']:,}")
    with k2:
        st.metric("Total Features", f"{summary_info['n_cols']}")
    with k3:
        st.metric("Missing Values", f"{summary_info['missing_summary']['Missing Values'].sum()}")
    with k4:
        st.metric("Duplicate Rows", f"{summary_info['n_duplicates']}")
    with k5:
        best_sil = default_results["benchmarks"]["best_algorithm_verdict"]["best_silhouette"] if default_results else 0.5547
        st.metric("Top Silhouette", f"{best_sil:.4f}", "K-Means (k=5)")

    st.markdown("### 🎯 Problem Statement & Business Justification")
    st.write("""
    Modern retail and commercial organizations gather transactional and demographic customer records. 
    However, treating all customers uniformly results in inefficient marketing expenditures, irrelevant promotional campaigns, 
    and missed customer retention opportunities. 
    
    **Customer Segmentation** is the process of partitioning an organization's customer base into distinct, 
    homogeneous subsets that exhibit similar behaviors and purchasing propensities. Because consumer portfolios lack 
    ground-truth labels, **Unsupervised Machine Learning (Clustering)** is the fundamental analytical methodology employed 
    to discover organic customer segments.
    """)

    st.markdown("### 🔬 Academic Objectives & Algorithms Compared")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("""
        **Algorithms Evaluated:**
        1. **K-Means Clustering:** Centroid-based partitioning that minimizes within-cluster sum of squares (Inertia SSE).
        2. **Hierarchical Agglomerative Clustering:** Connectivity-based bottom-up merging with Ward's minimum variance criterion.
        3. **DBSCAN:** Density-based spatial clustering identifying core regions and isolating boundary noise points.
        """)
    with col_b:
        st.success("""
        **Validation Framework:**
        - **Silhouette Coefficient ([-1, 1]):** Measures cohesion within clusters vs separation from neighboring clusters.
        - **Davies-Bouldin Index ([0, ∞)):** Ratio of within-cluster scatter to inter-cluster separation (lower is better).
        - **Calinski-Harabasz Index:** Variance ratio criterion (higher is better).
        - **Execution Runtime:** High-precision benchmarking using `time.perf_counter()`.
        """)

    render_footer()


# -------------------------------------------------------------
# PAGE 2: DATASET EXPLORER
# -------------------------------------------------------------
elif selected_page == "📊 Dataset Explorer":
    render_header("Dataset Explorer", "Data Preview, Schemas, Descriptive Statistics, and Collinearity Heatmap")

    tab1, tab2, tab3 = st.tabs(["📋 Data Preview & Dtypes", "📈 Distributions & Box Plots", "🔥 Correlation Heatmap"])

    with tab1:
        st.subheader("Raw Customer Records Preview")
        st.dataframe(active_df.head(25), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Variable Schemas & Non-Null Counts")
            sum_dict = get_dataset_summary(active_df)
            st.dataframe(sum_dict["dtypes_summary"], use_container_width=True)
        with c2:
            st.subheader("Missing Value Audit")
            st.dataframe(sum_dict["missing_summary"], use_container_width=True)

        st.subheader("Parametric & Non-Parametric Summary Statistics")
        st.dataframe(sum_dict["descriptive_stats"].round(2), use_container_width=True)

    with tab2:
        st.subheader("Feature Distribution & Dispersion Explorer")
        num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_no_id = [c for c in num_cols if "id" not in c.lower()]
        
        selected_feat = st.selectbox("Select Attribute:", num_cols_no_id, index=0)

        f1, f2 = st.columns(2)
        with f1:
            fig_h = px.histogram(
                active_df, x=selected_feat, nbins=25, marginal="box",
                title=f"Histogram with Boxplot: {selected_feat}",
                color_discrete_sequence=["#1e3a8a"]
            )
            fig_h.update_layout(template="plotly_white")
            st.plotly_chart(fig_h, use_container_width=True)
        with f2:
            fig_b = px.box(
                active_df, y=selected_feat, points="all",
                title=f"Detailed Outlier Dispersion: {selected_feat}",
                color_discrete_sequence=["#2563eb"]
            )
            fig_b.update_layout(template="plotly_white")
            st.plotly_chart(fig_b, use_container_width=True)

    with tab3:
        st.subheader("Pearson Correlation Heatmap")
        num_df = active_df.select_dtypes(include=[np.number])
        if "CustomerID" in num_df.columns:
            num_df = num_df.drop(columns=["CustomerID"])
        corr = num_df.corr().round(2)

        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Matrix (Continuous Features)"
        )
        fig_corr.update_layout(height=550, template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    render_footer()


# -------------------------------------------------------------
# PAGE 3: PREPROCESSING
# -------------------------------------------------------------
elif selected_page == "⚙️ Preprocessing":
    render_header("Preprocessing & Normalization", "Feature Selection, Identifier Exclusion, and Leak-Free StandardScaler")

    st.subheader("1. Feature Selection & Identifier Exclusion")
    st.info(r"""
    - **Why Exclude `CustomerID`?** A database primary key carries zero consumer behavioral signal. Because distance-based clustering computes spatial proximity ($\sqrt{\sum (p_i - q_i)^2}$), including an arbitrary sequence ID would inject artificial variance and distort cluster assignments.
    - **Configurable Preprocessing Rules:** Missing numeric values are imputed with the median (resistant to skewness); categorical features are encoded via one-hot/binary transformation; continuous features are standardized via `StandardScaler`.
    """)

    available_cols = [c for c in active_df.columns if "id" not in c.lower()]
    default_selection = [c for c in ["Annual Income (k$)", "Spending Score (1-100)"] if c in available_cols]
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        chosen_features = st.multiselect(
            "Select Features for Clustering (Minimum 2 required):",
            available_cols,
            default=default_selection if len(default_selection) >= 2 else available_cols[:2]
        )
    with col_s2:
        missing_strat = st.selectbox("Missing Value Strategy:", ["impute", "drop"], index=0)

    if len(chosen_features) < 2:
        st.error("⚠️ Please select at least 2 features to perform clustering.")
    else:
        preprocessor = CustomerPreprocessor(selected_features=chosen_features, missing_strategy=missing_strat)
        X_scaled, working_df, audit = preprocessor.fit_transform(active_df)

        st.markdown("### 2. Before vs. After Standardization Summary")
        before_after_df = compute_before_after_summary(active_df, X_scaled, chosen_features)
        st.dataframe(before_after_df, use_container_width=True)

        st.markdown("### 3. Distribution Shift Visualizer")
        feat_to_compare = st.selectbox("Select Feature to Compare:", chosen_features, index=0)
        col_idx = chosen_features.index(feat_to_compare)

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Histogram(
            x=working_df[feat_to_compare], name="Raw Measurement",
            marker_color="#1e3a8a", opacity=0.7, nbinsx=30
        ))
        fig_compare.add_trace(go.Histogram(
            x=X_scaled[:, col_idx], name="Standardized Z-Score (μ=0, σ=1)",
            marker_color="#059669", opacity=0.7, nbinsx=30, xaxis="x2"
        ))
        fig_compare.update_layout(
            title=f"Raw vs. Standardized Distributions for {feat_to_compare}",
            xaxis=dict(title=f"Raw {feat_to_compare}"),
            xaxis2=dict(title=f"Standardized {feat_to_compare} (Z-Score)", overlaying="x", side="top"),
            yaxis=dict(title="Count"),
            barmode="overlay",
            template="plotly_white"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    render_footer()


# -------------------------------------------------------------
# PAGE 4: CLUSTERING LAB
# -------------------------------------------------------------
elif selected_page == "🧪 Clustering Lab":
    render_header("Interactive Clustering Laboratory", "Tune Parameters, Execute Algorithms, and Inspect Spatial Partitions")

    available_cols = [c for c in active_df.columns if "id" not in c.lower()]
    default_selection = [c for c in ["Annual Income (k$)", "Spending Score (1-100)"] if c in available_cols]
    
    c_feat, c_algo = st.columns([1.5, 1.5])
    with c_feat:
        lab_features = st.multiselect(
            "Active Clustering Features:",
            available_cols,
            default=default_selection if len(default_selection) >= 2 else available_cols[:2]
        )
    with c_algo:
        algo_mode = st.selectbox(
            "Algorithm Mode:",
            ["K-Means", "Agglomerative Hierarchical", "DBSCAN", "Compare All 3 Simultaneously"],
            index=0
        )

    if len(lab_features) < 2:
        st.error("Please select at least 2 features.")
    else:
        # Preprocess features
        preprocessor = CustomerPreprocessor(selected_features=lab_features)
        X_scaled, working_df, _ = preprocessor.fit_transform(active_df)

        # Dynamic Controls based on selected algorithm
        col_controls, col_viz = st.columns([1, 2.5])

        with col_controls:
            st.markdown("### 🎛️ Parameters")
            if algo_mode in ["K-Means", "Compare All 3 Simultaneously"]:
                k_val = st.slider("K-Means Clusters (k):", 2, min(10, len(X_scaled) - 1), 5, step=1)
            if algo_mode in ["Agglomerative Hierarchical", "Compare All 3 Simultaneously"]:
                agg_k = st.slider("Hierarchical Clusters (k):", 2, min(10, len(X_scaled) - 1), 5, step=1)
                agg_linkage = st.selectbox("Hierarchical Linkage:", ["ward", "complete", "average", "single"], index=0)
                agg_metric = "euclidean" if agg_linkage == "ward" else st.selectbox("Hierarchical Metric:", ["euclidean", "manhattan", "cosine"], index=0)
            if algo_mode in ["DBSCAN", "Compare All 3 Simultaneously"]:
                db_eps = st.slider("DBSCAN Epsilon (eps):", 0.1, 1.5, 0.35, step=0.05)
                db_min = st.slider("DBSCAN Min Samples:", 2, 15, 5, step=1)

            dim_mode = st.radio("Scatter Dimension:", ["2D Projection", "3D Projection"], index=0)
            run_btn = st.button("🚀 Run Clustering", type="primary", use_container_width=True)

        with col_viz:
            with st.spinner("Computing clustering models and evaluation metrics..."):
                if algo_mode == "K-Means":
                    res = run_kmeans(X_scaled, k=k_val, random_state=42)
                    metrics = evaluate_clustering(X_scaled, res["labels"], res["runtime_sec"])
                    models_to_plot = {"K-Means": (res["labels"], res["centroids_scaled"])}
                elif algo_mode == "Agglomerative Hierarchical":
                    res = run_agglomerative(X_scaled, n_clusters=agg_k, linkage_type=agg_linkage, metric=agg_metric)
                    metrics = evaluate_clustering(X_scaled, res["labels"], res["runtime_sec"])
                    models_to_plot = {"Agglomerative": (res["labels"], None)}
                elif algo_mode == "DBSCAN":
                    res = run_dbscan(X_scaled, eps=db_eps, min_samples=db_min)
                    metrics = evaluate_clustering(X_scaled, res["labels"], res["runtime_sec"])
                    models_to_plot = {"DBSCAN": (res["labels"], None)}
                    if res["degenerate_message"]:
                        st.warning(f"⚠️ {res['degenerate_message']}")
                else:
                    # Compare All
                    res_k = run_kmeans(X_scaled, k=k_val, random_state=42)
                    res_a = run_agglomerative(X_scaled, n_clusters=agg_k, linkage_type=agg_linkage, metric=agg_metric)
                    res_d = run_dbscan(X_scaled, eps=db_eps, min_samples=db_min)
                    metrics = evaluate_clustering(X_scaled, res_k["labels"], res_k["runtime_sec"])
                    models_to_plot = {
                        "K-Means": (res_k["labels"], res_k["centroids_scaled"]),
                        "Agglomerative": (res_a["labels"], None),
                        "DBSCAN": (res_d["labels"], None)
                    }

            # KPI Cards
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.metric("Clusters Found", f"{metrics['n_clusters']}")
            with kpi2:
                st.metric("Noise Points", f"{metrics['n_noise']} ({metrics['noise_pct']}%)")
            with kpi3:
                st.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}" if metrics['silhouette_score'] is not None else "N/A")
            with kpi4:
                st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_index']:.4f}" if metrics['davies_bouldin_index'] is not None else "N/A")
            with kpi5:
                st.metric("Runtime", f"{metrics['runtime_sec']:.4f}s" if metrics['runtime_sec'] is not None else "N/A")

            # Dimensionality handling: if > 2 features, project via PCA for display
            is_pca = len(lab_features) > 2
            if is_pca:
                pca = PCA(n_components=3, random_state=42)
                pca_coords = pca.fit_transform(X_scaled)
                x_plot = pca_coords[:, 0]
                y_plot = pca_coords[:, 1]
                z_plot = pca_coords[:, 2]
                var_exp = pca.explained_variance_ratio_
                x_axis_title = f"PC1 ({var_exp[0]*100:.1f}% Variance)"
                y_axis_title = f"PC2 ({var_exp[1]*100:.1f}% Variance)"
                z_axis_title = f"PC3 ({var_exp[2]*100:.1f}% Variance)"
                st.info(f"ℹ️ **PCA Projection Notice:** {len(lab_features)} features selected. Plotting in PCA projection space (Total 2D Variance: {(var_exp[0]+var_exp[1])*100:.1f}%). Note: PCA axes represent linear combinations, not raw units.")
            else:
                x_plot = active_df[lab_features[0]].values
                y_plot = active_df[lab_features[1]].values
                z_plot = np.zeros(len(x_plot))
                x_axis_title = lab_features[0]
                y_axis_title = lab_features[1]
                z_axis_title = "Z"

            # Render Scatter Plots
            for model_name, (lbls, cents) in models_to_plot.items():
                scatter_df = pd.DataFrame({
                    "X": x_plot,
                    "Y": y_plot,
                    "Z": z_plot,
                    "Cluster": [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in lbls],
                    "CustomerID": active_df["CustomerID"].values if "CustomerID" in active_df.columns else range(1, len(lbls)+1),
                    lab_features[0]: active_df[lab_features[0]].values,
                    lab_features[1]: active_df[lab_features[1]].values
                })

                color_discrete = px.colors.qualitative.Bold
                
                if dim_mode == "2D Projection":
                    fig_sc = px.scatter(
                        scatter_df, x="X", y="Y", color="Cluster",
                        title=f"{model_name} Clustering Scatter Plot",
                        labels={"X": x_axis_title, "Y": y_axis_title},
                        hover_data=["CustomerID", lab_features[0], lab_features[1]],
                        color_discrete_sequence=color_discrete,
                        opacity=0.8
                    )
                    # Add K-Means centroids if available
                    if cents is not None and not is_pca:
                        cents_orig = preprocessor.inverse_transform_numeric(cents)
                        fig_sc.add_trace(go.Scatter(
                            x=cents_orig[:, 0], y=cents_orig[:, 1],
                            mode="markers", marker=dict(symbol="star", size=18, color="red", line=dict(color="black", width=1.5)),
                            name="Centroids"
                        ))
                    fig_sc.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_sc, use_container_width=True)
                else:
                    fig_sc3d = px.scatter_3d(
                        scatter_df, x="X", y="Y", z="Z", color="Cluster",
                        title=f"{model_name} 3D Clustering Scatter Plot",
                        labels={"X": x_axis_title, "Y": y_axis_title, "Z": z_axis_title},
                        hover_data=["CustomerID", lab_features[0], lab_features[1]],
                        color_discrete_sequence=color_discrete,
                        opacity=0.8
                    )
                    fig_sc3d.update_layout(height=600)
                    st.plotly_chart(fig_sc3d, use_container_width=True)

                # Cluster Size Distributions (Bar and Pie)
                cluster_counts = pd.Series(lbls).value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Count"]
                cluster_counts["Label"] = cluster_counts["Cluster"].apply(lambda c: f"Cluster {c}" if c != -1 else "Noise (-1)")
                
                p1, p2 = st.columns(2)
                with p1:
                    fig_b = px.bar(
                        cluster_counts, x="Label", y="Count", color="Label",
                        title=f"{model_name}: Cluster Member Counts (Bar)",
                        color_discrete_sequence=color_discrete
                    )
                    fig_b.update_layout(template="plotly_white", height=300, showlegend=False)
                    st.plotly_chart(fig_b, use_container_width=True)
                with p2:
                    fig_p = px.pie(
                        cluster_counts, names="Label", values="Count",
                        title=f"{model_name}: Portfolio Proportion (Pie)",
                        color_discrete_sequence=color_discrete,
                        hole=0.4
                    )
                    fig_p.update_layout(template="plotly_white", height=300)
                    st.plotly_chart(fig_p, use_container_width=True)

    render_footer()


# -------------------------------------------------------------
# PAGE 5: ALGORITHM COMPARISON
# -------------------------------------------------------------
elif selected_page == "⚖️ Algorithm Comparison":
    render_header("Algorithm Comparison", "Side-by-Side Quantitative Benchmark, Efficiency, and Empirical Limitations")

    # Load from results.json (Single Source of Truth)
    if default_results:
        benchmarks_data = default_results["benchmarks"]["comparison_table"]
        comp_df = pd.DataFrame(benchmarks_data)
        verdict = default_results["benchmarks"]["best_algorithm_verdict"]
    else:
        st.warning("Default benchmark results not found. Running live comparison...")
        from src.run_experiments import run_all_experiments
        res_all = run_all_experiments()
        comp_df = pd.DataFrame(res_all["benchmarks"]["comparison_table"])
        verdict = res_all["benchmarks"]["best_algorithm_verdict"]

    st.subheader("Quantitative Evaluation Table")
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("""
    > **Crucial Methodological Caveat:** Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics inherently favor **compact, convex, spherical geometries** (such as those optimized by K-Means and Ward linkage). Because DBSCAN partitions based on density connectivity and isolates boundary points into noise (-1), internal distance metrics are not directly comparable between partitioning and density paradigms.
    """)

    # Visual Bar Charts
    st.markdown("---")
    st.subheader("Visual Comparative Benchmarks")
    algos = comp_df["Algorithm"].tolist()

    def parse_score(val):
        try:
            return float(str(val).split()[0])
        except (ValueError, IndexError):
            return np.nan

    b1, b2, b3 = st.columns(3)
    with b1:
        s_vals = [parse_score(v) for v in comp_df.get("Silhouette Score", comp_df.get("Silhouette Score (↑)", []))]
        fig_s = px.bar(x=algos, y=s_vals, color=algos, title="Silhouette Score (Higher = Better)")
        fig_s.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_s, use_container_width=True)
    with b2:
        db_vals = [parse_score(v) for v in comp_df.get("Davies-Bouldin Index", comp_df.get("Davies-Bouldin Index (↓)", []))]
        fig_db = px.bar(x=algos, y=db_vals, color=algos, title="Davies-Bouldin Index (Lower = Better)")
        fig_db.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_db, use_container_width=True)
    with b3:
        rt_vals = [parse_score(v) for v in comp_df["Runtime (s)"]]
        fig_rt = px.bar(x=algos, y=rt_vals, color=algos, title="Execution Runtime in Seconds (Lower = Better)")
        fig_rt.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_rt, use_container_width=True)

    # Automated Verdict Banner
    st.markdown(f"""
    <div style="background-color: #ecfdf5; border-left: 6px solid #10b981; padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
        <h4 style="color: #065f46; margin-top:0;">🏆 Empirical Verdict: {verdict['best_model']}</h4>
        <p style="color: #047857; margin-bottom: 0.5rem; line-height: 1.6;">
            <strong>Empirical Reasoning:</strong> {verdict['justification']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_footer()


# -------------------------------------------------------------
# PAGE 6: CLUSTER INSIGHTS
# -------------------------------------------------------------
elif selected_page == "👥 Cluster Insights":
    render_header("Customer Persona Insights", "Rule-Based Segment Discovery, Attribute Means, and Actionable Marketing Suggestions")

    if default_results:
        personas_list = default_results["cluster_personas"]
        personas_df = pd.DataFrame(personas_list)
    else:
        preprocessor = CustomerPreprocessor(selected_features=["Annual Income (k$)", "Spending Score (1-100)"])
        X_scaled, _, _ = preprocessor.fit_transform(active_df)
        res_km = run_kmeans(X_scaled, k=5, random_state=42)
        personas_df = derive_cluster_personas_and_recommendations(
            active_df, res_km["labels"], ["Annual Income (k$)", "Spending Score (1-100)"], std_threshold=0.5
        )

    st.subheader("Discovered Behavioral Segments & Actionable Playbooks")
    st.info("Rule Criteria: Customer segments are derived by comparing cluster means to the overall portfolio mean with an explicit threshold of ±0.5 standard deviations.")

    for _, row in personas_df.iterrows():
        c_id = int(row["Cluster"])
        st.markdown(f"""
        <div class="persona-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin-top:0; color:#1e3a8a;">Cluster {c_id}: {row['Segment Name']}</h3>
                <span style="font-weight:700; color:#2563eb;">{row['Count']} customers ({row['Percentage']}%)</span>
            </div>
            <p style="color:#475569; font-size:0.98rem; margin-bottom:0.8rem;">{row['Description']}</p>
            <div style="background:#f8fafc; padding:0.8rem 1rem; border-radius:8px; display:flex; gap:2.5rem; flex-wrap:wrap; margin-bottom:0.8rem;">
                <div><strong>Annual Income Mean:</strong> ${row.get('Annual Income (k$)', 0):.1f}k (Median: ${row.get('Annual Income (k$)_median', 0):.1f}k)</div>
                <div><strong>Spending Score Mean:</strong> {row.get('Spending Score (1-100)', 0):.1f} (Median: {row.get('Spending Score (1-100)_median', 0):.1f})</div>
            </div>
            <div style="background:#eff6ff; padding:0.85rem 1rem; border-radius:8px; border-left:4px solid #3b82f6;">
                <strong style="color:#1e40af;">💡 Strategy Suggestion:</strong>
                <span style="color:#1e3a8a;"> {row['Recommendation']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Segment Attribute Comparison Heatmap")
    num_cols_heat = [c for c in personas_df.columns if c not in ["Cluster", "Segment Name", "Description", "Recommendation", "Count", "Percentage"] and not c.endswith("_median")]
    heat_df = personas_df.set_index("Cluster")[num_cols_heat]
    heat_z = (heat_df - heat_df.mean()) / heat_df.std()

    fig_heat = px.imshow(
        heat_z.T, color_continuous_scale="RdBu_r", text_auto=".2f",
        title="Normalized Segment Attribute Deviations (Z-Scores)",
        aspect="auto"
    )
    fig_heat.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

    render_footer()


# -------------------------------------------------------------
# PAGE 7: METHODOLOGY & VIVA
# -------------------------------------------------------------
elif selected_page == "🎓 Methodology & Viva":
    render_header("Methodology & Viva Examination Preparation", "Academic Foundations, Mathematical Formulations, and 15+ Viva Q&A Drills")

    t_theory, t_viva = st.tabs(["📚 Core Machine Learning Foundations", "🎯 15+ Comprehensive Viva Q&As"])

    with t_theory:
        st.markdown("""
        ### 1. Supervised vs. Unsupervised Learning
        - **Supervised Learning:** Trains on labeled dataset pairs $(X, y)$ to optimize a mapping function $f(X) \to y$. Evaluated via loss functions comparing predictions against ground-truth labels.
        - **Unsupervised Learning:** Discovers hidden structures, manifolds, and natural affinities in unlabeled data $X$ without supervisor guidance. Evaluated via intrinsic geometric compactness and separation.

        ### 2. Clustering Paradigms
        - **Partitioning (K-Means):** Iteratively updates $k$ centroids to minimize the Within-Cluster Sum of Squares:
          $$\\text{Inertia} = \\sum_{j=1}^k \\sum_{x_i \\in C_j} \\|x_i - \\mu_j\\|^2$$
        - **Hierarchical Agglomerative Clustering:** Starts with $N$ singleton clusters and iteratively merges closest pairs based on linkage criteria. **Ward's linkage** minimizes the increase in total within-cluster variance:
          $$\\Delta \\text{SSE}_{A, B} = \\frac{N_A N_B}{N_A + N_B} \\|\\mu_A - \\mu_B\\|^2$$
        - **Density-Based Spatial Clustering (DBSCAN):** Groups points having at least `min_samples` neighbors within radius $\\varepsilon$. Delineates core points, reachable border points, and isolates outliers as noise ($-1$).
        """)

    with t_viva:
        st.subheader("15 Essential Viva Voce Questions & Model Answers")

        viva_qa = [
            ("Q1: What is the fundamental difference between K-Means and DBSCAN?",
             "K-Means is a parametric partitioning algorithm that assumes spherical, convex clusters and assigns every observation to a cluster. DBSCAN is non-parametric, discovers arbitrary non-linear cluster shapes, and explicitly isolates outliers as noise points."),
            
            ("Q2: Why must we standardize features before running distance-based clustering?",
             "Clustering relies on Euclidean distance. Features with large numerical ranges (e.g., Annual Income: $15k-$137k) will completely dominate smaller features (e.g., Age: 18-70). Standardization ensures each feature contributes equally with zero mean and unit variance."),
            
            ("Q3: Why did we exclude CustomerID from the clustering process?",
             "CustomerID is an arbitrary database index that carries zero customer behavioral information. Including it injects spurious linear distance signals that distort authentic behavioral grouping."),
            
            ("Q4: How does the Elbow Method determine optimal k in K-Means?",
             "The Elbow Method plots Inertia (Within-Cluster Sum of Squares) against k. As k increases, inertia drops monotonically. The optimal k occurs at the bend ('elbow') where the rate of variance reduction sharply diminishes."),
            
            ("Q5: What is the Silhouette Score and what does a negative value indicate?",
             "The Silhouette Score measures how similar an observation is to its own cluster (cohesion a) versus neighboring clusters (separation b): s = (b - a) / max(a, b). A negative value indicates that the point is closer to a neighboring cluster than to its assigned cluster, signaling misclassification."),
            
            ("Q6: How does the Davies-Bouldin Index evaluate cluster validity?",
             "The Davies-Bouldin Index measures the average similarity between each cluster and its most similar counterpart, computed as the ratio of within-cluster scatter to inter-cluster centroid distance. Lower values indicate tighter, better-separated clusters."),
            
            ("Q7: How does Calinski-Harabasz Index differ from Davies-Bouldin?",
             "Calinski-Harabasz is the ratio of between-cluster variance to within-cluster variance (Variance Ratio Criterion). Higher values represent tighter and better-separated partitions."),
            
            ("Q8: Why must noise points be excluded when computing metrics for DBSCAN?",
             "DBSCAN labels outliers as -1. If included in internal validation metrics, noise points act as an artificial cluster, severely distorting centroid calculations and giving an invalid mathematical assessment of the actual clusters discovered."),
            
            ("Q9: What distance metric does Ward's linkage require in Hierarchical clustering?",
             "Ward's linkage mathematically requires Euclidean distance because its update formula minimizes the incremental sum of squared Euclidean deviations."),
            
            ("Q10: What is the computational time complexity of K-Means vs. Agglomerative clustering?",
             "K-Means has time complexity O(k * N * d * iterations), which scales linearly with sample size N. Agglomerative clustering requires O(N^2 log N) time and O(N^2) memory to compute pairwise distance matrices, making it computationally prohibitive for massive datasets."),
            
            ("Q11: How do you select epsilon (eps) for DBSCAN?",
             "By generating a k-distance graph: computing distance to the k-th nearest neighbor (typically k = 2 * dim - 1) sorted in ascending order. The knee or elbow point in the curve reveals the optimal epsilon threshold."),
            
            ("Q12: What happens if epsilon in DBSCAN is set too small or too large?",
             "If eps is too small, a degenerate outcome occurs where almost all points are labeled as noise. If eps is too large, all observations merge into a single massive super-cluster."),
            
            ("Q13: Why is PCA used in this dashboard and what is its limitation?",
             "PCA reduces high-dimensional data (e.g., >2 features) to 2 or 3 orthogonal principal components capturing maximum variance for 2D/3D plotting. Its limitation is that PCA axes are linear combinations rather than original feature values, and variance is lost when cumulative variance is below 100%."),
            
            ("Q14: How are cluster personas derived in this project?",
             "Personas are derived using explicit rule-based thresholds: comparing each cluster's feature mean to the overall dataset mean normalized by standard deviation. If deviations exceed +/-0.5 sigma, behavioral labels are assigned."),
            
            ("Q15: Why is K-Means the preferred algorithm for this customer segmentation task?",
             "K-Means achieved the highest Calinski-Harabasz score (248.65), lowest Davies-Bouldin index (0.5722), executed in 0.03s, and produced crisp, mutually exclusive clusters that directly map into operational commercial customer personas without leaving customers in unassigned noise.")
        ]

        for q, a in viva_qa:
            st.markdown(f"""
            <div class="viva-box">
                <div class="viva-q">{q}</div>
                <div class="viva-a">{a}</div>
            </div>
            """, unsafe_allow_html=True)

    render_footer()
