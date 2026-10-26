# SegmentIQ: A Comparative Study of Clustering Algorithms for Customer Segmentation

**Academic Level:** Third Year Computer Engineering Mini-Project (Assignment 5)  
**Dataset:** Mall Customer Segmentation Data (`Mall_Customers.csv`, 200 records, 5 features)  
**Algorithms Evaluated:** K-Means Partitioning, Hierarchical Agglomerative (Ward Linkage), and DBSCAN  
**Architecture:** Modular Python package (`src/`), Comprehensive Test Suite (`tests/`), Interactive Streamlit Dashboard (`app.py`), Executed Jupyter Notebook (`notebooks/`), and Formal Report (`reports/`).

---

## 📌 Executive Summary & Problem Statement
Modern businesses collect customer demographic and transactional records. Treating an entire consumer base as a monolithic population leads to untargeted marketing budgets, customer churn, and missed sales opportunities. 

**SegmentIQ** performs an empirical comparative evaluation across three fundamental unsupervised clustering paradigms:
- **Centroid-Based Partitioning:** K-Means
- **Connectivity-Based Hierarchical:** Agglomerative Clustering with Ward's Linkage
- **Density-Based Spatial Clustering:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

All models are trained on standardized feature matrices without data leakage. The single source of truth pipeline (`src/run_experiments.py`) executes benchmarks, validates mathematical metrics, serializes results to `reports/results.json`, and renders high-resolution figures to `reports/figures/`.

---

## 🏆 Key Benchmark Results (Single Source of Truth)

All metrics were computed on standardized customer features (`Annual Income (k$)` and `Spending Score (1-100)`). For DBSCAN, noise points ($-1$) were strictly excluded from internal distance calculations:

| Algorithm | Parameters | Clusters Found (excl. noise) | Noise Count & % | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Runtime (s) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means** | $k=5$, `init='k-means++'` | **5** | **0 (0.0%)** | **0.5547** | **0.5722** | **248.65** | **0.0326 s** | **🏆 Best Overall Algorithm** |
| **Agglomerative** | $k=5$, `linkage='ward'`, `metric='euclidean'` | 5 | 0 (0.0%) | 0.5538 | 0.5779 | 244.41 | 0.0013 s | Validates natural cluster structure (ARI > 0.98) |
| **DBSCAN** | $\varepsilon=0.35$, `min_samples=5` | 6 | 23 (11.5%) | 0.5577 | 0.5106 | 245.36 | 0.0019 s | Superior fraud/anomaly detector |

> **Crucial Methodological Caveat:** Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics inherently favor **compact, spherical, convex geometries** (such as those optimized by K-Means and Ward linkage). Because DBSCAN partitions based on density connectivity and isolates boundary points into noise ($-1$), internal distance metrics are not directly comparable between partitioning and density paradigms.

### 🎯 Best Algorithm Verdict: **K-Means ($k=5$)**
1. **Mathematical Cluster Quality:** Delivers the highest Calinski-Harabasz variance ratio (**248.65**) and an optimal Silhouette score (**0.5547**).
2. **Computational Speed:** Executes in **0.0326 seconds**, scaling linearly $\mathcal{O}(k \cdot N \cdot d \cdot i)$ compared to Hierarchical clustering's $\mathcal{O}(N^2 \log N)$ distance matrix cost.
3. **Operational Banking/Retail Fit:** Forms mutually exclusive, convex clusters with clearly defined centroids that directly map onto commercial marketing personas without leaving customers in unassigned noise.

---

## 👥 Discovered Customer Personas (Rule-Based Discovery)

Using K-Means ($k=5$), personas were derived using explicit rules comparing cluster feature means to portfolio averages normalized by standard deviation ($\pm 0.5$ standard deviations):

1. **Cluster 0: Average Income, Average Spending (Mainstream Middle)** &mdash; 81 customers (40.5%)
   - *Profile:* Average earnings ($\approx \$55.3k$), moderate spending score ($\approx 49.5$).
   - *Suggested Action:* Standard seasonal sales newsletters, family event promos, and cross-merchant reward points.
2. **Cluster 1: High Income, High Spending (Affluent Spenders)** &mdash; 39 customers (19.5%)
   - *Profile:* High earnings ($\approx \$86.5k$), high spending score ($\approx 82.1$).
   - *Suggested Action:* Exclusive VIP tier invitations, concierge loyalty perks, and early-access luxury retail promotions.
3. **Cluster 2: Low Income, High Spending (Aspirational Trendsetters)** &mdash; 22 customers (11.0%)
   - *Profile:* Lower earnings ($\approx \$25.7k$), high spending score ($\approx 79.4$).
   - *Suggested Action:* Flexible installment credit plans (BNPL), seasonal flash discounts, and youth lifestyle brand promotions.
4. **Cluster 3: High Income, Low Spending (Cautious Affluent)** &mdash; 35 customers (17.5%)
   - *Profile:* High earnings ($\approx \$88.2k$), low spending score ($\approx 17.1$).
   - *Suggested Action:* Targeted premium quality campaigns, high-value investment partnerships, and personalized value-oriented messaging.
5. **Cluster 4: Low Income, Low Spending (Frugal Budgeters)** &mdash; 23 customers (11.5%)
   - *Profile:* Lower earnings ($\approx \$26.3k$), conservative spending score ($\approx 20.9$).
   - *Suggested Action:* Value-pack retail coupons, essential utility offers, and zero-fee loyalty rewards to drive repeat visits.

---

## 📂 Repository Directory Structure

```text
customer-segmentation-project/
├── app.py                         # Streamlit multi-page web dashboard (imports from src/)
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # Comprehensive project documentation
├── generate_report_docx.py        # Automated report generator pulling from results.json
├── build_and_execute_mall_notebook.py # Automated notebook compiler and executor
├── data/
│   ├── Mall_Customers.csv         # Authentic Kaggle dataset (200 records)
│   ├── sample_synthetic.csv       # Fallback synthetic demo dataset
│   └── README.md                  # Dataset origin and manual download instructions
├── src/
│   ├── __init__.py                # Package root
│   ├── data_loader.py             # Data loading, header normalization & validation
│   ├── preprocessing.py          # StandardScaler, imputation, encoding & leak-free pipeline
│   ├── clustering.py              # K-Means, Agglomerative, DBSCAN wrappers & timers
│   ├── evaluation.py              # Metric validity guards, noise filters & comparison tables
│   ├── visualization.py           # Plotly (app) & Matplotlib/Seaborn (reports) helpers
│   └── run_experiments.py         # Single source of truth: writes results.json & figures
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py        # Tests for headers, formats & error handling
│   ├── test_preprocessing.py      # Tests for scaling, imputation, encoding & reuse
│   ├── test_clustering.py         # Tests for algorithms, seed reproducibility & edge cases
│   ├── test_evaluation.py         # Tests for validity guards (1 cluster, all-noise, k=n)
│   └── test_app_smoke.py          # Smoke test confirming app.py starts without exceptions
├── notebooks/
│   └── customer_segmentation_comparative_study.ipynb # Executed notebook with saved outputs
└── reports/
    ├── comparative_study_report.docx # Formal editable DOCX report (cover, tables, embedded figures)
    ├── comparative_study_report.pdf  # Compiled PDF report
    ├── results.json               # Single source of truth JSON payload
    └── figures/                   # High-resolution generated report figures (Fig 1 to Fig 7)
        ├── fig1_eda_distributions.png
        ├── fig2_elbow_and_silhouette.png
        ├── fig3_dendrogram.png
        ├── fig4_k_distance.png
        ├── fig5_clustering_scatter_comparison.png
        ├── fig6_metric_bars.png
        └── fig7_cluster_profiles.png
```

---

## 🚀 Setup & Execution Guide (Windows PowerShell & VS Code)

### 1. Prerequisites
Ensure Python 3.10+ (tested on Python 3.12) is installed and accessible in your PowerShell terminal.

### 2. Virtual Environment Setup
Open Windows PowerShell in the project directory:

```powershell
# Navigate to the project directory
cd c:\Users\Shahuraj\Desktop\SegmentIQ

# If script execution is blocked on Windows, bypass for current process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Create Python virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install all pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite (Pytest)
Execute all 29 unit and integration tests across data loading, preprocessing, clustering, evaluation, and app smoke-testing:

```powershell
pytest tests/ -v
```
*(All 29 tests pass with 100% compliance).*

### 4. Execute the Single Source of Truth Pipeline
Re-run the default benchmark experiments to re-generate `reports/results.json` and all figures in `reports/figures/`:

```powershell
python src/run_experiments.py
```

### 5. Launch the Streamlit Web Dashboard
Launch the interactive web application:

```powershell
streamlit run app.py
```
Open your web browser at: **`http://localhost:8501`**.

### 6. View / Re-Run the Jupyter Notebook
Open the notebook in VS Code or Jupyter:

```powershell
jupyter notebook notebooks/customer_segmentation_comparative_study.ipynb
```
*(The notebook is already pre-executed top-to-bottom with all markdown explanations, tables, and figures saved).*

### 7. Re-generate the Editable DOCX & PDF Report
```powershell
python generate_report_docx.py
```

---

## 🛠️ Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Activate.ps1 cannot be loaded because running scripts is disabled`** | Windows PowerShell execution policy restricts script execution by default. | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then re-run `.\venv\Scripts\Activate.ps1`. |
| **`Port 8501 is already in use`** | A previously running Streamlit instance is occupying port 8501. | Launch on an alternative port: `streamlit run app.py --server.port 8502`. |
| **`No module named 'src'`** | Python execution path did not include the repository root. | Ensure virtual environment is active or run: `python -m src.run_experiments`. |
| **`FileNotFoundError: Mall_Customers.csv`** | Dataset missing from `data/` directory. | The application automatically falls back to `data/sample_synthetic.csv`. Alternatively, download `Mall_Customers.csv` following instructions in `data/README.md`. |
| **`Kernel selection in VS Code`** | VS Code opened the notebook with a global Python kernel instead of `venv`. | Click the kernel picker in the top-right corner of VS Code, select **Python Environments...**, and choose `venv\Scripts\python.exe`. |
