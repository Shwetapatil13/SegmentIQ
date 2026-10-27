import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

out_dir = os.path.join("reports", "figures", "report_matched")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv("data/Mall_Customers.csv")

# 1. Figure 1: Distribution of Annual Income
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
sns.histplot(df['Annual Income (k$)'], kde=True, color='#6baed6', edgecolor='white', linewidth=1.2, bins=15, ax=ax)
ax.set_title("Distribution of Annual Income", fontsize=12, pad=12)
ax.set_xlabel("Annual Income (k$)", fontsize=10)
ax.set_ylabel("Number of customers", fontsize=10)
plt.tight_layout()
fig1_path = os.path.join(out_dir, "fig1_distribution_annual_income.png")
fig.savefig(fig1_path, dpi=300)
plt.close(fig)

# 2. Figure 2: Distributions of numerical features (Age, Annual Income, Spending Score)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=300)
cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
titles = ['Distribution of age', 'Distribution of annual income', 'Distribution of spending score']
for ax, col, title in zip(axes, cols, titles):
    sns.histplot(df[col], kde=True, color='#9ecae1', edgecolor='white', linewidth=1.0, bins=14, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(col.lower().replace(" (k$)", "").replace(" (1-100)", ""), fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
plt.tight_layout()
fig2_path = os.path.join(out_dir, "fig2_feature_distributions.png")
fig.savefig(fig2_path, dpi=300)
plt.close(fig)

# 3. Figure 3: Correlation heatmap of features
fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
corr = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': ''}, ax=ax, linewidths=0.5)
ax.set_title("Correlation Between Customer Features", fontsize=11, pad=10)
plt.tight_layout()
fig3_path = os.path.join(out_dir, "fig3_correlation_heatmap.png")
fig.savefig(fig3_path, dpi=300)
plt.close(fig)

# 4. Figure 4: Pairwise scatter plots of selected features
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=300)

axes[0].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], color='#3182bd', alpha=0.6, edgecolors='none', s=25)
axes[0].set_title("Annual Income vs Spending Score", fontsize=9.5)
axes[0].set_xlabel("annual income (k$)", fontsize=9)
axes[0].set_ylabel("spending score", fontsize=9)

axes[1].scatter(df['Age'], df['Spending Score (1-100)'], color='#3182bd', alpha=0.6, edgecolors='none', s=25)
axes[1].set_title("Age vs Spending Score", fontsize=9.5)
axes[1].set_xlabel("age", fontsize=9)
axes[1].set_ylabel("spending score", fontsize=9)

axes[2].scatter(df['Age'], df['Annual Income (k$)'], color='#3182bd', alpha=0.6, edgecolors='none', s=25)
axes[2].set_title("Age vs Annual Income", fontsize=9.5)
axes[2].set_xlabel("age", fontsize=9)
axes[2].set_ylabel("annual income (k$)", fontsize=9)

plt.tight_layout()
fig4_path = os.path.join(out_dir, "fig4_pairwise_scatter.png")
fig.savefig(fig4_path, dpi=300)
plt.close(fig)

# 5. Figure 5: K-Means clusters projected onto PCA / 2D
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for cl in range(5):
    mask = labels == cl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'{cl}', color=palette[cl], alpha=0.75, s=35, edgecolors='none')

ax.set_title("Customer Segments (PCA Projection)", fontsize=11, pad=10)
ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% variance)", fontsize=9.5)
ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% variance)", fontsize=9.5)
ax.legend(title="Cluster", frameon=True, fontsize=8.5, title_fontsize=9)
plt.tight_layout()
fig5_path = os.path.join(out_dir, "fig5_kmeans_pca_clusters.png")
fig.savefig(fig5_path, dpi=300)
plt.close(fig)

print("All 5 report-matched figures successfully created!")
