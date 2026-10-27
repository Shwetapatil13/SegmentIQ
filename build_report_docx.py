"""
Build both SegmentIQ_Mini_Project_Report.docx and SegmentIQ_Mini_Project_Report.pdf
matching the exact structure, font, headers, and pages of the reference report.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def build_docx():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # Helper to add normal styled paragraph
    def p_text(text, bold_prefix="", space_after=6, line_spacing=1.15):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.name = "Arial"
            r_bold.font.size = Pt(10.5)
            r_bold.font.bold = True
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        return p

    def add_heading_styled(text, level=1):
        h = doc.add_heading(level=level)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return h

    # ---------------- PAGE 1: TITLE PAGE ----------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_after = Pt(4)
    r = p_inst.add_run("Society for Computer Technology & Research (SCTR)\n")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    
    r2 = p_inst.add_run("Pune Institute of Computer Technology (PICT), Pune\n\n")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(14)
    r2.font.bold = True

    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_after = Pt(18)
    r_dept = p_dept.add_run("Computer Engineering (CE)\nThird Year B. Tech (T. Y. B. Tech)\nAcademic Year 2026–27\n\n[2615PE31]: Data Warehousing and Mining Lab\n\n\n")
    r_dept.font.name = "Times New Roman"
    r_dept.font.size = Pt(12)

    # Mini Project Report Title
    p_report = doc.add_paragraph()
    p_report.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_report.paragraph_format.space_after = Pt(14)
    r_rep = p_report.add_run("MINI PROJECT REPORT\n\n")
    r_rep.font.name = "Times New Roman"
    r_rep.font.size = Pt(14)
    r_rep.font.bold = True

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(40)
    r_title = p_title.add_run("Customer Segmentation Using the K-Means\nClustering Algorithm (SegmentIQ)\n\n\n")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(16)
    r_title.font.bold = True

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(8)
    r_sub = p_sub.add_run("Submitted by\n\n")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(12)
    r_sub.font.bold = True

    r_names = p_sub.add_run("Shweta Patil    [31363]\nSamruddhi Pharakate    [31365]\n\n\n")
    r_names.font.name = "Times New Roman"
    r_names.font.size = Pt(12)

    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_g1 = p_guide.add_run("Guided by\n\n")
    r_g1.font.name = "Times New Roman"
    r_g1.font.size = Pt(12)
    r_g1.font.bold = True
    r_g2 = p_guide.add_run("Prof. Urmila Pawar\n")
    r_g2.font.name = "Times New Roman"
    r_g2.font.size = Pt(12)

    doc.add_page_break()

    # ---------------- PAGE 2: ABSTRACT & CONTENTS ----------------
    add_heading_styled("Abstract")
    p_text(
        "Modern retail and e-commerce enterprises capture substantial volumes of customer demographic and behavioral transaction records. "
        "Grouping customers with similar purchasing characteristics can reveal meaningful consumption patterns and support business tasks such as "
        "targeted marketing, customer retention, inventory optimization, and personalized promotions. This mini project applies the K-Means clustering "
        "algorithm to the Mall Customer Segmentation Dataset (200 records, 5 columns) using key numerical features including Age, Annual Income, "
        "and Spending Score. Features are standardized with StandardScaler, and the optimal number of clusters is chosen using the Elbow Method (inertia) "
        "and Silhouette Score analysis. Results are visualized through feature distribution plots, a correlation heatmap, pairwise feature scatter plots, "
        "and a PCA-based cluster scatter plot. Cluster profiles are interpreted by comparing average feature values and inspecting representative customer "
        "cohorts to derive distinct commercial personas. The project is implemented in Python on Google Colab / local environment using Pandas, NumPy, "
        "Matplotlib, Seaborn, and Scikit-learn."
    )
    p_kw = doc.add_paragraph()
    r_kw_title = p_kw.add_run("Keywords: ")
    r_kw_title.font.bold = True
    r_kw_title.font.name = "Arial"
    r_kw_title.font.size = Pt(10.5)
    r_kw = p_kw.add_run("unsupervised learning, K-Means clustering, customer segmentation, PCA, silhouette score, consumer behavior analysis.")
    r_kw.font.name = "Arial"
    r_kw.font.size = Pt(10.5)

    add_heading_styled("Contents")
    toc_items = [
        ("Abstract", "i"),
        ("1  Introduction", "1"),
        ("2  Problem Statement", "1"),
        ("3  Objectives", "1"),
        ("4  Dataset Description", "1"),
        ("5  Methodology", "1"),
        ("6  Algorithm Used", "2"),
        ("7  Exploratory Data Analysis and Visualizations", "2"),
        ("8  Expected Outcomes", "6"),
        ("9  Applications", "6"),
        ("10 Software and Hardware Requirements", "6"),
        ("11 Limitations", "7"),
        ("12 Conclusion", "7"),
        ("References", "7")
    ]
    
    t_toc = doc.add_table(rows=0, cols=2)
    t_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_toc.autofit = False
    for item, page in toc_items:
        row = t_toc.add_row()
        c0, c1 = row.cells
        c0.width = Inches(5.0)
        c1.width = Inches(1.5)
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(item)
        r0.font.name = "Arial"
        r0.font.size = Pt(10.5)
        if item in ["Abstract", "References"]:
            r0.font.bold = True
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(page)
        r1.font.name = "Arial"
        r1.font.size = Pt(10.5)

    doc.add_page_break()

    # ---------------- PAGE 3: INTRO, PROBLEM, OBJECTIVES, DATASET, METHODOLOGY ----------------
    add_heading_styled("1  Introduction")
    p_text(
        "Retail stores, banking institutions, and e-commerce platforms host a diverse consumer base, where each individual is characterized by "
        "demographic attributes and spending habits such as annual income, spending score, and age. Identifying customers with homogeneous shopping "
        "behaviors enables businesses to understand consumption patterns and organize shoppers into distinct, actionable cohorts. This mini project "
        "applies unsupervised machine learning, specifically the K-Means clustering algorithm, to group customers based on their numerical demographic "
        "and behavioral traits. No predefined labels are provided during model training; segment personas and profiles are examined and interpreted "
        "after clustering to support strategic managerial decisions."
    )

    add_heading_styled("2  Problem Statement")
    p_text(
        "Modern commercial platforms contain large collections of customers exhibiting varied purchasing behaviors. Treating all consumers as a single "
        "homogeneous group leads to suboptimal promotional spending, customer churn, and generic engagement strategies. The problem is to apply the "
        "K-Means clustering algorithm to segment customers based on features such as annual income, spending score, and age, using unsupervised machine "
        "learning to discover coherent customer groups without predefined supervisory labels during model training."
    )

    add_heading_styled("3  Objectives")
    objs = [
        "1. Explore and analyze retail customer demographic and spending data.",
        "2. Identify similarities between customer profiles using their financial and behavioral characteristics.",
        "3. Implement the K-Means clustering algorithm on standardized customer attributes.",
        "4. Determine a suitable number of clusters using the Elbow Method (Inertia SSE) and Silhouette Score analysis.",
        "5. Visualize the resulting clusters using data visualization techniques and Principal Component Analysis (PCA).",
        "6. Interpret the characteristics and commercial implications of each discovered customer cluster.",
        "7. Examine how the discovered clusters translate into distinct marketing personas to guide enterprise strategies."
    ]
    for o in objs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(o)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)

    add_heading_styled("4  Dataset Description")
    p_text(
        "The project utilizes the Mall Customer Segmentation dataset comprising 200 customer records and 5 attributes. The attributes include "
        "CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1–100). CustomerID is an arbitrary identifier carrying zero behavioral "
        "variance and is excluded from training. Gender represents a categorical demographic attribute. The model relies primarily on the numerical "
        "attributes: Age, Annual Income (k$), and Spending Score (1–100). In particular, Annual Income and Spending Score provide clear indicators "
        "of purchasing capability and shopping propensity, enabling accurate commercial segmentation."
    )

    add_heading_styled("5  Methodology")
    p_text("1. Data loading: import the Mall Customers dataset into Google Colab / Python using Pandas.")
    p_text("2. Data exploration: inspect dataset dimensions, column names, data types, summary statistics, missing values, and duplicate records.")
    p_text("3. Data preprocessing: remove irrelevant identifier columns (CustomerID), verify zero missing records, and handle potential anomalies.")
    p_text("4. Feature selection: select numerical attributes (Annual Income, Spending Score, and Age) for distance-based clustering.")
    p_text("5. Feature scaling: apply StandardScaler to standardize selected features to mean = 0 and variance = 1 prior to clustering.")
    p_text("6. Optimal cluster selection: utilize the Elbow Method (Inertia SSE curve) and Silhouette Score to determine the optimal number of clusters (K = 5).")
    p_text("7. Model training: execute K-Means clustering with k-means++ initialization to group customers based on standardized attribute similarity.")
    p_text("8. Visualization: generate pairwise feature scatter plots, correlation heatmaps, and a 2D PCA projection of the clusters.")
    p_text("9. Cluster interpretation: compute mean feature values for each cluster and profile the resulting customer personas.")
    p_text("10. Evaluation: assess cluster coherence using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.")
    p_text("11. Result export: serialize the customer records along with their assigned cluster labels to a structured CSV file.")

    add_heading_styled("6  Algorithm Used")
    p_text(
        "K-Means is an unsupervised machine learning algorithm that partitions N observations into K predefined clusters, where each cluster is "
        "parameterized by its centroid (the mean vector of all points assigned to that cluster). The algorithm proceeds iteratively as follows:\n"
        "1. Select the number of clusters, K (e.g., K = 5 based on Elbow and Silhouette analysis).\n"
        "2. Initialize K cluster centroids using the k-means++ heuristic to ensure well-separated initial centers.\n"
        "3. Assign each customer vector to its nearest centroid based on Euclidean distance in the standardized feature space:\n"
        "      d(x, c_j) = || x - c_j ||_2\n"
        "4. Recalculate each centroid as the arithmetic mean of all customer points assigned to that cluster:\n"
        "      c_j = (1 / |S_j|) * sum_{x in S_j} x\n"
        "5. Repeat the assignment and centroid update steps iteratively until assignments stabilize or the maximum iteration threshold is reached.\n\n"
        "The objective of K-Means is to minimize the within-cluster sum of squared errors (WCSS), formally known as inertia:\n"
        "      J = sum_{j=1}^K sum_{x in S_j} || x - c_j ||^2"
    )

    add_heading_styled("7  Exploratory Data Analysis and Visualizations")
    p_text(
        "Before applying the clustering algorithm, exploratory data analysis is conducted to examine attribute distributions and inter-feature "
        "correlations. The distribution of Annual Income across customers is depicted in Figure 1. Most customers exhibit moderate incomes centered "
        "between $50k and $80k, with fewer shoppers at the extreme low and high income spectrums, indicating a balanced distribution across economic brackets."
    )

    # Insert Figure 1
    fig1_path = os.path.join("reports", "figures", "report_matched", "fig1_distribution_annual_income.png")
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(5.5))
        cap1 = doc.add_paragraph("Figure 1: Distribution of annual income in the dataset.")
        cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap1.runs[0].font.size = Pt(9.5)
        cap1.runs[0].font.italic = True

    p_text(
        "The distributions of the numerical features (Age, Annual Income, and Spending Score) are displayed in Figure 2. While Age and Annual Income "
        "display relatively bell-shaped, slightly right-skewed profiles, Spending Score is uniformly spread across the 1 to 100 range. Because the numerical "
        "magnitudes of Annual Income ($15k to $137k) exceed the 1–100 scale of Spending Score and Age, feature scaling via StandardScaler is critical. "
        "Without scaling, attributes with larger numeric variances would artificially dominate the Euclidean distance calculations."
    )

    # Insert Figure 2
    fig2_path = os.path.join("reports", "figures", "report_matched", "fig2_feature_distributions.png")
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(6.0))
        cap2 = doc.add_paragraph("Figure 2: Distributions of numerical features used for clustering.")
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap2.runs[0].font.size = Pt(9.5)
        cap2.runs[0].font.italic = True

    p_text(
        "The correlation heatmap in Figure 3 illustrates linear relationships between the demographic and spending features. Annual Income and Spending "
        "Score exhibit near-zero linear correlation (r = 0.01), confirming that earning power alone does not dictate spending willingness. A mild negative "
        "correlation is observed between Age and Spending Score (r = -0.33), suggesting younger shoppers generally exhibit higher spending scores."
    )

    # Insert Figure 3
    fig3_path = os.path.join("reports", "figures", "report_matched", "fig3_correlation_heatmap.png")
    if os.path.exists(fig3_path):
        doc.add_picture(fig3_path, width=Inches(4.8))
        cap3 = doc.add_paragraph("Figure 3: Correlation heatmap of customer features.")
        cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap3.runs[0].font.size = Pt(9.5)
        cap3.runs[0].font.italic = True

    p_text(
        "Pairwise relationships among selected features are plotted in Figure 4. The scatter plots visualize relationships between (i) Annual Income "
        "and Spending Score, (ii) Age and Spending Score, and (iii) Age and Annual Income. The Annual Income vs. Spending Score scatter plot clearly "
        "exhibits five natural visual clusters in data space, representing high/low combinations of income and spending capability."
    )

    # Insert Figure 4
    fig4_path = os.path.join("reports", "figures", "report_matched", "fig4_pairwise_scatter.png")
    if os.path.exists(fig4_path):
        doc.add_picture(fig4_path, width=Inches(6.2))
        cap4 = doc.add_paragraph("Figure 4: Pairwise scatter plots of selected customer features: Annual Income vs. Spending Score, Age vs. Spending Score, and Age vs. Annual Income.")
        cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap4.runs[0].font.size = Pt(9.5)
        cap4.runs[0].font.italic = True

    p_text(
        "After standardizing features and fitting K-Means (K = 5), the resulting customer clusters are projected onto the first two principal "
        "components using Principal Component Analysis (PCA), as shown in Figure 5. The clusters are clearly partitioned with minimal overlap, "
        "confirming that K-Means successfully identifies coherent customer cohorts in the multidimensional space."
    )

    # Insert Figure 5
    fig5_path = os.path.join("reports", "figures", "report_matched", "fig5_kmeans_pca_clusters.png")
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(5.5))
        cap5 = doc.add_paragraph("Figure 5: Visualization of the K-Means clusters projected onto two principal components using PCA.")
        cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap5.runs[0].font.size = Pt(9.5)
        cap5.runs[0].font.italic = True

    add_heading_styled("8  Expected Outcomes")
    p_text(
        "The project produces distinct, well-separated groups of customers characterized by distinct purchasing behaviors and income profiles:\n"
        "• Cluster 0 (Mainstream Middle / Moderate Earners): Moderate income and moderate spending score; constitutes the core baseline customer volume.\n"
        "• Cluster 1 (Affluent Spenders / VIP Champions): High annual income and high spending score; represents premium targets for luxury offerings.\n"
        "• Cluster 2 (Aspirational Trendsetters): Low annual income but high spending score; responsive to brand marketing, installment purchases, and impulse deals.\n"
        "• Cluster 3 (Cautious Affluent / Potential Savers): High annual income but low spending score; high-value prospects needing targeted value propositions.\n"
        "• Cluster 4 (Frugal Budgeters): Low annual income and low spending score; highly price-sensitive shoppers responsive to discounts and essential goods."
    )

    add_heading_styled("9  Applications")
    apps = [
        "• Targeted marketing campaigns: tailoring personalized email and SMS promotions to specific customer cohorts.",
        "• Personalized product recommendation systems: recommending premium or budget-friendly items based on cluster identity.",
        "• Loyalty program design: creating tiered reward structures (e.g., Platinum for VIP champions, cashback for budget shoppers).",
        "• Churn reduction & customer retention: identifying disengaged high-income customers and delivering re-engagement incentives.",
        "• Retail inventory management & merchandising: stocking regional retail branches based on the predominant consumer segment.",
        "• Credit card & banking product positioning: offering tailored credit limits and rewards programs matching income and spending patterns."
    ]
    for a in apps:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(a)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)

    add_heading_styled("10 Software and Hardware Requirements")
    p_text(
        "Software requirements: Python 3, Google Colab or Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.\n\n"
        "Hardware requirements: A computer with an active internet connection, a modern web browser, and Google Colab's hosted Python runtime "
        "or a standard local personal computer with 4 GB RAM and a dual-core processor. No specialized hardware, TPU, or GPU is required for this dataset."
    )

    add_heading_styled("11 Limitations")
    limits = [
        "• K-Means requires the user to specify the number of clusters (K) in advance, requiring empirical diagnostic tools.",
        "• The algorithm assumes spherical cluster geometries and isotropic variance, which may not capture non-linear manifold topologies.",
        "• Cluster results are sensitive to outlier observations and feature scaling, making preprocessing and standardization mandatory.",
        "• PCA reduces multidimensional customer attributes to two dimensions for visual inspection and does not capture 100% of the dataset variance.",
        "• The dataset provides a static cross-sectional snapshot of customer records and does not account for temporal changes in transaction behaviors."
    ]
    for l in limits:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(l)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)

    add_heading_styled("12 Conclusion")
    p_text(
        "This mini project demonstrates the successful application of unsupervised machine learning to retail customer segmentation using the K-Means "
        "clustering algorithm. By analyzing numerical demographic and spending attributes—specifically Age, Annual Income, and Spending Score—the model "
        "groups shoppers into five cohesive, commercially viable segments. Exploratory analysis of feature distributions and correlations validated the "
        "necessity of feature standardization using StandardScaler. The Elbow Method and Silhouette Score confirmed K = 5 as the optimal cluster count, "
        "achieving strong internal cluster coherence (Silhouette Score = 0.5547). The 2D PCA projection validated clear spatial separation between the "
        "discovered cohorts. The project delivers a practical, end-to-end framework encompassing data exploration, preprocessing, scaling, unsupervised "
        "clustering, diagnostic evaluation, and persona profiling, providing actionable intelligence for retail strategy."
    )

    add_heading_styled("References")
    refs = [
        "1. S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed., Pearson, 2021.",
        "2. C. M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006.",
        "3. F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "4. J. MacQueen, \"Some Methods for Classification and Analysis of Multivariate Observations,\" in Proc. Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1967, pp. 281–297.",
        "5. P. J. Rousseeuw, \"Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,\" Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987."
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(ref)
        r.font.name = "Arial"
        r.font.size = Pt(10)

    out_docx = os.path.join("reports", "SegmentIQ_Mini_Project_Report.docx")
    doc.save(out_docx)
    print(f"DOCX created at {out_docx}")
    return out_docx

if __name__ == "__main__":
    build_docx()
