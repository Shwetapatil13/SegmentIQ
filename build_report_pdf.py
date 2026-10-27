"""
Generate SegmentIQ_Mini_Project_Report.pdf using ReportLab
matching the exact typography, headers, footers, structure, figures, and pagination
of the reference report.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically draw headers and footers with correct page numbers.
    Page 1: Cover page (no header/footer).
    Page 2: Abstract & Contents (Footer 'i').
    Page 3+: Body pages (Footer '1', '2', ...).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Page 1: Cover page -> No headers or footers
        if self._pageNumber == 1:
            return

        # Running Header
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#222222"))
        self.drawString(54, 742, "Customer Segmentation Using K-Means")
        self.drawRightString(558, 742, "Mini Project Report")
        
        # Subtle horizontal rule under header
        # self.setStrokeColor(colors.HexColor("#dddddd"))
        # self.setLineWidth(0.5)
        # self.line(54, 735, 558, 735)

        # Running Footer (Page numbers)
        self.setFont("Helvetica", 9)
        if self._pageNumber == 2:
            self.drawCentredString(306, 36, "i")
        else:
            # Page 3 is body page 1
            body_num = str(self._pageNumber - 2)
            self.drawCentredString(306, 36, body_num)
        self.restoreState()


def build_pdf():
    pdf_path = os.path.join("reports", "SegmentIQ_Mini_Project_Report.pdf")
    
    # 54pt margin = 0.75 in, total width = 612, total height = 792
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles matching LaTeX / formal academic template
    title_bold = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        alignment=1, # Center
        textColor=colors.black
    )
    
    cover_meta = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14.5,
        alignment=1,
        textColor=colors.black
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.black
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
        alignment=4, # Justified
        textColor=colors.black
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=3,
        alignment=4,
        textColor=colors.black
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=8,
        textColor=colors.black
    )

    story = []

    # ==================== PAGE 1: COVER PAGE ====================
    story.append(Spacer(1, 35))
    story.append(Paragraph("Society for Computer Technology & Research (SCTR)", title_bold))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Pune Institute of Computer Technology (PICT), Pune", title_bold))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Computer Engineering (CE)", cover_meta))
    story.append(Paragraph("Third Year B. Tech (T. Y. B. Tech)", cover_meta))
    story.append(Paragraph("Academic Year 2026–27", cover_meta))
    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>[2615PE31]: Data Warehousing and Mining Lab</b>", cover_meta))
    
    story.append(Spacer(1, 80))
    story.append(Paragraph("MINI PROJECT REPORT", title_bold))
    story.append(Spacer(1, 16))
    
    proj_title_style = ParagraphStyle(
        'ProjTitle',
        parent=title_bold,
        fontSize=15,
        leading=20
    )
    story.append(Paragraph("Customer Segmentation Using the K-Means<br/>Clustering Algorithm", proj_title_style))
    story.append(Spacer(1, 80))

    story.append(Paragraph("<b>Submitted by</b>", cover_meta))
    story.append(Spacer(1, 6))
    
    # Author table
    authors = [
        ["Shweta Patil", "[31363]"],
        ["Samruddhi Pharakate", "[31365]"]
    ]
    t_auth = Table(authors, colWidths=[130, 60], hAlign='CENTER')
    t_auth.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_auth)

    story.append(Spacer(1, 36))
    story.append(Paragraph("<b>Guided by</b>", cover_meta))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Prof. Urmila Pawar", cover_meta))

    story.append(PageBreak())

    # ==================== PAGE 2: ABSTRACT & CONTENTS ====================
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Abstract</b>", h1_style))
    story.append(Paragraph(
        "Modern retail and e-commerce enterprises capture substantial volumes of customer demographic and "
        "transactional records. Grouping customers with similar purchasing characteristics can reveal meaningful "
        "consumption patterns and support strategic enterprise tasks such as targeted marketing, customer retention, "
        "personalized promotions, and inventory management. This mini project applies the K-Means clustering algorithm "
        "to the Mall Customer Segmentation Dataset (200 rows, 5 columns) using key numerical features: Age, Annual Income, "
        "and Spending Score. Features are standardized with StandardScaler, and the optimal number of clusters is chosen "
        "using the Elbow Method (Within-Cluster Sum of Squares) and Silhouette Score analysis. Results are visualized through "
        "feature distributions, a correlation heatmap, pairwise feature scatter plots, and a PCA-based cluster scatter plot. "
        "Cluster profiles are interpreted by comparing average feature values and evaluating representative customer cohorts, "
        "and the discovered clusters are characterized into five actionable commercial personas. The project is implemented "
        "in Python on Google Colab / local environment using Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Keywords:</b> unsupervised learning, K-Means clustering, customer segmentation, PCA, silhouette score, consumer behavior analysis.",
        body_style
    ))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Contents</b>", h1_style))
    toc_data = [
        ["Abstract", "i"],
        ["1   Introduction", "1"],
        ["2   Problem Statement", "1"],
        ["3   Objectives", "1"],
        ["4   Dataset Description", "1"],
        ["5   Methodology", "1"],
        ["6   Algorithm Used", "2"],
        ["7   Exploratory Data Analysis and Visualizations", "2"],
        ["8   Expected Outcomes", "6"],
        ["9   Applications", "6"],
        ["10 Software and Hardware Requirements", "6"],
        ["11 Limitations", "7"],
        ["12 Conclusion", "7"],
        ["References", "7"]
    ]
    t_toc = Table(toc_data, colWidths=[430, 50], hAlign='CENTER')
    t_toc.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('FONTNAME', (0,0), (0,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,-1), (0,-1), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.8),
        ('TOPPADDING', (0,0), (-1,-1), 2.8),
    ]))
    story.append(t_toc)

    story.append(PageBreak())

    # ==================== PAGE 3: INTRO, PROBLEM, OBJECTIVES, DATASET, METHODOLOGY ====================
    story.append(Paragraph("<b>1   Introduction</b>", h1_style))
    story.append(Paragraph(
        "Retail stores, financial institutions, and e-commerce platforms host large and diverse customer populations, "
        "each described by attributes capturing purchasing power, spending propensity, and demographic traits. "
        "Identifying customers with similar behavioral properties can help organizations understand consumption trends "
        "and organize shoppers into meaningful, actionable segments. This mini project applies unsupervised machine "
        "learning, specifically K-Means clustering, to group customers based on their numerical demographic and spending "
        "features. No predefined class labels are used during model training; segment profiles and commercial personas are "
        "examined and interpreted after clustering to extract operational business value.",
        body_style
    ))

    story.append(Paragraph("<b>2   Problem Statement</b>", h1_style))
    story.append(Paragraph(
        "Modern commercial platforms contain large collections of customers with diverse financial capacities and shopping habits. "
        "Treating all consumers as a monolithic audience leads to wasteful promotional spending, customer attrition, and suboptimal "
        "service experiences. The problem is to apply the K-Means clustering algorithm to group customers based on numerical "
        "features such as annual income, spending score, and age, using unsupervised learning to discover natural behavioral "
        "patterns without predefined category labels during model training.",
        body_style
    ))

    story.append(Paragraph("<b>3   Objectives</b>", h1_style))
    objs = [
        "1. Explore and analyze retail customer demographic and spending data.",
        "2. Identify similarities between customers using their financial and behavioral characteristics.",
        "3. Implement the K-Means clustering algorithm.",
        "4. Determine a suitable number of clusters using the Elbow Method and Silhouette Score.",
        "5. Visualize the resulting clusters using data visualization techniques and Principal Component Analysis (PCA).",
        "6. Interpret the characteristics of each cluster.",
        "7. Examine how the discovered clusters translate into distinct actionable customer personas."
    ]
    for o in objs:
        story.append(Paragraph(o, bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>4   Dataset Description</b>", h1_style))
    story.append(Paragraph(
        "The project uses the Mall Customer Segmentation dataset containing 200 customer records and 5 attributes. "
        "Attributes include CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1–100). The model uses "
        "three numerical features for clustering: Age, Annual Income, and Spending Score. In particular, Annual Income "
        "and Spending Score capture the economic capacity and shopping propensity of each customer. Customer identifiers "
        "are excluded from model training because they provide zero behavioral signal.",
        body_style
    ))

    story.append(Paragraph("<b>5   Methodology</b>", h1_style))
    story.append(Paragraph("1. Data loading: import the customer dataset into Google Colab / Python using Pandas.", bullet_style))
    story.append(Paragraph("2. Data exploration: examine dimensions, column names, data types, summary statistics, missing values, and duplicate records.", bullet_style))

    # ==================== PAGE 4: METHODOLOGY CONT., ALGORITHM, EDA ====================
    story.append(Paragraph("3. Data preprocessing: remove arbitrary index columns (CustomerID), verify data integrity, and check for missing values.", bullet_style))
    story.append(Paragraph("4. Feature selection: select the numerical attributes (Age, Annual Income, and Spending Score) described in Section 4.", bullet_style))
    story.append(Paragraph("5. Feature scaling: apply StandardScaler to standardize the selected features before clustering.", bullet_style))
    story.append(Paragraph("6. Optimal cluster selection: use the Elbow Method and Silhouette Score to help determine an appropriate number of clusters.", bullet_style))
    story.append(Paragraph("7. Model training: apply K-Means (k-means++) to group customers based on standardized feature similarity.", bullet_style))
    story.append(Paragraph("8. Visualization: use pairwise scatter plots, correlation heatmaps, and PCA to visualize the discovered clusters.", bullet_style))
    story.append(Paragraph("9. Cluster interpretation: compare average feature values and profile the resulting customer personas.", bullet_style))
    story.append(Paragraph("10. Evaluation: examine cluster sizes, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.", bullet_style))
    story.append(Paragraph("11. Result export: save the customers and their assigned cluster labels to a structured CSV file.", bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>6   Algorithm Used</b>", h1_style))
    story.append(Paragraph(
        "K-Means is an unsupervised machine learning algorithm that divides data points into a predefined number of clusters, "
        "each represented by a centroid (the mean position of the points assigned to that cluster). The algorithm proceeds as follows:",
        body_style
    ))
    kmeans_steps = [
        "1. Select the number of clusters, K.",
        "2. Initialize K cluster centroids using the k-means++ initialization scheme.",
        "3. Assign each customer to its nearest centroid based on Euclidean distance in the standardized feature space.",
        "4. Recalculate each centroid using the arithmetic mean of its assigned customer vectors.",
        "5. Repeat the assignment and centroid-update steps until assignments stabilize or the stopping condition is reached."
    ]
    for ks in kmeans_steps:
        story.append(Paragraph(ks, bullet_style))

    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "The objective is to minimize the within-cluster sum of squared distances, known as inertia:",
        body_style
    ))
    story.append(Paragraph("<i>J = &Sigma;<sub>j=1</sub><sup>K</sup> &Sigma;<sub>x &isin; S<sub>j</sub></sub> || x - c<sub>j</sub> ||<sup>2</sup></i>", caption_style))

    story.append(Paragraph("<b>7   Exploratory Data Analysis and Visualizations</b>", h1_style))
    story.append(Paragraph(
        "Before clustering, the data is explored to understand its structure and the relationships among the customer features. "
        "The distribution of customer annual income is shown in Figure 1. Most customers exhibit moderate incomes centered between "
        "$50k and $80k, with smaller numbers of lower-income and affluent shoppers, indicating a balanced income distribution.",
        body_style
    ))

    # ==================== PAGE 5: FIGURE 1 ====================
    story.append(PageBreak())
    fig1_path = os.path.join("reports", "figures", "report_matched", "fig1_distribution_annual_income.png")
    if os.path.exists(fig1_path):
        story.append(Spacer(1, 15))
        story.append(Image(fig1_path, width=420, height=270))
        story.append(Paragraph("Figure 1: Distribution of customer annual income in the dataset.", caption_style))
        story.append(Spacer(1, 15))

    story.append(Paragraph(
        "The distributions of the numerical features used for clustering are shown in Figure 2. Attributes such as Age and "
        "Annual Income exhibit moderate spreads with slight right-skewness, while Spending Score is more evenly distributed "
        "across the 1 to 100 range. This makes <b>feature scaling</b> essential, since features with larger numeric ranges "
        "(e.g., Annual Income: $15k–$137k) would otherwise dominate the Euclidean distance computations used by K-Means.",
        body_style
    ))

    # ==================== PAGE 6: FIGURE 2 & HEATMAP INTRO ====================
    story.append(PageBreak())
    fig2_path = os.path.join("reports", "figures", "report_matched", "fig2_feature_distributions.png")
    if os.path.exists(fig2_path):
        story.append(Spacer(1, 10))
        story.append(Image(fig2_path, width=480, height=160))
        story.append(Paragraph("Figure 2: Distributions of numerical features used for clustering.", caption_style))
        story.append(Spacer(1, 14))

    story.append(Paragraph(
        "The correlation heatmap in Figure 3 summarizes linear relationships among the customer features. Annual income and "
        "spending score exhibit near-zero correlation (r = 0.01), indicating that income level alone does not determine spending "
        "willingness. A slight negative correlation is observed between age and spending score (r = -0.33), demonstrating that "
        "younger customers tend to display higher spending scores on average.",
        body_style
    ))

    fig3_path = os.path.join("reports", "figures", "report_matched", "fig3_correlation_heatmap.png")
    if os.path.exists(fig3_path):
        story.append(Spacer(1, 8))
        story.append(Image(fig3_path, width=320, height=230))
        story.append(Paragraph("Figure 3: Correlation heatmap of the customer features.", caption_style))

    # ==================== PAGE 7: SCATTER PLOTS & PCA INTRO ====================
    story.append(PageBreak())
    story.append(Paragraph(
        "Pairwise relationships between selected features are plotted in Figure 4. The scatter plots visualize the relationship "
        "between (i) Annual Income and Spending Score, (ii) Age and Spending Score, and (iii) Age and Annual Income. The scatter plot "
        "of Annual Income vs. Spending Score reveals five prominent visual groupings, indicating natural clustering tendencies in "
        "the customer base.",
        body_style
    ))

    fig4_path = os.path.join("reports", "figures", "report_matched", "fig4_pairwise_scatter.png")
    if os.path.exists(fig4_path):
        story.append(Spacer(1, 4))
        story.append(Image(fig4_path, width=490, height=160))
        story.append(Paragraph("Figure 4: Pairwise scatter plots of selected customer features: Annual Income vs. Spending Score, Age vs. Spending Score, and Age vs. Annual Income.", caption_style))
        story.append(Spacer(1, 8))

    story.append(Paragraph(
        "After standardizing the features and applying K-Means, the resulting clusters are projected onto the first two principal "
        "components using PCA (Figure 5). The clusters are visible as distinct, well-separated regions in the two-dimensional "
        "projection. Because PCA captures the most dominant directions of variance in the feature space, the clear boundary "
        "separation confirms that K-Means produces meaningful, cohesive groupings in the multidimensional space.",
        body_style
    ))

    fig5_path = os.path.join("reports", "figures", "report_matched", "fig5_kmeans_pca_clusters.png")
    if os.path.exists(fig5_path):
        story.append(Spacer(1, 4))
        story.append(Image(fig5_path, width=370, height=265))
        story.append(Paragraph("Figure 5: Visualization of the K-Means clusters projected onto two principal components using PCA.", caption_style))

    # ==================== PAGE 8: EXPECTED OUTCOMES, APPLICATIONS, REQUIREMENTS ====================
    story.append(PageBreak())
    story.append(Paragraph("<b>8   Expected Outcomes</b>", h1_style))
    story.append(Paragraph(
        "The project produces distinct groups of customers with differentiated demographic and spending profiles. "
        "Specifically, the model discovers five well-defined commercial personas: (i) Moderate Earners with average spending (Cluster 0), "
        "(ii) High-Income High-Spenders representing prime VIP champions (Cluster 1), (iii) Low-Income High-Spenders representing "
        "aspirational trendsetters (Cluster 2), (iv) High-Income Low-Spenders representing cautious savers (Cluster 3), and "
        "(v) Low-Income Low-Spenders representing budget-conscious shoppers (Cluster 4). The project demonstrates how unsupervised "
        "learning extracts actionable intelligence and how visualization techniques aid in interpreting customer behavior.",
        body_style
    ))

    story.append(Paragraph("<b>9   Applications</b>", h1_style))
    apps = [
        "• Targeted marketing campaigns tailored to specific customer personas.",
        "• Personalized product recommendation systems for retail and e-commerce.",
        "• Strategic design of tiered loyalty and rewards programs.",
        "• Churn reduction through early detection of declining spenders.",
        "• Customer lifetime value (CLV) optimization and revenue management.",
        "• Retail branch merchandising and regional inventory planning."
    ]
    for a in apps:
        story.append(Paragraph(a, bullet_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>10 Software and Hardware Requirements</b>", h1_style))
    story.append(Paragraph(
        "<b>Software requirements.</b> Python 3, Google Colab or Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Hardware requirements.</b> A computer with an internet connection, a modern web browser, and Google Colab’s hosted "
        "Python runtime or a local Python environment. No specialized hardware or GPU is required for this dataset.",
        body_style
    ))

    # ==================== PAGE 9: LIMITATIONS, CONCLUSION, REFERENCES ====================
    story.append(PageBreak())
    story.append(Paragraph("<b>11 Limitations</b>", h1_style))
    limits = [
        "• K-Means requires the number of clusters to be specified in advance.",
        "• Results depend on feature selection, scaling, and centroid initialization.",
        "• The algorithm assumes spherical cluster geometries and equal variance across clusters.",
        "• PCA reduces the data to two dimensions for visualization and may not preserve all the information used by the model.",
        "• The dataset provides a static cross-sectional snapshot that does not capture evolving customer purchasing trends over time."
    ]
    for l in limits:
        story.append(Paragraph(l, bullet_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>12 Conclusion</b>", h1_style))
    story.append(Paragraph(
        "This mini project demonstrates the application of unsupervised machine learning to retail customer segmentation using "
        "the K-Means clustering algorithm. By analyzing numerical demographic and spending features, the model groups shoppers "
        "according to similarities in their financial capacity and purchasing propensity. Exploratory analysis of feature distributions, "
        "correlations, and pairwise scatter plots motivates feature scaling and PCA-based visualization. The Elbow Method and "
        "Silhouette Score help assess suitable cluster counts, while PCA projection supports interpretation of the final clusters. "
        "The project provides a practical introduction to data preprocessing, feature scaling, clustering, model evaluation, and "
        "visualization using Python. The resulting clusters translate into actionable commercial personas, empowering retail organizations "
        "to optimize marketing expenditures, enhance customer retention, and maximize lifetime customer value.",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>References</b>", h1_style))
    refs = [
        "1. S. Russell and P. Norvig, <i>Artificial Intelligence: A Modern Approach</i>, 4th ed., Pearson, 2021.",
        "2. C. M. Bishop, <i>Pattern Recognition and Machine Learning</i>, Springer, 2006.",
        "3. F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.",
        "4. J. MacQueen, \"Some Methods for Classification and Analysis of Multivariate Observations,\" in <i>Proc. Fifth Berkeley Symposium on Mathematical Statistics and Probability</i>, 1967, pp. 281–297.",
        "5. P. J. Rousseeuw, \"Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis,\" <i>Journal of Computational and Applied Mathematics</i>, vol. 20, pp. 53–65, 1987."
    ]
    for ref in refs:
        story.append(Paragraph(ref, bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF created successfully at {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    build_pdf()
