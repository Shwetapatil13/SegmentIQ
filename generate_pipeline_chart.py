import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_pipeline_diagram(output_path="assets/pipeline_architecture.png"):
    # 10 x 15.5 for spacious, perfectly balanced vertical layout
    fig, ax = plt.subplots(figsize=(10, 15.5), dpi=300)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.5)
    
    # Color Palette matching user's reference image
    bg_purple = "#EDE9FE"       # Soft lavender fill
    border_purple = "#7C3AED"   # Vibrant purple border
    text_color = "#1E1B4B"      # Deep navy / charcoal
    subtext_color = "#4C1D95"   # Medium dark purple for subtitle
    gray_badge_bg = "#F3F4F6"   # Neutral light gray pill
    gray_badge_border = "#D1D5DB"
    arrow_color = "#374151"     # Dark charcoal arrow
    
    def draw_pill(center_x, center_y, width, height, text_title, text_sub=""):
        # Draw pill (stadium shape)
        box = patches.FancyBboxPatch(
            (center_x - width/2, center_y - height/2),
            width, height,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=bg_purple,
            edgecolor=border_purple,
            linewidth=2.2,
            zorder=3
        )
        ax.add_patch(box)
        if text_sub:
            ax.text(center_x, center_y + 0.18, text_title, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=text_color, zorder=4, family='sans-serif')
            ax.text(center_x, center_y - 0.22, text_sub, ha='center', va='center',
                    fontsize=9.5, fontweight='normal', color=subtext_color, zorder=4, family='sans-serif')
        else:
            ax.text(center_x, center_y, text_title, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=text_color, zorder=4, family='sans-serif')

    def draw_card(center_x, center_y, width, height, text_title, text_sub=""):
        # Draw rectangular card with rounded corners and subtle shadow
        shadow = patches.FancyBboxPatch(
            (center_x - width/2 + 0.06, center_y - height/2 - 0.06),
            width, height,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor="#E2E8F0",
            edgecolor="none",
            linewidth=0,
            zorder=2
        )
        ax.add_patch(shadow)
        
        box = patches.FancyBboxPatch(
            (center_x - width/2, center_y - height/2),
            width, height,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor="#F5F3FF",
            edgecolor=border_purple,
            linewidth=1.8,
            zorder=3
        )
        ax.add_patch(box)
        
        if text_sub:
            lines = text_sub.split('\n')
            if len(lines) > 1:
                ax.text(center_x, center_y + 0.32, text_title, ha='center', va='center',
                        fontsize=12, fontweight='bold', color=text_color, zorder=4, family='sans-serif')
                ax.text(center_x, center_y - 0.12, lines[0], ha='center', va='center',
                        fontsize=9.5, fontweight='normal', color=subtext_color, zorder=4, family='sans-serif')
                ax.text(center_x, center_y - 0.38, lines[1], ha='center', va='center',
                        fontsize=9.5, fontweight='normal', color=subtext_color, zorder=4, family='sans-serif')
            else:
                ax.text(center_x, center_y + 0.22, text_title, ha='center', va='center',
                        fontsize=12, fontweight='bold', color=text_color, zorder=4, family='sans-serif')
                ax.text(center_x, center_y - 0.22, text_sub, ha='center', va='center',
                        fontsize=9.5, fontweight='normal', color=subtext_color, zorder=4, family='sans-serif')
        else:
            ax.text(center_x, center_y, text_title, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=text_color, zorder=4, family='sans-serif')

    def draw_badge(center_x, center_y, width, height, text_content):
        # Draw light gray intermediate data pill
        badge = patches.FancyBboxPatch(
            (center_x - width/2, center_y - height/2),
            width, height,
            boxstyle="round,pad=0.1,rounding_size=0.1",
            facecolor=gray_badge_bg,
            edgecolor=gray_badge_border,
            linewidth=1.0,
            zorder=3
        )
        ax.add_patch(badge)
        ax.text(center_x, center_y, text_content, ha='center', va='center',
                fontsize=9.5, fontweight='500', color="#374151", zorder=4, family='sans-serif')

    def draw_arrow(start_x, start_y, end_x, end_y):
        ax.annotate(
            '',
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle='->,head_width=0.4,head_length=0.4',
                color=arrow_color,
                lw=1.6
            ),
            zorder=1
        )

    # 1. Top Pill: Input
    y1 = 14.3
    draw_pill(5.0, y1, 5.0, 1.1, "Customer Raw Dataset / CSV Input", "(Mall_Customers.csv • 200 records)")

    # Arrow 1 -> 2
    draw_arrow(5.0, y1 - 0.75, 5.0, 12.85)

    # 2. Preprocessing Card
    y2 = 12.0
    draw_card(5.0, y2, 6.4, 1.25, "CustomerPreprocessor (src/preprocessing.py)", 
              "Median Imputation • Label Encoding • Leak-Free StandardScaler")

    # Arrow 2 -> Badge 1
    draw_arrow(5.0, y2 - 0.8, 5.0, 10.65)

    # Badge 1: Feature Space
    y_badge1 = 10.35
    draw_badge(5.0, y_badge1, 6.0, 0.60, "Standardized Features: Annual Income & Spending Score")

    # Arrow Badge 1 -> Clustering Card
    draw_arrow(5.0, y_badge1 - 0.45, 5.0, 9.35)

    # 3. Clustering Engines Box (Parallel Paradigms)
    y3 = 8.35
    draw_card(5.0, y3, 6.8, 1.55, "Comparative Clustering Engines (src/clustering.py)", 
              "• K-Means (k=5, k-means++) • Ward Agglomerative (k=5)\n• DBSCAN (ε=0.35, min_samples=5 • Outlier Isolation)")

    # Arrow 3 -> Badge 2
    draw_arrow(5.0, y3 - 0.95, 5.0, 6.85)

    # Badge 2: Partitioning & Labels
    y_badge2 = 6.55
    draw_badge(5.0, y_badge2, 6.2, 0.60, "Cluster Assignments & Noise Separation (Outliers = -1)")

    # Arrow Badge 2 -> Evaluation Card
    draw_arrow(5.0, y_badge2 - 0.45, 5.0, 5.55)

    # 4. Metric Evaluator Card
    y4 = 4.65
    draw_card(5.0, y4, 6.6, 1.45, "Metric Evaluation & Guard System (src/evaluation.py)", 
              "Silhouette Score (0.555) • Davies-Bouldin (0.572) • Calinski-Harabasz (248.65)\n(Strict guards for k < 2, k = n, and non-noise DBSCAN points)")

    # Arrow 4 -> Persona Profiler Card
    draw_arrow(5.0, y4 - 0.9, 5.0, 3.45)

    # 5. Persona Profiler Card
    y5 = 2.75
    draw_card(5.0, y5, 6.4, 1.15, "Persona Engine & Commercial Strategy", 
              "5 Rule-Based Personas (±0.5σ Standard Deviation Portfolio Delta)")

    # Arrow 5 -> Bottom Pill
    draw_arrow(5.0, y5 - 0.75, 5.0, 1.65)

    # 6. Bottom Output Pill
    y6 = 0.95
    draw_pill(5.0, y6, 5.8, 1.1, "Interactive Dashboard & Academic Reports", 
              "(Live Streamlit App • Executed Notebook • Formal DOCX/PDF)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Pipeline diagram successfully regenerated at {output_path}")

if __name__ == "__main__":
    create_pipeline_diagram()
