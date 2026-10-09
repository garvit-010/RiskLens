"""
Generate high-resolution architecture diagram (docs/architecture.png)
matching TASK.MD Section 9.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_architecture_diagram():
    output_path = Path("docs/architecture.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Coordinates space: [0, 16] x [0, 9]
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Title & Subtitle
    plt.text(8.0, 8.4, "RiskLens: System Architecture & Data Flow", 
             fontsize=22, fontweight='bold', color='#F3F4F6', ha='center', va='center', fontfamily='sans-serif')
    plt.text(8.0, 7.95, "Unified NLP News-to-Risk Engine with Wholesale Banking Stress Testing (S&P Global & Crisil)", 
             fontsize=12, color='#9CA3AF', ha='center', va='center', fontfamily='sans-serif')

    # Helper function for cards
    def draw_box(x, y, w, h, title, subtitle, items, bg_color='#111827', border_color='#3B82F6', title_color='#60A5FA'):
        rect = patches.FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.15,rounding_size=0.25",
                                      linewidth=1.8, edgecolor=border_color, facecolor=bg_color, zorder=2)
        ax.add_patch(rect)
        
        # Header title
        ax.text(x + w/2, y + h - 0.45, title, fontsize=12, fontweight='bold', color=title_color, ha='center', va='center')
        if subtitle:
            ax.text(x + w/2, y + h - 0.8, subtitle, fontsize=8.5, color='#9CA3AF', style='italic', ha='center', va='center')
            start_item_y = y + h - 1.25
        else:
            start_item_y = y + h - 0.95
            
        for i, item in enumerate(items):
            item_y = start_item_y - (i * 0.42)
            # small badge/bullet
            ax.add_patch(patches.Circle((x + 0.35, item_y), 0.06, facecolor=border_color, zorder=3))
            ax.text(x + 0.55, item_y, item, fontsize=9.2, color='#E5E7EB', va='center')

    # Draw Stage Columns
    # Stage 1: Sources
    draw_box(0.5, 3.2, 2.4, 3.8, 
             "1. DATA SOURCES", "Multi-modal Text",
             ["Kaggle News Headlines", "Kaggle Stock Tweets", "SVB Replay (JSONL)", "NewsAPI (Adapter Stub)"],
             bg_color='#131B2E', border_color='#3B82F6', title_color='#93C5FD')

    # Stage 2: Ingestion
    draw_box(3.4, 3.2, 2.4, 3.8, 
             "2. INGESTION", "Pluggable Adapters",
             ["Unified Document Schema", "ID, Source, Timestamp", "Text, URL, Author, Lang", "Replay Speed Control"],
             bg_color='#131B2E', border_color='#6366F1', title_color='#A5B4FC')

    # Stage 3: Normalization & Preprocessing
    draw_box(6.3, 3.2, 2.5, 3.8, 
             "3. NORMALIZATION", "Clean & Deduplicate",
             ["URL / Cashtag Stripping", "Language Filter (EN)", "Exact Hash Dedup", "RapidFuzz String Match", "Novelty Decay Factor"],
             bg_color='#131B2E', border_color='#8B5CF6', title_color='#C4B5FD')

    # Stage 4: NLP Risk Engine
    draw_box(9.3, 2.7, 3.0, 4.3, 
             "4. NLP RISK ENGINE", "Deep Finance Semantics",
             ["Entity Linking (S&P 100)", "ProsusAI/FinBERT (CUDA)", "VADER Sentiment Baseline", "MiniLM + LR Event Classifier", "11 Banking Risk Classes", "Auditable Impact Formula (1-10)"],
             bg_color='#1E1B4B', border_color='#EC4899', title_color='#F472B6')

    # Stage 5: Signal Store & API
    draw_box(12.8, 4.8, 2.7, 2.2, 
             "5. SIGNAL STORE & API", "Unified Distribution",
             ["JSONL Stream Cache", "FastAPI: GET /signals", "GET /signals/{ticker}", "Audit & Provenance Meta"],
             bg_color='#142B3B', border_color='#06B6D4', title_color='#67E8F9')

    # Downstream Module B: Strategic Stress Testing
    draw_box(12.8, 2.5, 2.7, 2.0, 
             "MODULE B: STRESS TEST", "Wholesale Banking (Primary)",
             ["300-asset Wholesale Port.", "Rate, Spread & PD Shocks", "Reverse Stress Test (Breaks)", "Waterfall & Drilldowns"],
             bg_color='#2A1215', border_color='#EF4444', title_color='#FCA5A5')

    # Downstream Module A: Tactical Rebalancer
    draw_box(12.8, 0.4, 2.7, 1.8, 
             "MODULE A: REBALANCER", "15 S&P 100 Equities",
             ["Sentiment EMA Tilts", "Turnover & Weight Caps", "Stacked Area Time Chart"],
             bg_color='#1F2937', border_color='#10B981', title_color='#6EE7B7')

    # Connective Arrows
    arrow_props = dict(arrowstyle="-|>", color='#9CA3AF', lw=2.2, mutation_scale=18)
    
    # 1 -> 2
    ax.annotate("", xy=(3.4, 5.1), xytext=(2.9, 5.1), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(6.3, 5.1), xytext=(5.8, 5.1), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate("", xy=(9.3, 5.1), xytext=(8.8, 5.1), arrowprops=arrow_props)
    # 4 -> 5
    ax.annotate("", xy=(12.8, 5.9), xytext=(12.3, 5.1), arrowprops=dict(arrowstyle="-|>", color='#06B6D4', lw=2.2, mutation_scale=18))
    # 5 -> Mod B
    ax.annotate("", xy=(14.15, 4.5), xytext=(14.15, 4.8), arrowprops=dict(arrowstyle="-|>", color='#EF4444', lw=2.0, mutation_scale=15))
    # 5 -> Mod A
    ax.annotate("", xy=(14.15, 2.2), xytext=(14.15, 2.5), arrowprops=dict(arrowstyle="-|>", color='#10B981', lw=2.0, mutation_scale=15))

    # Bottom Streamlit UI Badge
    ui_rect = patches.FancyBboxPatch((4.5, 0.5), 7.0, 1.3,
                                     boxstyle="round,pad=0.1,rounding_size=0.2",
                                     linewidth=1.8, edgecolor='#F59E0B', facecolor='#1C1917', zorder=2)
    ax.add_patch(ui_rect)
    ax.text(8.0, 1.45, "UNIFIED STREAMLIT DASHBOARD (python run.py)", 
            fontsize=12, fontweight='bold', color='#FBBF24', ha='center', va='center')
    ax.text(8.0, 1.05, "Live Signal Feed  |  Portfolio Waterfall & Drilldowns  |  Reverse Stress Test  |  Rebalancer Area Chart", 
            fontsize=9.2, color='#E5E7EB', ha='center', va='center')
    ax.text(8.0, 0.72, "Auditable Explainability Panel with Raw Source Proof & Transparent Multiplier Attribution", 
            fontsize=8.5, color='#9CA3AF', ha='center', va='center')

    # Arrow from Engine & Mod B & Mod A into UI
    ax.annotate("", xy=(11.5, 1.5), xytext=(12.8, 3.3), 
                arrowprops=dict(arrowstyle="-|>", color='#F59E0B', lw=1.6, linestyle='dashed', mutation_scale=14))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Architecture diagram generated at: {output_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
