"""
Generate 7-slide presentation PDF (docs/presentation.pdf)
matching TASK.MD Section 15 using ReportLab.
"""
from pathlib import Path
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_presentation():
    output_path = Path("docs/presentation.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 16:9 Landscape slide: 960 x 540 points
    width = 960
    height = 540

    c = canvas.Canvas(str(output_path), pagesize=(width, height))

    # Color Palette
    bg_dark = HexColor("#0A0E17")
    card_bg = HexColor("#111827")
    card_border = HexColor("#1F2937")
    accent_blue = HexColor("#2563EB")
    accent_cyan = HexColor("#06B6D4")
    accent_amber = HexColor("#F59E0B")
    accent_rose = HexColor("#F43F5E")
    text_white = HexColor("#F9FAFB")
    text_muted = HexColor("#9CA3AF")
    text_sub = HexColor("#D1D5DB")

    def draw_slide_bg(slide_num, title, category="S&P GLOBAL & CRISIL CAMPUS HACKATHON"):
        # Dark gradient-like background
        c.setFillColor(bg_dark)
        c.rect(0, 0, width, height, fill=True, stroke=False)
        
        # Subtle header bar
        c.setFillColor(HexColor("#111827"))
        c.rect(0, height - 65, width, 65, fill=True, stroke=False)
        c.setStrokeColor(HexColor("#1F2937"))
        c.setLineWidth(1)
        c.line(0, height - 65, width, height - 65)

        # Header titles
        c.setFillColor(accent_cyan)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(45, height - 25, category.upper())
        
        c.setFillColor(text_white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(45, height - 48, title)

        # Slide footer
        c.setFillColor(HexColor("#1F2937"))
        c.rect(0, 0, width, 30, fill=True, stroke=False)
        c.setFillColor(text_muted)
        c.setFont("Helvetica", 9)
        c.drawString(45, 11, "RiskLens: Explainable News-to-Risk Engine with Wholesale-Banking Stress Testing")
        c.drawRightString(width - 45, 11, f"Slide {slide_num} of 7")

    def draw_card(x, y, w, h, border=card_border, fill=card_bg):
        c.setFillColor(fill)
        c.setStrokeColor(border)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, radius=8, fill=True, stroke=True)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    c.setFillColor(bg_dark)
    c.rect(0, 0, width, height, fill=True, stroke=False)

    # Decorative top glow
    c.setFillColor(accent_blue)
    c.circle(width / 2, height + 80, 250, fill=True, stroke=False)
    c.setFillColor(bg_dark)
    c.rect(0, 0, width, height - 4, fill=True, stroke=False)

    # Main Title Badge
    c.setFillColor(HexColor("#1E293B"))
    c.setStrokeColor(accent_cyan)
    c.setLineWidth(1.5)
    c.roundRect(width/2 - 160, height - 120, 320, 28, radius=14, fill=True, stroke=True)
    c.setFillColor(accent_cyan)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width/2, height - 102, "S&P GLOBAL & CRISIL CAMPUS HACKATHON")

    # Big Title
    c.setFillColor(text_white)
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(width/2, height - 175, "RiskLens")

    c.setFillColor(HexColor("#60A5FA"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height - 212, "Explainable News-to-Risk Engine with Wholesale-Banking Stress Testing")

    c.setFillColor(text_muted)
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height - 245, "Transforming Unstructured Market News into Auditable Credit Shocks & Portfolio Stress Tests")

    # Feature pills
    pills = ["Real-Event Replay (SVB)", "Auditable Impact Formula (1-10)", "300-Asset Wholesale Portfolio", "Reverse Stress Testing"]
    pill_w = 190
    start_x = width/2 - (len(pills) * pill_w + (len(pills)-1)*15) / 2
    for i, pill in enumerate(pills):
        px = start_x + i * (pill_w + 15)
        c.setFillColor(HexColor("#111827"))
        c.setStrokeColor(HexColor("#374151"))
        c.roundRect(px, height - 310, pill_w, 32, radius=6, fill=True, stroke=True)
        c.setFillColor(HexColor("#93C5FD"))
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(px + pill_w/2, height - 290, pill)

    # Candidate Meta Card
    draw_card(width/2 - 250, 60, 500, 130, border=HexColor("#2563EB"), fill=HexColor("#0F172A"))
    c.setFillColor(text_white)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(width/2, 160, "Project Submission Details")
    c.setFillColor(text_sub)
    c.setFont("Helvetica", 11)
    c.drawCentredString(width/2, 136, "Candidate Name: Garvit Kataria")
    c.drawCentredString(width/2, 114, "College / Campus: Engineering Institution")
    c.drawCentredString(width/2, 92, "Stack: Python 3.11 | ProsusAI/FinBERT | MiniLM-L6-v2 | FastAPI | Streamlit")
    c.setFillColor(accent_amber)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(width/2, 72, "Deterministic Offline Demo Mode  |  Zero API Key Dependency")

    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 2: Problem & Solution Approach
    # -------------------------------------------------------------
    draw_slide_bg(2, "Problem Statement & Solution Approach", "Industry Problem & Engineering Strategy")

    # Left Column: Problem
    draw_card(45, 60, 415, 395, border=HexColor("#EF4444"))
    c.setFillColor(HexColor("#EF4444"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(65, 420, "THE PROBLEM IN WHOLESALE RISK")
    
    prob_bullets = [
        ("Information Asymmetry:", "Market-moving credit shocks occur in unstructured news/social feeds hours before official rating revisions."),
        ("Sentiment Fallacy:", "Generic sentiment (positive/negative) is insufficient for risk teams. 'Profits dropped 5%' and 'Bank seized by FDIC' are both negative, but have vastly different institutional consequences."),
        ("Black-Box Disconnect:", "Deep neural nets output probabilities without audit trails. Risk managers cannot defend a black-box 0.89 probability to regulators or credit committees."),
        ("Static Stress Testing:", "Traditional regulatory stress testing (CCAR/DFAST) is quarterly and rigid, incapable of evaluating fast-breaking geopolitical or credit events intraday.")
    ]
    curr_y = 385
    for title, desc in prob_bullets:
        c.setFillColor(HexColor("#FCA5A5"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(65, curr_y, title)
        c.setFillColor(text_sub)
        c.setFont("Helvetica", 9.5)
        # 2 lines per bullet
        words = desc.split()
        l1, l2 = " ".join(words[:10]), " ".join(words[10:])
        c.drawString(65, curr_y - 16, l1)
        c.drawString(65, curr_y - 30, l2)
        curr_y -= 65

    # Right Column: Approach
    draw_card(500, 60, 415, 395, border=HexColor("#10B981"))
    c.setFillColor(HexColor("#10B981"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(520, 420, "RISKLENS SOLUTION APPROACH")

    sol_bullets = [
        ("Pluggable Ingestion Pipeline:", "Multi-source normalization of news and social data into a unified schema with strict deduplication and English filtering."),
        ("Domain-Aware 11-Class Taxonomy:", "Maps unstructured headlines directly into Basel-aligned banking categories (Credit Event, Macro, Legal, Cyber, Supply Chain)."),
        ("Auditable Impact Scoring Formula:", "Transparent 1-10 formula decomposing base severity, sentiment intensity, source credibility, novelty, and cross-source corroboration."),
        ("Dynamic Wholesale Stress Testing:", "Module B maps impact scores directly to spread shocks, PD multipliers, and expected loss across 300 loans, bonds, derivatives, and equities.")
    ]
    curr_y = 385
    for title, desc in sol_bullets:
        c.setFillColor(HexColor("#6EE7B7"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(520, curr_y, title)
        c.setFillColor(text_sub)
        c.setFont("Helvetica", 9.5)
        words = desc.split()
        l1, l2 = " ".join(words[:10]), " ".join(words[10:])
        c.drawString(520, curr_y - 16, l1)
        c.drawString(520, curr_y - 30, l2)
        curr_y -= 65

    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 3: Architecture & Data Flow
    # -------------------------------------------------------------
    draw_slide_bg(3, "System Architecture & End-to-End Pipeline", "Technical Architecture")

    # If architecture diagram image exists, embed preview or visual boxes
    arch_img_path = Path("docs/architecture.png")
    if arch_img_path.exists():
        # Draw image centered
        c.drawImage(str(arch_img_path), 45, 60, width=870, height=395, preserveAspectRatio=True)
    else:
        draw_card(45, 60, 870, 395)
        c.setFillColor(text_white)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(65, 420, "Architecture Pipeline Diagram (docs/architecture.png)")

    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 4: Implementation Highlights & Technical Choices
    # -------------------------------------------------------------
    draw_slide_bg(4, "Implementation Highlights & Defensible Technical Choices", "Engineering & Model Choices")

    tech_cards = [
        ("NLP & Sentiment Engine", HexColor("#3B82F6"), [
            ("ProsusAI/FinBERT (fp16 on CUDA):", "Specialized finance domain BERT (110M params) avoids generic NLP bias."),
            ("Score Mapping:", "Net Sentiment = P(Positive) - P(Negative) in [-1.0, +1.0]."),
            ("VADER Baseline:", "Rule-based lexicon provided for objective benchmark and CPU fallback.")
        ]),
        ("Event Classification", HexColor("#8B5CF6"), [
            ("MiniLM-L6-v2 + Logistic Regression:", "384-dim semantic embeddings with fast linear classifier; 0 training GPU overhead."),
            ("Keyword Fallback:", "Deterministic taxonomy pattern matcher ensures 100% coverage on novel texts."),
            ("Taxonomy:", "11 credit & market risk categories aligned with institutional portfolios.")
        ]),
        ("Transparent Impact Formula", HexColor("#EC4899"), [
            ("Multiplicative Scoring (1 to 10):", "Impact = clip(1, 10, Base_Sev * (0.5 + |Sent|) * Cred * Novelty * Corrob)."),
            ("Corroboration Boost:", "Tracks temporal clusters across independent sources within 48-hour windows."),
            ("Every signal carries explanation:", "Full audit metadata dictionary exposed via API and dashboard.")
        ]),
        ("Wholesale Stress Testing Engine", HexColor("#EF4444"), [
            ("300-Position Wholesale Portfolio:", "Corporate loans, bonds, IR swaps, and equities with realistic seeds."),
            ("Credit Revaluation:", "Delta EL = Delta PD * LGD * EAD + Spread MTM duration-convexity approximation."),
            ("Reverse Stress Testing:", "Bisection search solves for the exact headline impact score that breaks 5% equity.")
        ])
    ]

    card_coords = [(45, 260), (490, 260), (45, 60), (490, 60)]
    for i, (ctitle, cborder, citems) in enumerate(tech_cards):
        cx, cy = card_coords[i]
        draw_card(cx, cy, 425, 185, border=cborder)
        c.setFillColor(cborder)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(cx + 15, cy + 160, ctitle)
        
        iy = cy + 135
        for ititle, idesc in citems:
            c.setFillColor(text_white)
            c.setFont("Helvetica-Bold", 9.5)
            c.drawString(cx + 15, iy, ititle)
            c.setFillColor(text_sub)
            c.setFont("Helvetica", 8.8)
            c.drawString(cx + 15, iy - 13, idesc)
            iy -= 38

    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 5: Key Results & Efficiency Gains vs Naive Baseline
    # -------------------------------------------------------------
    draw_slide_bg(5, "Key Results: Benchmark vs. Naive Baseline & SVB Replay", "Empirical Evaluation")

    # Sentiment Card
    draw_card(45, 230, 420, 225, border=HexColor("#3B82F6"))
    c.setFillColor(accent_cyan)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(65, 425, "1. SENTIMENT BENCHMARK: FinBERT vs. VADER")
    
    # Table headers
    c.setFillColor(HexColor("#374151"))
    c.rect(65, 375, 380, 26, fill=True, stroke=False)
    c.setFillColor(text_white)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(75, 383, "Model")
    c.drawString(180, 383, "Accuracy")
    c.drawString(280, 383, "Macro-F1")
    c.drawString(370, 383, "Domain Fit")

    # Table rows
    rows_s = [
        ("FinBERT", "87.6%", "0.876", "High (Domain Transformer)", HexColor("#60A5FA")),
        ("VADER (Naive)", "57.1%", "0.487", "Baseline (General Lexicon)", HexColor("#9CA3AF"))
    ]
    ry = 345
    for m, acc, f1, fit, col in rows_s:
        c.setFillColor(col)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(75, ry, m)
        c.setFont("Helvetica", 9.5)
        c.drawString(180, ry, acc)
        c.drawString(280, ry, f1)
        c.drawString(370, ry, fit)
        ry -= 26

    c.setFillColor(HexColor("#10B981"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(65, 280, "Gain: +30.5% Accuracy / +0.389 Macro-F1 (Full n=2,264 Subset)")
    c.setFillColor(text_sub)
    c.setFont("Helvetica", 8.8)
    c.drawString(65, 262, "Negative Class Recall: FinBERT 98.7% (F1: 0.91) vs VADER 22.4% (F1: 0.28).")
    c.drawString(65, 248, "Captures critical credit downside where naive lexicons miss 78% of distressed signals.")

    # Event Classifier Card
    draw_card(495, 230, 420, 225, border=HexColor("#8B5CF6"))
    c.setFillColor(HexColor("#C4B5FD"))
    c.setFont("Helvetica-Bold", 13)
    c.drawString(515, 425, "2. EVENT CLASSIFIER: Stratified CV Benchmark")

    c.setFillColor(HexColor("#374151"))
    c.rect(515, 375, 380, 26, fill=True, stroke=False)
    c.setFillColor(text_white)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(525, 383, "Model")
    c.drawString(630, 383, "Macro-F1")
    c.drawString(715, 383, "Credit F1")
    c.drawString(820, 383, "Role")

    rows_e = [
        ("MiniLM+LR (CV)", "0.679", "0.923", "Continuous Prob", HexColor("#A78BFA")),
        ("Keyword Baseline", "0.679", "0.923", "Rule Fallback", HexColor("#9CA3AF"))
    ]
    ry = 345
    for m, f1, cf1, inf, col in rows_e:
        c.setFillColor(col)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(525, ry, m)
        c.setFont("Helvetica", 9.5)
        c.drawString(630, ry, f1)
        c.drawString(715, ry, cf1)
        c.drawString(820, ry, inf)
        ry -= 26

    c.setFillColor(HexColor("#10B981"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(515, 280, "Out-of-Fold Cross-Validation: High Credit Event F1 (0.92)")
    c.setFillColor(text_sub)
    c.setFont("Helvetica", 8.8)
    c.drawString(515, 262, "MiniLM embeddings provide calibrated continuous confidence and generalize")
    c.drawString(515, 248, "to unseen phrases; transparent keyword fallback activates when confidence < 0.55.")

    # Bottom SVB Case Study Card
    draw_card(45, 60, 870, 155, border=HexColor("#F59E0B"))
    c.setFillColor(accent_amber)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(65, 185, "3. LIVE REPLAY CASE STUDY: SILICON VALLEY BANK (MARCH 8-13, 2023)")

    svb_points = [
        ("Event Ingestion & Credibility:", "30 curated chronological headlines & tweets replayed with independent source credibility (0.85 news, 0.55 social)."),
        ("Primary Demo Trigger:", "March 8, 2023: 'SVB stock plunges 60% as bank run fears intensify' triggers Credit Event at max impact 10.0."),
        ("Credit-Aware Sector Targeting:", "Hits Financials (-$42.1M loss, >2.7x other sectors) with 25% spillover to others, producing -$142M net loss (-3.24%)."),
        ("Reverse Stress Test:", "15-step bisection search pinpoints portfolio 2.5% loss breach at Impact 8.85 ($109.7M monetary loss).")
    ]
    sy = 155
    for stitle, sdesc in svb_points:
        c.setFillColor(text_white)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(65, sy, stitle)
        c.setFillColor(text_sub)
        c.setFont("Helvetica", 9)
        c.drawString(240, sy, sdesc)
        sy -= 22

    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 6: Domain Impact & Regulatory Practice
    # -------------------------------------------------------------
    draw_slide_bg(6, "Wholesale Banking Domain Impact & Regulatory Alignment", "Business & Institutional Relevance")

    impact_pillars = [
        ("Credit & Counterparty Risk", HexColor("#3B82F6"), [
            "Early Warning Indicators (EWI): Signals detect credit deterioration before rating agencies issue formal downgrade notices.",
            "PD / Rating Migration: Converts qualitative negative sentiment directly into calibrated rating transitions and expected-loss surges.",
            "Concentration Monitoring: Immediate asset-class, sector, and counterparty exposure aggregation."
        ]),
        ("Supervisory Stress Testing Parallels", HexColor("#8B5CF6"), [
            "ICAAP & Supervisory Practices: Conceptually aligned with scenario analysis practices used in ICAAP and supervisory stress testing.",
            "Reverse Stress Testing (SR 12-7): Empirically solves for boundary conditions where portfolio capital thresholds are breached.",
            "Auditable Lineage: Every stressed valuation links to the exact source headline, timestamp, and model component score."
        ]),
        ("Tactical Capital Allocation", HexColor("#10B981"), [
            "Reusable Core Engine: Powers both strategic stress testing (Module B) and index rebalancing (Module A) from one feed.",
            "Constrained Rebalancing: Rigorous 2%-15% weight boundaries, turnover caps (5%), and sentiment EMA smoothing.",
            "Downstream API Accessibility: Structured REST API allows any enterprise analytics team to subscribe to signals."
        ])
    ]

    for i, (ptitle, pcolor, pitems) in enumerate(impact_pillars):
        px = 45 + i * 295
        draw_card(px, 60, 280, 395, border=pcolor)
        c.setFillColor(pcolor)
        c.setFont("Helvetica-Bold", 12.5)
        c.drawString(px + 15, 420, ptitle)
        
        py = 375
        for pitem in pitems:
            head, body = pitem.split(": ")
            c.setFillColor(text_white)
            c.setFont("Helvetica-Bold", 9.8)
            c.drawString(px + 15, py, head + ":")
            
            c.setFillColor(text_sub)
            c.setFont("Helvetica", 9)
            words = body.split()
            lines = []
            curr = []
            for w in words:
                curr.append(w)
                if len(" ".join(curr)) > 34:
                    lines.append(" ".join(curr[:-1]))
                    curr = [w]
            if curr:
                lines.append(" ".join(curr))
            
            for line_idx, line in enumerate(lines):
                c.drawString(px + 15, py - 14 - (line_idx * 13), line)
            
            py -= (22 + len(lines) * 13)

    c.showPage()

    # -------------------------------------------------------------
    # SLIDE 7: Limitations, Assumptions & Future Roadmap
    # -------------------------------------------------------------
    draw_slide_bg(7, "Limitations, Defensible Assumptions & Future Roadmap", "Governance & Transparency")

    # Left: Defensible Assumptions & Limitations (Honesty Rule)
    draw_card(45, 60, 420, 395, border=HexColor("#F59E0B"))
    c.setFillColor(accent_amber)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(65, 420, "DEFENSIBLE ASSUMPTIONS & LIMITATIONS")

    assumptions = [
        ("Synthetic Wholesale Portfolio:", "300 positions are synthetically generated with fixed random seeds for demonstration. Not actual banking institution holdings."),
        ("First-Order Pricing Approximations:", "Duration/convexity bond pricing, loan expected loss (dPD * LGD * EAD), and linear DV01 swap valuations are first-order proxies, not production full revaluation."),
        ("Sentiment Is Not Causality:", "Headline sentiment correlation does not imply direct market causation. Signals serve as decision-support alerts, not autonomous trade execution."),
        ("Documented Severity Table:", "Base severity table reflects domain assumptions rather than full econometric event-study calibration (honestly documented in docs/ASSUMPTIONS.md).")
    ]
    ay = 385
    for atitle, adesc in assumptions:
        c.setFillColor(HexColor("#FCD34D"))
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(65, ay, atitle)
        c.setFillColor(text_sub)
        c.setFont("Helvetica", 9)
        words = adesc.split()
        l1, l2 = " ".join(words[:11]), " ".join(words[11:])
        c.drawString(65, ay - 14, l1)
        c.drawString(65, ay - 27, l2)
        ay -= 65

    # Right: Roadmap
    draw_card(495, 60, 420, 395, border=HexColor("#3B82F6"))
    c.setFillColor(HexColor("#60A5FA"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(515, 420, "FUTURE ROADMAP & PRODUCTION EXTENSIONS")

    roadmap = [
        ("Live Ingestion Connectors:", "Implement full streaming NewsAPI, Bloomberg B-PIPE, and Reuters Elektron adapters beyond the offline replay adapter."),
        ("Multi-Factor Credit Migration:", "Incorporate multi-factor Merton structural models and transition matrices (e.g. Crisil rating transition tables)."),
        ("Non-Linear Derivatives Pricing:", "Integrate QuantLib or full Monte Carlo simulation for multi-currency cross-asset options and structured credit derivatives."),
        ("Empirical Event Study Calibration:", "Run large-scale econometric calibration against historical CRSP/Compustat abnormal returns to fine-tune severity weights.")
    ]
    ry = 385
    for rtitle, rdesc in roadmap:
        c.setFillColor(HexColor("#93C5FD"))
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(515, ry, rtitle)
        c.setFillColor(text_sub)
        c.setFont("Helvetica", 9)
        words = rdesc.split()
        l1, l2 = " ".join(words[:11]), " ".join(words[11:])
        c.drawString(515, ry - 14, l1)
        c.drawString(515, ry - 27, l2)
        ry -= 65

    c.showPage()
    c.save()
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_presentation()
