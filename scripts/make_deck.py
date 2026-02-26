#!/usr/bin/env python3
"""Generate a beautiful CLV presentation deck.

Creates 15 HTML slides → screenshots via Playwright → PPTX + PDF.
"""

import asyncio
import base64
import json
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DELIVERABLES = ROOT / "deliverables"
PREVIEWS = DELIVERABLES / "slide_previews"
PREVIEWS.mkdir(parents=True, exist_ok=True)


# ── Image helper ────────────────────────────────────────────────────────
def img_b64(path: Path) -> str:
    """Return base64 data URI for a PNG image."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode()
    return f"data:image/png;base64,{encoded}"


# ── Base CSS ────────────────────────────────────────────────────────────
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

html, body {
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
    -webkit-font-smoothing: antialiased;
}

body {
    display: flex;
    flex-direction: column;
}

.slide-dark {
    background: linear-gradient(135deg, #0A1628 0%, #0F2847 50%, #0A1628 100%);
    color: #F8FAFC;
    padding: 72px 96px;
}

.slide-light {
    background: #F8FAFC;
    padding: 72px 96px;
}

.section-tag {
    display: inline-block;
    background: rgba(26, 115, 232, 0.1);
    color: #1A73E8;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 100px;
    margin-bottom: 16px;
}

.section-tag-dark {
    background: rgba(26, 115, 232, 0.25);
    color: #60A5FA;
}

.slide-title {
    font-size: 44px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 8px;
}

.slide-subtitle {
    font-size: 18px;
    font-weight: 400;
    color: #475569;
    margin-bottom: 40px;
}

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
}

.card-accent-top { border-top: 4px solid #1A73E8; }
.card-accent-left { border-left: 4px solid #1A73E8; }

.metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    border-top: 4px solid #1A73E8;
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
}

.metric-label {
    font-size: 13px;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; }
.grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 24px; }

.chart-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: 12px;
}

.banner {
    background: linear-gradient(135deg, #0A1628, #0F2847);
    color: #F8FAFC;
    border-radius: 12px;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
}
"""


def wrap_slide(body_html: str, dark: bool = False) -> str:
    """Wrap slide body HTML in full HTML document."""
    body_class = "slide-dark" if dark else "slide-light"
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><style>{BASE_CSS}</style></head>
<body class="{body_class}">
{body_html}
</body></html>"""


# ── Slide 1: Title ──────────────────────────────────────────────────────
def slide_01_title():
    return wrap_slide("""
<div style="display:flex; flex-direction:column; height:100%; justify-content:center; align-items:center; text-align:center;">
    <div class="section-tag section-tag-dark" style="margin-bottom:32px;">Enterprise Data Science</div>
    <h1 style="font-size:72px; font-weight:900; line-height:1.05; max-width:1200px; margin-bottom:24px;
        background: linear-gradient(135deg, #FFFFFF 0%, #94C4FF 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Customer Lifetime Value<br>Prediction
    </h1>
    <p style="font-size:22px; font-weight:300; color:rgba(248,250,252,0.5); margin-bottom:56px; max-width:700px;">
        A data-driven approach to quantifying future customer value using RFM features, regularized regression, and causal inference
    </p>
    <div style="display:flex; gap:40px; align-items:center; margin-bottom:48px;">
        <div style="text-align:center;">
            <div style="font-weight:600; font-size:16px;">Othmane Zizi</div>
            <div style="font-size:13px; color:rgba(248,250,252,0.4);">261255341</div>
        </div>
        <div style="text-align:center;">
            <div style="font-weight:600; font-size:16px;">Fares Joni</div>
            <div style="font-size:13px; color:rgba(248,250,252,0.4);">261254593</div>
        </div>
        <div style="text-align:center;">
            <div style="font-weight:600; font-size:16px;">Tanmay Giri</div>
            <div style="font-size:13px; color:rgba(248,250,252,0.4);">261272443</div>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:10px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px 20px;">
        <svg width="18" height="18" viewBox="0 0 16 16" fill="rgba(248,250,252,0.5)"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
        <span style="font-size:14px; color:rgba(248,250,252,0.5); font-weight:500;">github.com/othmane-zizi-pro/nwa</span>
    </div>
</div>
""", dark=True)


# ── Slide 2: Context ────────────────────────────────────────────────────
def slide_02_context():
    return wrap_slide("""
<div class="section-tag">Context</div>
<h2 class="slide-title">Business Context &amp; Our Approach</h2>
<p class="slide-subtitle">Understanding customer value is critical for resource allocation and retention strategy</p>

<div class="grid-2" style="flex:1; margin-bottom:24px;">
    <div class="card card-accent-top">
        <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">The Problem</div>
        <div style="font-size:18px; font-weight:600; margin-bottom:12px;">Not all customers are equal</div>
        <p style="font-size:15px; color:#475569; line-height:1.7;">
            E-commerce businesses invest equally across their customer base, but a small segment drives the majority of revenue.
            Without predictive CLV, marketing spend is misallocated and high-value customers may churn unnoticed.
        </p>
    </div>
    <div class="card" style="border-top: 4px solid #00C9A7;">
        <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Our Approach</div>
        <div style="font-size:18px; font-weight:600; margin-bottom:12px;">Predict &rarr; Explain &rarr; Act</div>
        <p style="font-size:15px; color:#475569; line-height:1.7;">
            Build a regression model on RFM + behavioral features to predict 6-month forward CLV.
            Use SHAP for explainability, causal inference for treatment effects, and monitoring for production readiness.
        </p>
    </div>
</div>

<div class="grid-4">
    <div class="metric-card">
        <div class="metric-value" style="color:#1A73E8;">4,266</div>
        <div class="metric-label">Customers</div>
    </div>
    <div class="metric-card" style="border-top-color:#00C9A7;">
        <div class="metric-value" style="color:#00C9A7;">1.07M</div>
        <div class="metric-label">Transactions</div>
    </div>
    <div class="metric-card" style="border-top-color:#F59E0B;">
        <div class="metric-value" style="color:#F59E0B;">8</div>
        <div class="metric-label">Features</div>
    </div>
    <div class="metric-card" style="border-top-color:#6366F1;">
        <div class="metric-value" style="color:#6366F1;">6 Mo</div>
        <div class="metric-label">Prediction Window</div>
    </div>
</div>
""")


# ── Slide 3: Hypotheses ─────────────────────────────────────────────────
def slide_03_hypotheses():
    return wrap_slide("""
<div class="section-tag">Hypotheses</div>
<h2 class="slide-title">Research Hypotheses</h2>
<p class="slide-subtitle">Three testable hypotheses guiding our analysis</p>

<div class="grid-3" style="flex:1;">
    <div class="card" style="border-top: 4px solid #1A73E8; display:flex; flex-direction:column;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:20px;">
            <div style="width:48px; height:48px; border-radius:50%; background:linear-gradient(135deg, #1A73E8, #3B82F6); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:18px; color:white; flex-shrink:0;">H1</div>
            <div style="font-size:17px; font-weight:700;">Monetary Features Dominate</div>
        </div>
        <p style="font-size:15px; color:#475569; line-height:1.7; flex:1;">
            Past spending behavior (Monetary, AvgOrderValue) will be the strongest predictor of future CLV, outweighing frequency and recency signals.
        </p>
        <div style="margin-top:20px; padding:12px 16px; background:rgba(26,115,232,0.06); border-radius:8px;">
            <span style="font-size:13px; font-weight:600; color:#1A73E8;">Verdict: Confirmed</span>
            <span style="font-size:13px; color:#475569;"> &mdash; Monetary importance = 2,136</span>
        </div>
    </div>
    <div class="card" style="border-top: 4px solid #00C9A7; display:flex; flex-direction:column;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:20px;">
            <div style="width:48px; height:48px; border-radius:50%; background:linear-gradient(135deg, #00C9A7, #34D399); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:18px; color:white; flex-shrink:0;">H2</div>
            <div style="font-size:17px; font-weight:700;">Regularization Improves Generalization</div>
        </div>
        <p style="font-size:15px; color:#475569; line-height:1.7; flex:1;">
            Regularized models (Lasso, Ridge) will generalize better than tree-based methods on this dataset due to moderate dimensionality and potential collinearity.
        </p>
        <div style="margin-top:20px; padding:12px 16px; background:rgba(0,201,167,0.06); border-radius:8px;">
            <span style="font-size:13px; font-weight:600; color:#00C9A7;">Verdict: Confirmed</span>
            <span style="font-size:13px; color:#475569;"> &mdash; Lasso R&sup2; = 0.81 vs RF 0.50</span>
        </div>
    </div>
    <div class="card" style="border-top: 4px solid #F59E0B; display:flex; flex-direction:column;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:20px;">
            <div style="width:48px; height:48px; border-radius:50%; background:linear-gradient(135deg, #F59E0B, #FBBF24); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:18px; color:white; flex-shrink:0;">H3</div>
            <div style="font-size:17px; font-weight:700;">High-Frequency &ne; High-CLV</div>
        </div>
        <p style="font-size:15px; color:#475569; line-height:1.7; flex:1;">
            Causal analysis will show that purchase frequency alone does not causally increase CLV; instead, it may reflect selection bias from already-loyal customers.
        </p>
        <div style="margin-top:20px; padding:12px 16px; background:rgba(245,158,11,0.06); border-radius:8px;">
            <span style="font-size:13px; font-weight:600; color:#F59E0B;">Verdict: Supported</span>
            <span style="font-size:13px; color:#475569;"> &mdash; ATE &asymp; &minus;0.17 (log scale)</span>
        </div>
    </div>
</div>
""")


# ── Slide 4: Data ───────────────────────────────────────────────────────
def slide_04_data():
    return wrap_slide("""
<div class="section-tag">Data</div>
<h2 class="slide-title">Data Pipeline</h2>
<p class="slide-subtitle">UCI Online Retail dataset &mdash; from raw transactions to clean customer-level features</p>

<div class="grid-2" style="margin-bottom:24px;">
    <div class="card card-accent-top">
        <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Source</div>
        <ul style="font-size:15px; color:#475569; line-height:2.2; list-style:none;">
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#1A73E8; font-size:18px;">&#x2022;</span> UCI Online Retail (UK e-commerce, 2010&ndash;2011)
            </li>
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#1A73E8; font-size:18px;">&#x2022;</span> 1,067,371 transaction rows
            </li>
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#1A73E8; font-size:18px;">&#x2022;</span> Fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
            </li>
        </ul>
    </div>
    <div class="card" style="border-top: 4px solid #00C9A7;">
        <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Cleaning Steps</div>
        <ul style="font-size:15px; color:#475569; line-height:2.2; list-style:none;">
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#00C9A7; font-size:18px;">&#x2022;</span> Remove cancelled orders (Invoice starts with &lsquo;C&rsquo;)
            </li>
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#00C9A7; font-size:18px;">&#x2022;</span> Drop null CustomerID rows (~25%)
            </li>
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#00C9A7; font-size:18px;">&#x2022;</span> Filter Quantity &gt; 0, UnitPrice &gt; 0
            </li>
            <li style="display:flex; align-items:center; gap:10px;">
                <span style="color:#00C9A7; font-size:18px;">&#x2022;</span> Remove outliers beyond 99th percentile
            </li>
        </ul>
    </div>
</div>

<div class="banner">
    <div style="width:44px; height:44px; border-radius:12px; background:rgba(0,201,167,0.15); display:flex; align-items:center; justify-content:center;">
        <span style="color:#00C9A7; font-size:22px; font-weight:800;">&check;</span>
    </div>
    <div>
        <div style="font-size:18px; font-weight:700;">805,549 clean rows &rarr; 4,266 unique customers</div>
        <div style="font-size:14px; color:rgba(248,250,252,0.5);">24.5% rows removed during cleaning &bull; observation period ending Dec 2010 &bull; 6-month prediction window to Jun 2011</div>
    </div>
</div>
""")


# ── Slide 5: Features ───────────────────────────────────────────────────
def slide_05_features():
    return wrap_slide("""
<div class="section-tag">Features</div>
<h2 class="slide-title">Feature Engineering</h2>
<p class="slide-subtitle">RFM-based features augmented with behavioral signals</p>

<div style="display:flex; gap:24px; flex:1;">
    <div style="flex:1; display:flex; flex-direction:column; gap:16px;">
        <div class="card" style="padding:24px; display:flex; flex-direction:column; gap:14px; flex:1;">
            <div style="font-size:14px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Feature Importance (Lasso)</div>

            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:180px; font-size:14px; font-weight:600; color:#0F172A;">Monetary</div>
                <div style="flex:1; height:28px; background:#EEF2FF; border-radius:6px; overflow:hidden;">
                    <div style="width:100%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:6px;"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:14px; font-weight:700; color:#1A73E8;">2,136</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:180px; font-size:14px; font-weight:600; color:#0F172A;">Frequency</div>
                <div style="flex:1; height:28px; background:#EEF2FF; border-radius:6px; overflow:hidden;">
                    <div style="width:31.9%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:6px;"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:14px; font-weight:700; color:#1A73E8;">681</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:180px; font-size:14px; font-weight:600; color:#0F172A;">AvgOrderValue</div>
                <div style="flex:1; height:28px; background:#EEF2FF; border-radius:6px; overflow:hidden;">
                    <div style="width:25%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:6px;"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:14px; font-weight:700; color:#1A73E8;">534</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:180px; font-size:14px; font-weight:600; color:#0F172A;">AvgBasketSize</div>
                <div style="flex:1; height:28px; background:#EEF2FF; border-radius:6px; overflow:hidden;">
                    <div style="width:19%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:6px;"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:14px; font-weight:700; color:#1A73E8;">407</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:180px; font-size:14px; font-weight:600; color:#0F172A;">NumUniqueProducts</div>
                <div style="flex:1; height:28px; background:#EEF2FF; border-radius:6px; overflow:hidden;">
                    <div style="width:12.9%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:6px;"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:14px; font-weight:700; color:#1A73E8;">276</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:180px; font-size:14px; font-weight:600; color:#0F172A;">Recency</div>
                <div style="flex:1; height:28px; background:#EEF2FF; border-radius:6px; overflow:hidden;">
                    <div style="width:2.6%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:6px;"></div>
                </div>
                <div style="width:60px; text-align:right; font-size:14px; font-weight:700; color:#1A73E8;">55</div>
            </div>
        </div>
    </div>

    <div style="flex:1; display:flex; flex-direction:column; gap:16px;">
        <div class="card" style="border-top: 4px solid #1A73E8; flex:1;">
            <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">RFM Core</div>
            <p style="font-size:15px; color:#475569; line-height:1.8;">
                <strong>Recency</strong> &mdash; days since last purchase<br>
                <strong>Frequency</strong> &mdash; total number of orders<br>
                <strong>Monetary</strong> &mdash; total revenue generated
            </p>
        </div>
        <div class="card" style="border-top: 4px solid #00C9A7; flex:1;">
            <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Behavioral Extensions</div>
            <p style="font-size:15px; color:#475569; line-height:1.8;">
                <strong>AvgOrderValue</strong> &mdash; mean spend per order<br>
                <strong>AvgBasketSize</strong> &mdash; items per transaction<br>
                <strong>NumUniqueProducts</strong> &mdash; product diversity<br>
                <strong>AvgTimeBetweenPurchases</strong> &mdash; purchase cadence<br>
                <strong>Tenure</strong> &mdash; days as customer
            </p>
        </div>
    </div>
</div>

<div class="banner" style="margin-top:24px; background:linear-gradient(135deg, #92400E, #B45309);">
    <span style="font-size:22px;">&#x1F3AF;</span>
    <div>
        <div style="font-size:17px; font-weight:700;">Target Variable: FutureCLV</div>
        <div style="font-size:14px; color:rgba(248,250,252,0.6);">Sum of customer revenue in the 6-month prediction window (Dec 2010 &ndash; Jun 2011)</div>
    </div>
</div>
""")


# ── Slide 6: Modeling ───────────────────────────────────────────────────
def slide_06_modeling():
    return wrap_slide("""
<div class="section-tag">Modeling</div>
<h2 class="slide-title">Modeling Strategy</h2>
<p class="slide-subtitle">Comparing regularized linear models against tree-based ensembles</p>

<div class="grid-2" style="flex:1;">
    <div class="card card-accent-top" style="display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:20px;">Models Evaluated</div>
        <div style="display:flex; flex-direction:column; gap:12px; flex:1;">
            <div style="display:flex; align-items:center; gap:12px; padding:14px 18px; background:#F0F7FF; border-radius:10px; border-left: 3px solid #1A73E8;">
                <div style="font-size:15px; font-weight:600; color:#0F172A; flex:1;">Linear Regression</div>
                <div style="font-size:13px; color:#475569;">Baseline</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px; padding:14px 18px; background:#F0F7FF; border-radius:10px; border-left: 3px solid #1A73E8;">
                <div style="font-size:15px; font-weight:600; color:#0F172A; flex:1;">Ridge Regression</div>
                <div style="font-size:13px; color:#475569;">L2 regularization</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px; padding:14px 18px; background:linear-gradient(90deg, #EFF6FF, #DBEAFE); border-radius:10px; border-left: 3px solid #1A73E8;">
                <div style="font-size:15px; font-weight:700; color:#1A73E8; flex:1;">&#11088; Lasso Regression</div>
                <div style="font-size:13px; font-weight:600; color:#1A73E8;">Best Model</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px; padding:14px 18px; background:#F0F7FF; border-radius:10px; border-left: 3px solid #6366F1;">
                <div style="font-size:15px; font-weight:600; color:#0F172A; flex:1;">Random Forest</div>
                <div style="font-size:13px; color:#475569;">Ensemble</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px; padding:14px 18px; background:#F0F7FF; border-radius:10px; border-left: 3px solid #6366F1;">
                <div style="font-size:15px; font-weight:600; color:#0F172A; flex:1;">Gradient Boosting</div>
                <div style="font-size:13px; color:#475569;">Ensemble</div>
            </div>
        </div>
    </div>

    <div class="card" style="border-top: 4px solid #00C9A7; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:20px;">Strategy</div>
        <div style="display:flex; flex-direction:column; gap:20px; flex:1;">
            <div>
                <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:8px;">Train / Test Split</div>
                <p style="font-size:14px; color:#475569; line-height:1.6;">80/20 stratified split preserving CLV distribution. All features standardized via StandardScaler.</p>
                <div style="margin-top:10px; height:8px; border-radius:4px; background:#E2E8F0; overflow:hidden;">
                    <div style="width:80%; height:100%; background:linear-gradient(90deg, #1A73E8, #3B82F6); border-radius:4px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:4px;">
                    <span style="font-size:12px; color:#1A73E8; font-weight:600;">80% Train</span>
                    <span style="font-size:12px; color:#475569; font-weight:600;">20% Test</span>
                </div>
            </div>
            <div>
                <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:8px;">Cross-Validation</div>
                <p style="font-size:14px; color:#475569; line-height:1.6;">5-fold CV on training set. Best CV R&sup2; = 0.53 &plusmn; 0.18 (Lasso).</p>
            </div>
            <div>
                <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:8px;">Hyperparameter Tuning</div>
                <p style="font-size:14px; color:#475569; line-height:1.6;">Randomized search over alpha values. Best Lasso &alpha; = 4.90.</p>
            </div>
        </div>
    </div>
</div>
""")


# ── Slide 7: Results ────────────────────────────────────────────────────
def slide_07_results():
    mc = img_b64(FIGURES / "model_comparison.png")
    ba = img_b64(FIGURES / "best_model_analysis.png")
    return wrap_slide(f"""
<div class="section-tag">Results</div>
<h2 class="slide-title">Model Performance</h2>
<p class="slide-subtitle">Lasso Regression achieves the best balance of fit and generalization</p>

<div class="grid-4" style="margin-bottom:24px;">
    <div class="metric-card">
        <div class="metric-value" style="color:#1A73E8;">0.810</div>
        <div class="metric-label">Test R&sup2;</div>
    </div>
    <div class="metric-card" style="border-top-color:#00C9A7;">
        <div class="metric-value" style="color:#00C9A7;">&pound;1,756</div>
        <div class="metric-label">RMSE</div>
    </div>
    <div class="metric-card" style="border-top-color:#F59E0B;">
        <div class="metric-value" style="color:#F59E0B;">&pound;545</div>
        <div class="metric-label">MAE</div>
    </div>
    <div class="metric-card" style="border-top-color:#6366F1;">
        <div class="metric-value" style="color:#6366F1;">0.53</div>
        <div class="metric-label">CV R&sup2; (&plusmn; 0.18)</div>
    </div>
</div>

<div class="grid-2" style="flex:1;">
    <div class="card" style="padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Model Comparison</div>
        <img src="{mc}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>
    <div class="card" style="padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Best Model Analysis</div>
        <img src="{ba}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>
</div>
""")


# ── Slide 8: Explainability ─────────────────────────────────────────────
def slide_08_explainability():
    fi = img_b64(FIGURES / "feature_importance.png")
    ss = img_b64(FIGURES / "shap_summary.png")
    return wrap_slide(f"""
<div class="section-tag">Explainability</div>
<h2 class="slide-title">Feature Importance &amp; SHAP Analysis</h2>
<p class="slide-subtitle">Understanding what drives CLV predictions</p>

<div class="grid-2" style="flex:1; margin-bottom:24px;">
    <div class="card" style="padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Feature Importance (All Models)</div>
        <img src="{fi}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>
    <div class="card" style="padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">SHAP Summary Plot</div>
        <img src="{ss}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>
</div>

<div class="banner" style="background:linear-gradient(135deg, #0A1628, #1E3A5F);">
    <div style="width:44px; height:44px; border-radius:12px; background:rgba(26,115,232,0.2); display:flex; align-items:center; justify-content:center;">
        <span style="color:#60A5FA; font-size:20px;">&#x1F4A1;</span>
    </div>
    <div style="flex:1;">
        <div style="font-size:16px; font-weight:700;">Key Insight</div>
        <div style="font-size:14px; color:rgba(248,250,252,0.6);">Monetary dominates with 3&times; the importance of Frequency. SHAP confirms a positive, near-linear relationship: higher past spending directly predicts higher future CLV.</div>
    </div>
</div>
""")


# ── Slide 9: Lift ───────────────────────────────────────────────────────
def slide_09_lift():
    lc = img_b64(FIGURES / "lift_chart.png")
    return wrap_slide(f"""
<div class="section-tag">Lift Analysis</div>
<h2 class="slide-title">Lift Chart &amp; Business Impact</h2>
<p class="slide-subtitle">Quantifying the model's ability to identify high-value customers</p>

<div style="display:flex; gap:24px; flex:1;">
    <div class="card" style="flex:1.2; padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Cumulative Lift Chart</div>
        <img src="{lc}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>

    <div style="flex:0.8; display:flex; flex-direction:column; gap:16px;">
        <div class="card" style="border-left: 4px solid #1A73E8; flex:1; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:36px; font-weight:800; color:#1A73E8;">5.55&times;</div>
            <div style="font-size:13px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Lift @ Top 10%</div>
            <p style="font-size:13px; color:#94A3B8; margin-top:8px;">Top decile captures 5.5&times; more value than random</p>
        </div>
        <div class="card" style="border-left: 4px solid #00C9A7; flex:1; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:36px; font-weight:800; color:#00C9A7;">3.62&times;</div>
            <div style="font-size:13px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Lift @ Top 20%</div>
            <p style="font-size:13px; color:#94A3B8; margin-top:8px;">Top quintile still highly efficient for targeting</p>
        </div>
        <div class="card" style="border-left: 4px solid #F59E0B; flex:1; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:36px; font-weight:800; color:#F59E0B;">2.76&times;</div>
            <div style="font-size:13px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Lift @ Top 30%</div>
            <p style="font-size:13px; color:#94A3B8; margin-top:8px;">Nearly 3&times; improvement over untargeted outreach</p>
        </div>
        <div class="card" style="border-left: 4px solid #6366F1; flex:1; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:36px; font-weight:800; color:#6366F1;">1.83&times;</div>
            <div style="font-size:13px; font-weight:600; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-top:4px;">Lift @ Top 50%</div>
            <p style="font-size:13px; color:#94A3B8; margin-top:8px;">Model adds value even at broader targeting</p>
        </div>
    </div>
</div>
""")


# ── Slide 10: Divider ───────────────────────────────────────────────────
def slide_10_divider():
    return wrap_slide("""
<div style="display:flex; flex-direction:column; height:100%; justify-content:center; align-items:center; text-align:center;">
    <div class="section-tag section-tag-dark" style="margin-bottom:32px;">Part II</div>
    <h1 style="font-size:72px; font-weight:900; line-height:1.05;
        background: linear-gradient(135deg, #FFFFFF 0%, #00C9A7 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Causal Inference
    </h1>
    <p style="font-size:22px; font-weight:300; color:rgba(248,250,252,0.45); margin-top:24px; max-width:600px;">
        Moving beyond correlation to estimate the causal effect of purchase frequency on customer lifetime value
    </p>
    <div style="margin-top:48px; display:flex; gap:24px;">
        <div style="padding:12px 24px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:10px;">
            <div style="font-size:14px; color:rgba(248,250,252,0.4); font-weight:500;">Treatment</div>
            <div style="font-size:18px; font-weight:700; color:#00C9A7;">High Frequency</div>
        </div>
        <div style="padding:12px 24px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:10px;">
            <div style="font-size:14px; color:rgba(248,250,252,0.4); font-weight:500;">Outcome</div>
            <div style="font-size:18px; font-weight:700; color:#60A5FA;">Future CLV</div>
        </div>
        <div style="padding:12px 24px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:10px;">
            <div style="font-size:14px; color:rgba(248,250,252,0.4); font-weight:500;">Method</div>
            <div style="font-size:18px; font-weight:700; color:#F59E0B;">Meta-Learners</div>
        </div>
    </div>
</div>
""", dark=True)


# ── Slide 11: Causal Setup ──────────────────────────────────────────────
def slide_11_causal_setup():
    return wrap_slide("""
<div class="section-tag">Causal Design</div>
<h2 class="slide-title">Causal Inference Setup</h2>
<p class="slide-subtitle">Estimating the Average Treatment Effect of high purchase frequency on CLV</p>

<div class="grid-3" style="margin-bottom:24px;">
    <div class="card" style="border-top: 4px solid #F59E0B;">
        <div style="font-size:13px; font-weight:700; color:#F59E0B; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Treatment</div>
        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">High Frequency</div>
        <p style="font-size:14px; color:#475569; line-height:1.7;">
            Binary indicator: customer is above the median purchase frequency. Simulates a &ldquo;loyalty program&rdquo; or &ldquo;re-engagement&rdquo; intervention.
        </p>
    </div>
    <div class="card" style="border-top: 4px solid #1A73E8;">
        <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Outcome</div>
        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">log(1 + FutureCLV)</div>
        <p style="font-size:14px; color:#475569; line-height:1.7;">
            Log-transformed future CLV to handle right-skewed distribution and make treatment effects interpretable on a relative scale.
        </p>
    </div>
    <div class="card" style="border-top: 4px solid #00C9A7;">
        <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">Controls</div>
        <div style="font-size:20px; font-weight:700; margin-bottom:8px;">Confounders</div>
        <p style="font-size:14px; color:#475569; line-height:1.7;">
            Recency, Monetary, Tenure, AvgOrderValue, AvgBasketSize, NumUniqueProducts, AvgTimeBetweenPurchases.
        </p>
    </div>
</div>

<div class="grid-2">
    <div class="card" style="padding:24px; display:flex; align-items:center; gap:20px;">
        <div style="width:56px; height:56px; border-radius:14px; background:linear-gradient(135deg, #EEF2FF, #DBEAFE); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            <span style="font-size:24px; font-weight:800; color:#1A73E8;">S</span>
        </div>
        <div>
            <div style="font-size:16px; font-weight:700; color:#0F172A;">S-Learner (LRS Regressor)</div>
            <div style="font-size:14px; color:#475569; margin-top:4px;">Single model with treatment as feature. Estimates ATE via counterfactual prediction.</div>
        </div>
    </div>
    <div class="card" style="padding:24px; display:flex; align-items:center; gap:20px;">
        <div style="width:56px; height:56px; border-radius:14px; background:linear-gradient(135deg, #ECFDF5, #D1FAE5); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            <span style="font-size:24px; font-weight:800; color:#00C9A7;">T</span>
        </div>
        <div>
            <div style="font-size:16px; font-weight:700; color:#0F172A;">T-Learner (XGB Regressor)</div>
            <div style="font-size:14px; color:#475569; margin-top:4px;">Separate models for treated/control groups. More flexible, captures heterogeneous effects.</div>
        </div>
    </div>
</div>
""")


# ── Slide 12: Causal Results ────────────────────────────────────────────
def slide_12_causal_results():
    cd = img_b64(REPORTS / "causal_cate_distribution.png")
    cf = img_b64(REPORTS / "causal_feature_importance.png")
    return wrap_slide(f"""
<div class="section-tag">Causal Results</div>
<h2 class="slide-title">Causal Inference Results</h2>
<p class="slide-subtitle">Both meta-learners estimate a negative average treatment effect</p>

<div style="display:flex; gap:24px; margin-bottom:24px;">
    <div class="metric-card" style="flex:1; border-top-color:#1A73E8;">
        <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">S-Learner (LRS)</div>
        <div class="metric-value" style="color:#1A73E8;">&minus;0.179</div>
        <div style="font-size:13px; color:#94A3B8; margin-top:8px;">CI: [&minus;0.378, +0.021]</div>
    </div>
    <div class="metric-card" style="flex:1; border-top-color:#00C9A7;">
        <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">T-Learner (XGB)</div>
        <div class="metric-value" style="color:#00C9A7;">&minus;0.167</div>
        <div style="font-size:13px; color:#94A3B8; margin-top:8px;">CI: [&minus;0.258, &minus;0.077]</div>
    </div>
    <div class="card" style="flex:2; padding:24px; background:linear-gradient(135deg, #FFFBEB, #FEF3C7); border: 1px solid #FDE68A;">
        <div style="font-size:13px; font-weight:700; color:#92400E; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Interpretation</div>
        <p style="font-size:14px; color:#78350F; line-height:1.7;">
            High purchase frequency does <strong>not</strong> causally increase CLV. The negative ATE (&sim;&minus;0.17 on log scale) suggests that after controlling for confounders,
            high-frequency buyers may be <strong>deal-seekers</strong> with lower per-transaction value. Correlation &ne; causation.
        </p>
    </div>
</div>

<div class="grid-2" style="flex:1;">
    <div class="card" style="padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">CATE Distribution</div>
        <img src="{cd}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>
    <div class="card" style="padding:16px; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Causal Feature Importance</div>
        <img src="{cf}" class="chart-img" style="flex:1; object-fit:contain;">
    </div>
</div>
""")


# ── Slide 13: Threats ───────────────────────────────────────────────────
def slide_13_threats():
    return wrap_slide("""
<div class="section-tag">Validity</div>
<h2 class="slide-title">Threats to Validity</h2>
<p class="slide-subtitle">Acknowledging limitations and potential biases in our analysis</p>

<div style="display:flex; flex-direction:column; gap:16px; flex:1;">
    <div class="card card-accent-left" style="display:flex; align-items:flex-start; gap:20px; flex:1; border-left-color:#EF4444;">
        <div style="width:44px; height:44px; border-radius:12px; background:rgba(239,68,68,0.08); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            <span style="font-size:20px; font-weight:800; color:#EF4444;">1</span>
        </div>
        <div>
            <div style="font-size:17px; font-weight:700; color:#0F172A; margin-bottom:6px;">Selection Bias in Treatment Assignment</div>
            <p style="font-size:14px; color:#475569; line-height:1.6;">Frequency is observed, not randomized. The median split may conflate inherently loyal customers with those influenced by external factors. Propensity score matching could help, but unobserved confounders remain.</p>
        </div>
    </div>
    <div class="card card-accent-left" style="display:flex; align-items:flex-start; gap:20px; flex:1; border-left-color:#F59E0B;">
        <div style="width:44px; height:44px; border-radius:12px; background:rgba(245,158,11,0.08); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            <span style="font-size:20px; font-weight:800; color:#F59E0B;">2</span>
        </div>
        <div>
            <div style="font-size:17px; font-weight:700; color:#0F172A; margin-bottom:6px;">CV vs Test Gap (0.53 &rarr; 0.81)</div>
            <p style="font-size:14px; color:#475569; line-height:1.6;">The gap between cross-validation R&sup2; (0.53) and test R&sup2; (0.81) may indicate favorable test split or temporal patterns. Additional temporal CV would strengthen confidence.</p>
        </div>
    </div>
    <div class="card card-accent-left" style="display:flex; align-items:flex-start; gap:20px; flex:1; border-left-color:#1A73E8;">
        <div style="width:44px; height:44px; border-radius:12px; background:rgba(26,115,232,0.08); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            <span style="font-size:20px; font-weight:800; color:#1A73E8;">3</span>
        </div>
        <div>
            <div style="font-size:17px; font-weight:700; color:#0F172A; margin-bottom:6px;">Single-Geography, Single-Period</div>
            <p style="font-size:14px; color:#475569; line-height:1.6;">Data comes from one UK retailer in 2010&ndash;2011. Generalization to other markets, product categories, or time periods is not guaranteed.</p>
        </div>
    </div>
    <div class="card card-accent-left" style="display:flex; align-items:flex-start; gap:20px; flex:1; border-left-color:#6366F1;">
        <div style="width:44px; height:44px; border-radius:12px; background:rgba(99,102,241,0.08); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            <span style="font-size:20px; font-weight:800; color:#6366F1;">4</span>
        </div>
        <div>
            <div style="font-size:17px; font-weight:700; color:#0F172A; margin-bottom:6px;">Limited Feature Set</div>
            <p style="font-size:14px; color:#475569; line-height:1.6;">8 RFM-based features capture purchase behavior but miss demographics, marketing touchpoints, seasonality, and product affinity &mdash; factors that likely influence CLV.</p>
        </div>
    </div>
</div>
""")


# ── Slide 14: Monitoring ────────────────────────────────────────────────
def slide_14_monitoring():
    return wrap_slide("""
<div class="section-tag">Production</div>
<h2 class="slide-title">Monitoring &amp; Deployment Plan</h2>
<p class="slide-subtitle">Ensuring model reliability in production through drift detection and scheduled retraining</p>

<div class="grid-3" style="flex:1; margin-bottom:24px;">
    <div class="card card-accent-top" style="display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#1A73E8; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Production Checks</div>
        <div style="display:flex; flex-direction:column; gap:12px; flex:1;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#00C9A7;"></div>
                <span style="font-size:14px; color:#475569;">Input schema validation</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#00C9A7;"></div>
                <span style="font-size:14px; color:#475569;">Feature range checks</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#00C9A7;"></div>
                <span style="font-size:14px; color:#475569;">Prediction distribution monitoring</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#00C9A7;"></div>
                <span style="font-size:14px; color:#475569;">Latency &amp; throughput SLAs</span>
            </div>
        </div>
    </div>
    <div class="card" style="border-top: 4px solid #F59E0B; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#F59E0B; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Alert Thresholds</div>
        <div style="display:flex; flex-direction:column; gap:12px; flex:1;">
            <div style="padding:12px 16px; background:#FFFBEB; border-radius:8px;">
                <div style="font-size:13px; font-weight:600; color:#92400E;">PSI &gt; 0.1</div>
                <div style="font-size:12px; color:#A16207;">Moderate drift &mdash; investigate</div>
            </div>
            <div style="padding:12px 16px; background:#FEF2F2; border-radius:8px;">
                <div style="font-size:13px; font-weight:600; color:#991B1B;">PSI &gt; 0.25</div>
                <div style="font-size:12px; color:#B91C1C;">Severe drift &mdash; trigger retrain</div>
            </div>
            <div style="padding:12px 16px; background:#FEF2F2; border-radius:8px;">
                <div style="font-size:13px; font-weight:600; color:#991B1B;">R&sup2; drop &gt; 15%</div>
                <div style="font-size:12px; color:#B91C1C;">Performance degradation &mdash; retrain</div>
            </div>
        </div>
    </div>
    <div class="card" style="border-top: 4px solid #00C9A7; display:flex; flex-direction:column;">
        <div style="font-size:13px; font-weight:700; color:#00C9A7; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:16px;">Maintenance</div>
        <div style="display:flex; flex-direction:column; gap:12px; flex:1;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#1A73E8;"></div>
                <span style="font-size:14px; color:#475569;">Quarterly scheduled retraining</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#1A73E8;"></div>
                <span style="font-size:14px; color:#475569;">Automated drift-triggered retrain</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#1A73E8;"></div>
                <span style="font-size:14px; color:#475569;">Champion/challenger testing</span>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#1A73E8;"></div>
                <span style="font-size:14px; color:#475569;">Model versioning &amp; rollback</span>
            </div>
        </div>
    </div>
</div>

<div class="banner" style="background:linear-gradient(135deg, #064E3B, #065F46);">
    <div style="width:10px; height:10px; border-radius:50%; background:#00C9A7; box-shadow: 0 0 8px #00C9A7;"></div>
    <div style="flex:1;">
        <div style="font-size:16px; font-weight:700;">All Features Stable &mdash; Max PSI = 0.035</div>
        <div style="font-size:14px; color:rgba(248,250,252,0.6);">Simulated drift test shows no feature exceeds the 0.1 warning threshold. Model is production-ready.</div>
    </div>
</div>
""")


# ── Slide 15: Conclusions ───────────────────────────────────────────────
def slide_15_conclusions():
    return wrap_slide("""
<div style="display:flex; flex-direction:column; height:100%; justify-content:center; align-items:center; text-align:center;">
    <div class="section-tag section-tag-dark" style="margin-bottom:24px;">Conclusions</div>
    <h1 style="font-size:56px; font-weight:900; line-height:1.1; margin-bottom:48px; max-width:1100px;
        background: linear-gradient(135deg, #FFFFFF 0%, #94C4FF 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Key Takeaways
    </h1>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px 48px; max-width:1200px; text-align:left; margin-bottom:48px;">
        <div style="display:flex; align-items:flex-start; gap:16px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#1A73E8; margin-top:6px; flex-shrink:0;"></div>
            <div>
                <div style="font-size:18px; font-weight:700; margin-bottom:4px;">Lasso achieves R&sup2; = 0.81</div>
                <div style="font-size:14px; color:rgba(248,250,252,0.5); line-height:1.5;">Regularized regression outperforms tree-based models on this dataset, with strong predictive power and interpretability.</div>
            </div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:16px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#00C9A7; margin-top:6px; flex-shrink:0;"></div>
            <div>
                <div style="font-size:18px; font-weight:700; margin-bottom:4px;">5.55&times; lift in top decile</div>
                <div style="font-size:14px; color:rgba(248,250,252,0.5); line-height:1.5;">Model-driven targeting captures 5&times; more value than random selection &mdash; direct ROI for marketing spend.</div>
            </div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:16px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#F59E0B; margin-top:6px; flex-shrink:0;"></div>
            <div>
                <div style="font-size:18px; font-weight:700; margin-bottom:4px;">Frequency &ne; causal driver</div>
                <div style="font-size:14px; color:rgba(248,250,252,0.5); line-height:1.5;">Causal analysis reveals high frequency does not increase CLV &mdash; challenging the &ldquo;more visits = more value&rdquo; assumption.</div>
            </div>
        </div>
        <div style="display:flex; align-items:flex-start; gap:16px;">
            <div style="width:12px; height:12px; border-radius:50%; background:#6366F1; margin-top:6px; flex-shrink:0;"></div>
            <div>
                <div style="font-size:18px; font-weight:700; margin-bottom:4px;">Production-ready pipeline</div>
                <div style="font-size:14px; color:rgba(248,250,252,0.5); line-height:1.5;">Full monitoring with PSI drift detection (all stable at &lt; 0.035), automated alerts, and quarterly retraining schedule.</div>
            </div>
        </div>
    </div>

    <div style="padding:20px 40px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:14px; margin-bottom:32px;">
        <div style="font-size:13px; font-weight:600; color:rgba(248,250,252,0.35); text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">Next Steps</div>
        <div style="font-size:16px; color:rgba(248,250,252,0.7);">
            Add temporal cross-validation &bull; Integrate product-level features &bull; A/B test targeting strategies &bull; Expand to multi-market
        </div>
    </div>

    <div style="font-size:48px; font-weight:900; margin-bottom:8px;
        background: linear-gradient(135deg, #1A73E8, #00C9A7); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Thank You
    </div>
    <p style="font-size:16px; color:rgba(248,250,252,0.35);">Questions &amp; Discussion</p>
</div>
""", dark=True)


# ── All slides ──────────────────────────────────────────────────────────
ALL_SLIDES = [
    slide_01_title,
    slide_02_context,
    slide_03_hypotheses,
    slide_04_data,
    slide_05_features,
    slide_06_modeling,
    slide_07_results,
    slide_08_explainability,
    slide_09_lift,
    slide_10_divider,
    slide_11_causal_setup,
    slide_12_causal_results,
    slide_13_threats,
    slide_14_monitoring,
    slide_15_conclusions,
]


# ── Screenshot via Playwright ───────────────────────────────────────────
async def capture_slides():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        for i, slide_fn in enumerate(ALL_SLIDES, 1):
            html = slide_fn()
            await page.set_content(html)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(800)

            path = PREVIEWS / f"slide_{i:02d}.png"
            await page.screenshot(path=str(path), type="png")
            print(f"  [ok] Slide {i:02d} captured")

        await browser.close()


# ── Assemble PPTX ──────────────────────────────────────────────────────
def assemble_pptx():
    from pptx import Presentation
    from pptx.util import Inches, Emu

    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9
    prs.slide_height = Inches(7.5)

    for i in range(1, 16):
        img_path = PREVIEWS / f"slide_{i:02d}.png"
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
        slide.shapes.add_picture(
            str(img_path), Emu(0), Emu(0),
            prs.slide_width, prs.slide_height,
        )

    out = DELIVERABLES / "final_presentation.pptx"
    prs.save(str(out))
    print(f"  [ok] PPTX saved: {out}")


# ── Assemble PDF ────────────────────────────────────────────────────────
def assemble_pdf():
    from PIL import Image

    images = []
    for i in range(1, 16):
        img_path = PREVIEWS / f"slide_{i:02d}.png"
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    out = DELIVERABLES / "final_presentation.pdf"
    images[0].save(str(out), save_all=True, append_images=images[1:])
    print(f"  [ok] PDF saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  CLV Presentation Generator")
    print("=" * 60)

    print("\nCapturing slides with Playwright...\n")
    await capture_slides()

    print("\nAssembling PPTX...\n")
    assemble_pptx()

    print("\nAssembling PDF...\n")
    assemble_pdf()

    print("\n" + "=" * 60)
    print("  Done! Files saved to deliverables/")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
