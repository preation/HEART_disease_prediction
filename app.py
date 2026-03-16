"""
Cardio Risk Analyzer
Author  : preation07
Purpose : Educational / Research use only — NOT medical advice
"""

import base64
import io
import pandas as pd
import streamlit.components.v1 as components
import joblib
import streamlit as st
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:          #07111f;
    --bg2:         #0b1a2e;
    --card:        rgba(255,255,255,0.045);
    --card-hover:  rgba(255,255,255,0.07);
    --border:      rgba(255,255,255,0.09);
    --border-hi:   rgba(255,255,255,0.16);
    --red:         #e8445a;
    --red-soft:    rgba(232,68,90,0.18);
    --teal:        #2ecfbd;
    --teal-soft:   rgba(46,207,189,0.15);
    --navy:        #1a3a5c;
    --gold:        #f5a623;
    --green:       #3dd68c;
    --hi:          rgba(255,255,255,0.92);
    --mid:         rgba(255,255,255,0.50);
    --lo:          rgba(255,255,255,0.28);
    --r:           14px;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    color: var(--hi) !important;
}

/* ── Medical deep-navy background ── */
.stApp {
    background:
        radial-gradient(ellipse 100% 60% at 0%   0%,   rgba(14,40,80,0.85)   0%, transparent 55%),
        radial-gradient(ellipse 80%  50% at 100% 100%,  rgba(10,60,70,0.70)   0%, transparent 50%),
        radial-gradient(ellipse 60%  80% at 50%  50%,   rgba(7,17,31,1.00)    0%, transparent 100%),
        #07111f !important;
    min-height: 100vh;
}

/* ── Hide sidebar toggle ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Page wrapper ── */
.block-container {
    max-width: 1080px !important;
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Section label ── */
.sec-lbl {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal) !important;
    margin: 0 0 1rem;
    padding-bottom: 7px;
    border-bottom: 1px solid rgba(46,207,189,0.22);
    display: block;
}

/* ── Input cards ── */
.input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.5rem 1.6rem 1.2rem;
    margin-bottom: 1.1rem;
    transition: border-color 0.2s ease;
}
.input-card:hover { border-color: var(--border-hi); }

/* ── Hint text ── */
.hint {
    font-size: 11.5px;
    color: var(--lo) !important;
    font-style: italic;
    margin: -6px 0 10px;
    line-height: 1.55;
}

/* ── Analyze button ── */
.stButton > button {
    background: linear-gradient(135deg, #e8445a 0%, #b5203a 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.78rem 1.4rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    letter-spacing: 0.04em;
    box-shadow: 0 5px 26px rgba(232,68,90,0.40) !important;
    transition: all 0.22s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(232,68,90,0.62) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Radio ── */
.stRadio [role="radiogroup"] { gap: 7px !important; }
.stRadio [role="radio"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    padding: 4px 14px !important;
    font-size: 13px !important;
    transition: all 0.18s ease !important;
}
.stRadio [role="radio"][aria-checked="true"] {
    background: var(--red-soft) !important;
    border-color: rgba(232,68,90,0.55) !important;
    color: #ff8fa3 !important;
}

/* ── Slider ── */
[data-baseweb="slider"] [role="slider"] {
    background: var(--red) !important;
    border-color: var(--red) !important;
    box-shadow: 0 0 0 4px rgba(232,68,90,0.25) !important;
}
[data-baseweb="slider"] [data-testid="stSliderTrackActive"] {
    background: var(--red) !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 99px !important;
    height: 10px !important;
}
[data-testid="stProgress"] > div > div {
    border-radius: 99px !important;
    transition: width 1.4s cubic-bezier(.22,.68,0,1.1) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* ── Result section: fade + slide-up ── */
.res-wrap {
    animation: fadeUp 0.6s cubic-bezier(.22,.68,0,1.15) both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(32px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.res-delay-1 { animation-delay: 0.05s; }
.res-delay-2 { animation-delay: 0.20s; }
.res-delay-3 { animation-delay: 0.38s; }

/* ── Score card ── */
.score-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 2.2rem 1.5rem;
    text-align: center;
    margin-bottom: 1.4rem;
}
.score-num {
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
    font-family: 'Outfit', sans-serif;
}
.score-badge {
    display: inline-block;
    padding: 6px 22px;
    border-radius: 99px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 14px;
}
.clr-high { color: #ff4d6d; }
.clr-mid  { color: #fbbf24; }
.clr-low  { color: #3dd68c; }
.bg-high  { background:rgba(255,77,109,0.15); color:#ff8fa3; border:1px solid rgba(255,77,109,0.30); }
.bg-mid   { background:rgba(251,191,36,0.15); color:#fcd34d; border:1px solid rgba(251,191,36,0.30); }
.bg-low   { background:rgba(61,214,140,0.15); color:#86efac; border:1px solid rgba(61,214,140,0.30); }

/* ── Result panel cards (side by side) ── */
.panel {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(74,168,255,0.18);
    border-radius: var(--r);
    padding: 2.2rem 2rem;
    height: 100%;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* Panel top accent line */
.panel-graph { border-top: 3px solid rgba(232,68,90,0.70); }
.panel-ai    { border-top: 3px solid rgba(74,168,255,0.70); }

.panel-head {
    font-size: 13.5px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: rgba(180,220,255,0.75) !important;
    margin-bottom: 1.2rem;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(74,168,255,0.15);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Chart legend row */
.chart-legend {
    font-size: 13.5px;
    color: rgba(255,255,255,0.45) !important;
    margin: -4px 0 18px;
    display: flex;
    gap: 18px;
    align-items: center;
}
.leg-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    vertical-align: middle;
}

/* ── AI Summary text ── */
.ai-block {
    font-size: 17.5px;
    line-height: 2.1;
    color: rgba(230,242,255,0.88) !important;
    font-weight: 400;
    letter-spacing: 0.005em;
}
.ai-block p {
    margin-bottom: 1.3rem;
    padding-bottom: 1.3rem;
    border-bottom: 1px solid rgba(74,168,255,0.08);
}
.ai-block p:last-child {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
}

/* ── Section header above results ── */
.results-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(74,168,255,0.70) !important;
    margin: 0 0 1.4rem;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(74,168,255,0.15);
    display: block;
}

/* ── Empty state ── */
.empty {
    text-align: center;
    padding: 5.5rem 2rem;
    border: 1px dashed rgba(74,168,255,0.14);
    border-radius: var(--r);
    background: rgba(255,255,255,0.025);
}
.empty-icon {
    font-size: 3.8rem;
    display: block;
    margin-bottom: 1rem;
    animation: beat 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 14px rgba(74,168,255,0.40));
}
@keyframes beat { 0%,100%{transform:scale(1)} 50%{transform:scale(1.09)} }

/* ── Disclaimer ── */
.disc {
    margin-top: 3.5rem;
    padding: 1.4rem 1.2rem;
    border-top: 1px solid rgba(74,168,255,0.12);
    text-align: center;
    font-size: 13px;
    color: rgba(180,210,255,0.40) !important;
    line-height: 2.1;
}
.disc strong { color: rgba(180,210,255,0.65) !important; }
.disc a { color: var(--teal) !important; text-decoration: none; font-weight: 700; font-size: 13.5px; }
.disc a:hover { text-decoration: underline; color: #7dd8f8 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--red) !important; }

h1,h2,h3,h4,h5 { font-family: 'Outfit', sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CHEST_PAIN_MAP = {
    "Typical Angina":    0,
    "Atypical Angina":   1,
    "Non-Anginal Pain":  2,
    "Asymptomatic":      3,
}
THAL_MAP = {
    "Normal":            1,
    "Fixed Defect":      2,
    "Reversible Defect": 3,
}
FRIENDLY = {
    "age":      "Age",
    "sex":      "Biological Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Blood Pressure",
    "chol":     "Cholesterol",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina",
    "oldpeak":  "ST Depression",
    "ca":       "Major Vessels",
    "thal":     "Thalassemia",
}
USER_FEATURES = list(FRIENDLY.keys())
ALL_FEATURES  = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak","slope","ca","thal",
]


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("xgb_model.pkl")

@st.cache_resource(show_spinner=False)
def load_explainer(_mdl):
    # Leading underscore → skip hashing (XGBClassifier is not hashable by Streamlit)
    return shap.TreeExplainer(_mdl)

try:
    model     = load_model()
    explainer = load_explainer(model)
except Exception as e:
    st.error(f"⚠️ Model could not be loaded: {e}")
    st.stop()

groq_api_key: str = st.secrets.get("GROQ_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hint(txt: str):
    st.markdown(f'<p class="hint">{txt}</p>', unsafe_allow_html=True)


def shap_bar_chart(shap_vals: np.ndarray, feature_names: list,
                   base_val: float) -> plt.Figure:
    """Horizontal SHAP importance chart — all colors as float RGBA tuples."""
    idx  = np.argsort(shap_vals)
    sv   = shap_vals[idx]
    fns  = [feature_names[i] for i in idx]

    RED_BAR   = (0.910, 0.267, 0.353, 0.88)
    TEAL_BAR  = (0.180, 0.812, 0.741, 0.88)
    LINE_CLR  = (1.0,   1.0,   1.0,   0.12)
    AXIS_CLR  = (1.0,   1.0,   1.0,   0.38)
    TICK_CLR  = (1.0,   1.0,   1.0,   0.52)
    LABEL_CLR = (1.0,   1.0,   1.0,   0.36)
    BG_CLR    = (0.027, 0.067, 0.118, 1.00)  # matches --bg2

    colors = [TEAL_BAR if v <= 0 else RED_BAR for v in sv]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig.patch.set_facecolor("none")
    ax.set_facecolor(BG_CLR)

    ax.barh(fns, sv, color=colors, height=0.60, edgecolor="none")
    ax.axvline(0, color=LINE_CLR, linewidth=0.85, linestyle="--")

    for i, val in enumerate(sv):
        x  = val + 0.004 if val >= 0 else val - 0.004
        ha = "left"      if val >= 0 else "right"
        ax.text(x, i, f"{val:+.3f}", va="center", ha=ha,
                fontsize=9.5, color=LABEL_CLR, fontfamily="monospace")

    ax.set_xlabel("SHAP value  ·  impact on risk score",
                  color=AXIS_CLR, fontsize=11, labelpad=10)
    ax.tick_params(axis="both", colors=TICK_CLR, labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor((1, 1, 1, 0.06))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"Population avg risk: {base_val:.1%}",
                 color=LABEL_CLR, fontsize=10, pad=10,
                 loc="right", fontstyle="italic")
    fig.tight_layout()
    return fig


def groq_summary(client: Groq, risk_pct: float, risk_label: str,
                 age: int, sex: str, cp: str,
                 trestbps: int, chol: int, thalach: int,
                 exang: str, oldpeak: float, ca: int, thal: str) -> str:
    prompt = f""" You are a compassionate medical AI assistant having cardio specialization.
An XGBoost model predicted a heart disease risk of {risk_pct}% ({risk_label}).

Patient profile — Age: {age}, Sex: {sex}, Chest pain: {cp},
BP: {trestbps} mmHg, Cholesterol: {chol} mg/dL, Max HR: {thalach} bpm,
Exercise angina: {exang}, ST depression: {oldpeak},
Major vessels: {ca}, Thalassemia: {thal}.

Write exactly 4 short paragraphs (NO headings, NO bullet points, NO markdown):
1. Explain the score warmly and simply.
2. Highlight 2–3 concerning indicators.
3. Suggest 2–3 actionable lifestyle improvements.
4. Strongly advise consulting a cardiologist. State this is NOT medical advice.

Keep under 100 words.dont mention any model or ai name behaviour just give in Friendly, clear, zero  medical jargon.in less words"""

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.35,
        max_tokens=480,
    )
    return resp.choices[0].message.content.strip()


def build_print_report(
    risk_pct, risk_label,
    age, sex, cp,
    trestbps, chol, thalach,
    exang, oldpeak, ca, thal,
    summary_text,
    chart_fig,
):
    """
    Build a self-contained HTML report with NO external dependencies.
    System fonts only — so window.print() fires instantly on load
    without waiting for Google Fonts, eliminating the blank-page issue.
    """
    import base64, io
    from datetime import datetime

    # Render chart to PNG in memory (white bg for print)
    buf = io.BytesIO()
    chart_fig.savefig(buf, format="png", dpi=150,
                      facecolor="#f4f7fb", bbox_inches="tight")
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode()

    paras     = [p.strip() for p in summary_text.split("\n\n") if p.strip()]
    para_html = "".join(f"<p>{p}</p>" for p in paras)

    risk_color = (
        "#c0392b" if risk_pct >= 65 else
        "#d97706" if risk_pct >= 35 else
        "#16a34a"
    )
    report_date = datetime.now().strftime("%B %d, %Y  %I:%M %p")

    # NOTE: No @import / external fonts — system font stack only.
    # This means the page is 100% self-contained and prints instantly.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cardio Risk Report — {report_date}</title>
<style>
  /* System font stack — no external dependencies, prints instantly */
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{
    font-family:-apple-system,'Segoe UI',system-ui,Arial,sans-serif;
    background:#f4f7fb;color:#1a2035;padding:32px 40px;font-size:14px;
  }}

  /* ── Header ── */
  .rpt-header{{
    display:flex;align-items:center;justify-content:space-between;
    border-bottom:2px solid #d0dff0;padding-bottom:18px;margin-bottom:28px;
  }}
  .rpt-title{{font-size:24px;font-weight:800;color:#0f2a4a;letter-spacing:-0.02em}}
  .rpt-title span{{color:#2563eb}}
  .rpt-meta{{font-size:12px;color:#64748b;text-align:right;line-height:1.8}}

  /* ── Score block ── */
  .score-block{{
    background:#fff;border:1px solid #d0dff0;
    border-left:5px solid {risk_color};border-radius:12px;
    padding:22px 28px;display:flex;align-items:center;gap:28px;
    margin-bottom:26px;box-shadow:0 2px 10px rgba(0,0,0,.06);
  }}
  .score-big{{font-size:52px;font-weight:800;color:{risk_color};line-height:1}}
  .score-label{{font-size:13px;font-weight:700;text-transform:uppercase;
    letter-spacing:.1em;color:{risk_color};margin-bottom:6px}}
  .score-sub{{font-size:13px;color:#64748b;line-height:1.6}}

  /* ── Section title ── */
  .section-title{{
    font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
    color:#2563eb;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #d0dff0;
  }}

  /* ── Patient data grid ── */
  .data-grid{{
    display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-bottom:26px;
  }}
  .data-item{{background:#fff;border:1px solid #e2ecf8;border-radius:8px;padding:10px 13px}}
  .data-key{{font-size:9px;color:#94a3b8;text-transform:uppercase;
    letter-spacing:.08em;margin-bottom:3px}}
  .data-val{{font-size:14px;font-weight:700;color:#1a2035}}

  /* ── Two-column layout ── */
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
  .rpt-card{{
    background:#fff;border:1px solid #d0dff0;border-radius:12px;
    padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,.04);
  }}
  .rpt-card img{{width:100%;border-radius:6px;margin-top:8px}}

  /* ── AI Summary ── */
  .ai-text p{{
    font-size:13.5px;line-height:1.85;color:#334155;
    margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #e8f0fb;
  }}
  .ai-text p:last-child{{border-bottom:none;margin-bottom:0;padding-bottom:0}}

  /* ── Print CTA (hidden when printing) ── */
  .print-cta{{
    text-align:center;margin:26px 0 8px;
  }}
  .print-cta button{{
    background:linear-gradient(135deg,#2563eb,#1d4ed8);
    color:#fff;border:none;border-radius:10px;
    padding:13px 34px;font-size:15px;font-weight:700;
    font-family:inherit;cursor:pointer;
    box-shadow:0 4px 18px rgba(37,99,235,0.38);letter-spacing:0.03em;
  }}
  .print-cta p{{font-size:12px;color:#94a3b8;margin-top:9px}}

  /* ── Footer ── */
  .rpt-footer{{
    margin-top:22px;padding-top:13px;border-top:1px solid #d0dff0;
    font-size:11px;color:#94a3b8;text-align:center;line-height:1.9;
  }}

  /* ── Print media ── */
  @media print{{
    body{{background:#fff;padding:18px 26px}}
    .print-cta{{display:none}}          /* hide button when printing */
    .rpt-card,.score-block{{box-shadow:none}}
    @page{{margin:1cm;size:A4 landscape}}
  }}
</style>
</head>
<body>

<div class="rpt-header">
  <div>
    <div class="rpt-title">&#x1FAC0;&nbsp; Cardio <span>Risk</span> Analyzer</div>
    <div style="font-size:12px;color:#64748b;margin-top:3px;">
      AI-Powered Heart Disease Risk Report
    </div>
  </div>
  <div class="rpt-meta">
    Generated: {report_date}<br>
    &#9888;&#xFE0F; For educational purposes only
  </div>
</div>

<div class="score-block">
  <div class="score-big">{risk_pct}%</div>
  <div>
    <div class="score-label">{risk_label}</div>
    <div class="score-sub">
      Based on 10 cardiovascular risk indicators.<br>
      This is <strong>not</strong> a medical diagnosis — please consult a qualified doctor.
    </div>
  </div>
</div>

<div class="section-title">&#x1F9D1;&#x200D;&#x2695;&#xFE0F;&nbsp; Patient Information</div>
<div class="data-grid">
  <div class="data-item"><div class="data-key">Age</div><div class="data-val">{age} yrs</div></div>
  <div class="data-item"><div class="data-key">Sex</div><div class="data-val">{sex}</div></div>
  <div class="data-item"><div class="data-key">Chest Pain</div><div class="data-val">{cp}</div></div>
  <div class="data-item"><div class="data-key">Blood Pressure</div><div class="data-val">{trestbps} mmHg</div></div>
  <div class="data-item"><div class="data-key">Cholesterol</div><div class="data-val">{chol} mg/dL</div></div>
  <div class="data-item"><div class="data-key">Max Heart Rate</div><div class="data-val">{thalach} bpm</div></div>
  <div class="data-item"><div class="data-key">Exercise Angina</div><div class="data-val">{exang}</div></div>
  <div class="data-item"><div class="data-key">ST Depression</div><div class="data-val">{oldpeak}</div></div>
  <div class="data-item"><div class="data-key">Major Vessels</div><div class="data-val">{ca}</div></div>
  <div class="data-item"><div class="data-key">Thalassemia</div><div class="data-val">{thal}</div></div>
</div>

<div class="two-col">
  <div class="rpt-card">
    <div class="section-title">&#x1F9E0;&nbsp; Risk Factor Analysis (SHAP)</div>
    <img src="data:image/png;base64,{chart_b64}" alt="SHAP Chart">
  </div>
  <div class="rpt-card">
    <div class="section-title">&#x1F4AC;&nbsp; AI Clinical Summary</div>
    <div class="ai-text">{para_html}</div>
  </div>
</div>

<div class="rpt-footer">
  &#9888; For Educational &amp; Research Purposes Only &nbsp;&middot;&nbsp;
  Not a substitute for professional medical advice &nbsp;&middot;&nbsp;
  &copy; 2026 All Rights Reserved &nbsp;&middot;&nbsp; preation07
</div>

<!-- Manual print button (hidden when actually printing via @media print) -->
<div class="print-cta">
  <button onclick="window.print()">&#x1F5A8;&#xFE0F;&nbsp; Save as PDF / Print</button>
  <p>Choose <strong>Save as PDF</strong> in the print dialog to download.</p>
</div>

<script>
  // No external fonts = page is fully rendered on load.
  // Auto-open print dialog immediately.
  window.addEventListener('load', function() {{
    window.print();
  }});
</script>

</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# ── HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Header section ── */
.hero-wrap {
    text-align: center;
    padding: 3.2rem 1rem 2rem;
    position: relative;
}

/* Subtle horizontal rule glow under the icon */
.hero-wrap::after {
    content: '';
    display: block;
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #4aa8ff, transparent);
    margin: 1.6rem auto 0;
    border-radius: 99px;
}

/* Pulse ring behind the icon */
.hero-icon-wrap {
    position: relative;
    display: inline-block;
    margin-bottom: 1rem;
}
.hero-icon-wrap::before {
    content: '';
    position: absolute;
    inset: -10px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(74,168,255,0.18) 0%, transparent 70%);
    animation: ripple 3s ease-in-out infinite;
}
@keyframes ripple {
    0%,100% { transform: scale(1);   opacity: 0.7; }
    50%      { transform: scale(1.2); opacity: 0.3; }
}
.hero-icon {
    font-size: 4rem;
    display: block;
    filter: drop-shadow(0 0 18px rgba(74,168,255,0.55));
    animation: beat 2.6s ease-in-out infinite;
}
@keyframes beat { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }

/* Title */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #ffffff 30%, #a8d4ff 70%, #4aa8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Tagline */
.hero-tag {
    font-size: 16px;
    font-weight: 400;
    color: rgba(180, 220, 255, 0.60) !important;
    letter-spacing: 0.01em;
    margin: 0 0 1rem;
    line-height: 1.6;
}

/* Pill badges row */
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 1.1rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: 1px solid rgba(74,168,255,0.25);
    background: rgba(74,168,255,0.09);
    color: rgba(180,220,255,0.75) !important;
}
</style>

<div class="hero-wrap">
  <div class="hero-icon-wrap">
    <span class="hero-icon">🫀</span>
  </div>
  <h1 class="hero-title">Cardio Risk Analyzer</h1>
  <p class="hero-tag">
    AI-powered heart disease risk assessment — explainable, personalised &amp; instant
  </p>
  <div class="hero-pills">
    <span class="pill">🔒 Session-only &nbsp;·&nbsp; Your data is never stored</span>
  </div>
</div>

<hr style="border:none;border-top:1px solid rgba(74,168,255,0.15);margin:0.5rem 0 1.8rem;">
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── INPUT FORM (centered in main area)
# ─────────────────────────────────────────────────────────────────────────────
_, form_col, _ = st.columns([1, 3.5, 1])

with form_col:

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<span class="sec-lbl">👤  Demographics</span>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="medium")

    with d1:
        age = st.slider("Age", 20, 100, 50)
        hint("Your current age in years.")

    with d2:
        sex_lbl = st.radio("Biological Sex", ["Male", "Female"], horizontal=True)
        hint("Biological sex influences cardiovascular risk profiles.")
        sex_val = 1 if sex_lbl == "Male" else 0

    # ── Heart Symptoms ────────────────────────────────────────────────────────
    st.markdown('<span class="sec-lbl">💓  Heart Symptoms</span>', unsafe_allow_html=True)
    h1, h2 = st.columns(2, gap="medium")

    with h1:
        cp_lbl = st.radio("Chest Pain Type", list(CHEST_PAIN_MAP.keys()))
        hint("Typical Angina = pressure on exertion · Asymptomatic = no chest pain.")
        cp_val = CHEST_PAIN_MAP[cp_lbl]

        exang_lbl = st.radio("Chest Pain During Exercise?",
                              ["No", "Yes"], horizontal=True)
        hint("Do you feel chest tightness or pain when physically active?")
        exang_val = 1 if exang_lbl == "Yes" else 0

    with h2:
        thalach = st.slider("Max Heart Rate Achieved (bpm)", 70, 220, 150)
        hint("Highest heart rate recorded during exercise or a stress test.")

        oldpeak = st.slider("ST Depression (ECG)", 0.0, 6.0, 1.0, step=0.1)
        hint("How much the ECG ST-segment dips during exercise. Higher = more concern.")

    # ── Lab Results ───────────────────────────────────────────────────────────
    st.markdown('<span class="sec-lbl">🧪  Lab & Imaging Results</span>',
                unsafe_allow_html=True)
    l1, l2 = st.columns(2, gap="medium")

    with l1:
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
        hint("Your blood pressure at rest. Healthy range: 90–120 mmHg.")

        ca_lbl = st.radio("Major Vessels Visible on Imaging",
                           ["0","1","2","3"], horizontal=True)
        hint("Number of major heart vessels coloured during fluoroscopy. 0 = best.")
        ca_val = int(ca_lbl)

    with l2:
        chol = st.slider("Serum Cholesterol (mg/dL)", 100, 400, 200)
        hint("Total cholesterol from a blood test. Desirable: below 200 mg/dL.")

        thal_lbl = st.radio("Thalassemia Status", list(THAL_MAP.keys()))
        hint("Heart blood-flow pattern on nuclear stress test. "
             "Reversible Defect = highest risk.")
        thal_val = THAL_MAP[thal_lbl]

    # ── Analyze button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("🔍  Analyze My Risk", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hidden defaults for low-importance features
# fbs=0 (most common), restecg=0 (normal), slope=1 (flat / population median)
# ─────────────────────────────────────────────────────────────────────────────
input_df = pd.DataFrame(
    [[age, sex_val, cp_val, trestbps, chol,
      0, 0, thalach, exang_val, oldpeak, 1, ca_val, thal_val]],
    columns=ALL_FEATURES,
)


# ─────────────────────────────────────────────────────────────────────────────
# ── RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if analyze:

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(74,168,255,0.15);margin:1.5rem 0 0.5rem;">
    <span class="results-label">🫀 &nbsp; Your Risk Results</span>
    """, unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    try:
        prob     = float(model.predict_proba(input_df)[0][1])
        risk_pct = round(prob * 100, 2)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    risk_int = min(int(risk_pct), 100)

    if risk_pct >= 65:
        clr, bg_cls, lbl, emoji, bar_color = \
            "clr-high", "bg-high", "High Risk",     "🔴", "#e8445a"
    elif risk_pct >= 35:
        clr, bg_cls, lbl, emoji, bar_color = \
            "clr-mid",  "bg-mid",  "Moderate Risk", "🟡", "#fbbf24"
    else:
        clr, bg_cls, lbl, emoji, bar_color = \
            "clr-low",  "bg-low",  "Low Risk",      "🟢", "#3dd68c"

    # Dynamic progress bar colour
    st.markdown(f"""
    <style>
    [data-testid="stProgress"] > div > div {{ background: {bar_color} !important; }}
    </style>""", unsafe_allow_html=True)

    # ── Score card (full width, centered) ─────────────────────────────────────
    _, sc, _ = st.columns([1, 3.5, 1])
    with sc:
        st.markdown(f"""
        <div class="score-card res-wrap res-delay-1">
          <div style="font-size:11px;letter-spacing:.15em;text-transform:uppercase;
                      color:rgba(255,255,255,0.30);margin-bottom:.9rem;">
            Heart Disease Risk Score
          </div>
          <div class="score-num {clr}">{emoji}&nbsp;{risk_pct}%</div>
          <div><span class="score-badge {bg_cls}">{lbl}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(risk_int)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="results-label">📊 &nbsp; Detailed Analysis</span>',
                unsafe_allow_html=True)

    # ── Graph + AI Summary — side by side ─────────────────────────────────────
    graph_col, summary_col = st.columns([1.05, 1], gap="large")

    # ── LEFT: SHAP chart ──────────────────────────────────────────────────────
    with graph_col:
        st.markdown("""
        <div class="panel panel-graph res-wrap res-delay-2">
          <div class="panel-head">🧠 &nbsp; What drove this score?</div>
          <div class="chart-legend">
            <span><span class="leg-dot" style="background:#e8445a;"></span>
              <strong style="color:#ff8fa3;">Red</strong> &nbsp;increased risk</span>
            <span><span class="leg-dot" style="background:#2ecfbd;"></span>
              <strong style="color:#6ee7df;">Teal</strong> &nbsp;decreased risk</span>
          </div>
        """, unsafe_allow_html=True)

        try:
            shap_raw = explainer.shap_values(input_df)
            if isinstance(shap_raw, list):
                sv_full = np.array(shap_raw[1]).flatten()
            else:
                sv_full = np.array(shap_raw).flatten()

            all_cols = list(input_df.columns)
            user_idx = [all_cols.index(f) for f in USER_FEATURES]
            sv_user  = sv_full[user_idx]
            fn_user  = [FRIENDLY[f] for f in USER_FEATURES]

            ev   = explainer.expected_value
            base = float(np.array(ev).flat[-1]) if hasattr(ev, "__len__") else float(ev)

            fig = shap_bar_chart(sv_user, fn_user, base)
            st.session_state["last_fig"] = fig   # keep for print report
            st.pyplot(fig, use_container_width=True)
            # Note: do NOT close fig here — needed by print report builder

        except Exception as e:
            st.warning(f"SHAP chart could not be rendered: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT: AI Summary ──────────────────────────────────────────────────────
    with summary_col:
        st.markdown("""
        <div class="panel panel-ai res-wrap res-delay-3">
          <div class="panel-head">💬 &nbsp; AI Clinical Summary</div>
        """, unsafe_allow_html=True)

        if not groq_api_key:
            st.warning("Add `GROQ_API_KEY` to `.streamlit/secrets.toml` "
                       "to enable AI summaries.")
        else:
            with st.spinner("Generating your personalised summary…"):
                try:
                    client  = Groq(api_key=groq_api_key)
                    summary = groq_summary(
                        client, risk_pct, lbl,
                        age, sex_lbl, cp_lbl,
                        trestbps, chol, thalach,
                        exang_lbl, oldpeak, ca_val, thal_lbl,
                    )
                    paras     = [p.strip() for p in summary.split("\n\n") if p.strip()]
                    para_html = "".join(f"<p>{p}</p>" for p in paras)
                    st.markdown(f'<div class="ai-block">{para_html}</div>',
                                unsafe_allow_html=True)
                    # Store for print report
                    st.session_state["last_summary"] = summary
                except Exception as e:
                    st.error(f"Groq API error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Print / Save button ─────────────────────────────────────────────────────
    # Builds a self-contained HTML report, encodes it as a data-URI,
    # and opens it in a new tab — which then auto-prints.
    # This bypasses the Streamlit iframe so printing actually works.
    st.markdown("<br>", unsafe_allow_html=True)

    if "last_summary" in st.session_state and "last_fig" in st.session_state:
        report_html = build_print_report(
            risk_pct, lbl,
            age, sex_lbl, cp_lbl,
            trestbps, chol, thalach,
            exang_lbl, oldpeak, ca_val, thal_lbl,
            st.session_state["last_summary"],
            st.session_state["last_fig"],
        )
        # Encode full HTML as a data URI and open in new tab via JS
        encoded = base64.b64encode(report_html.encode()).decode()
        components.html(f"""
        <style>
          .print-btn {{
            display: flex; align-items: center; justify-content: center;
            gap: 9px; width: 100%; padding: 0.72rem 1.4rem;
            border-radius: 11px; border: 1px solid rgba(74,168,255,0.40);
            background: rgba(74,168,255,0.12);
            color: rgba(180,220,255,0.88); font-family: 'Outfit', sans-serif;
            font-size: 15px; font-weight: 600; letter-spacing: 0.03em;
            cursor: pointer; text-decoration: none; transition: all 0.2s ease;
            box-shadow: 0 3px 16px rgba(74,168,255,0.18);
          }}
          .print-btn:hover {{
            background: rgba(74,168,255,0.22); color: #fff;
            box-shadow: 0 6px 26px rgba(74,168,255,0.35);
            transform: translateY(-1px);
          }}
        </style>
        <a class="print-btn"
           href="data:text/html;base64,{encoded}"
           target="_blank">
          🖨️ &nbsp; Open Print Report in New Tab
        </a>
        <p style="font-size:13.5px;color:rgba(180,210,255,0.58);
                  text-align:center;margin-top:11px;font-style:italic;
                  font-weight:500;letter-spacing:0.01em;">
          &#x2139;&#xFE0F;&nbsp; If the report opens blank — refresh that tab once to load it.
        </p>
        """, height=80)

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown("<br>", unsafe_allow_html=True)
    _, ec, _ = st.columns([1, 3, 1])
    with ec:
        st.markdown("""
        <div class="empty">
          <span class="empty-icon">🫀</span>
          <div style="font-size:1.15rem;font-weight:700;margin-bottom:8px;
                      font-family:'Outfit',sans-serif;">
            Your risk assessment will appear here
          </div>
          <div style="font-size:13.5px;color:rgba(255,255,255,0.30);line-height:1.75;">
            Complete all sections above and press
            <strong style="color:rgba(255,255,255,0.55);">Analyze My Risk</strong>.
          </div>
          <br>
          <div style="font-size:11px;color:rgba(255,255,255,0.15);">
            Your data stays in this session only &nbsp;·&nbsp; No account needed
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# TODO: Replace the empty string with your portfolio URL
#       Example → PORTFOLIO_URL = "https://preation07.dev"
# ─────────────────────────────────────────────────────────────────────────────
PORTFOLIO_URL = ""   # ← YOUR PORTFOLIO LINK HERE

st.markdown(f"""
<div class="disc">
  ⚠️ <strong>For Educational &amp; Research Purposes Only.</strong><br>
  This tool is not a substitute for professional medical advice, diagnosis, or treatment.
  Always consult a qualified healthcare provider before making any health decision.<br><br>
  © 2026 All Rights Reserved &nbsp;·&nbsp;
  <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer">preation07</a>
</div>
""", unsafe_allow_html=True)