"""
app.py
------
Streamlit dashboard for the Credit-Risk XGBoost model.

Tabs
  1. Model Performance  - ROC curve, AUC-ROC, Gini, KS statistic
  2. SHAP Explorer       - placeholder
  3. Applicant Predictor - placeholder

Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark header bar */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(100, 100, 255, 0.15);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #9d9db5;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c6aff, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ---------- Tab placeholder ---------- */
    .placeholder-box {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 60px 40px;
        text-align: center;
        margin-top: 40px;
    }
    .placeholder-box h2 {
        color: #c084fc;
        margin-bottom: 10px;
    }
    .placeholder-box p {
        color: #9d9db5;
        font-size: 1.05rem;
    }

    /* ---------- Section header ---------- */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e0e0ec;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 0.85rem;
        color: #7a7a96;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING  (cached so it only runs once)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Load and prepare the dataset (same pipeline as train_model.py)."""
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "cs-training.csv"))

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    TARGET = "SeriousDlqin2yrs"
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Median imputation
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@st.cache_resource
def load_model():
    """Load the persisted XGBoost model."""
    base = os.path.dirname(__file__)
    return joblib.load(os.path.join(base, "model", "xgb_model.pkl"))


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## Credit Risk Model Dashboard")
st.caption("XGBoost classifier trained on the *Give Me Some Credit* dataset")

tab_perf, tab_shap, tab_predict = st.tabs(
    ["Model Performance", "SHAP Explorer", "Applicant Predictor"]
)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    # Load data & model
    X_train, X_test, y_train, y_test = load_data()
    model = load_model()

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    auc_roc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc_roc - 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks_from_roc = float(max(tpr - fpr))
    ks_stat, ks_pval = ks_2samp(
        y_prob[y_test == 1], y_prob[y_test == 0]
    )

    # --- Metric cards ---
    st.markdown(
        '<p class="section-header">Key Metrics</p>'
        '<p class="section-sub">Evaluated on a held-out 20% stratified test split (same seed as training)</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AUC-ROC</div>
                <div class="metric-value">{auc_roc:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Gini Coefficient</div>
                <div class="metric-value">{gini:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">KS Statistic</div>
                <div class="metric-value">{ks_stat:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROC Curve (Plotly) ---
    st.markdown(
        '<p class="section-header">ROC Curve</p>'
        '<p class="section-sub">Receiver Operating Characteristic — true positive rate vs false positive rate</p>',
        unsafe_allow_html=True,
    )

    # Find the KS point (max TPR - FPR)
    ks_idx = np.argmax(tpr - fpr)

    fig = go.Figure()

    # Diagonal (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc_roc:.4f})",
            line=dict(
                color="#7c6aff",
                width=3,
            ),
            fill="tozeroy",
            fillcolor="rgba(124,106,255,0.10)",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    # KS point
    fig.add_trace(
        go.Scatter(
            x=[fpr[ks_idx]],
            y=[tpr[ks_idx]],
            mode="markers+text",
            name=f"KS = {ks_from_roc:.4f}",
            marker=dict(color="#c084fc", size=12, symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"  KS = {ks_from_roc:.4f}"],
            textposition="middle right",
            textfont=dict(color="#c084fc", size=13, family="Inter"),
            hovertemplate=(
                f"KS = {ks_from_roc:.4f}<br>"
                f"FPR = {fpr[ks_idx]:.3f}<br>"
                f"TPR = {tpr[ks_idx]:.3f}<extra></extra>"
            ),
        )
    )

    # KS vertical line
    fig.add_shape(
        type="line",
        x0=fpr[ks_idx], x1=fpr[ks_idx],
        y0=fpr[ks_idx], y1=tpr[ks_idx],
        line=dict(color="#c084fc", width=2, dash="dot"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,47,1)",
        xaxis=dict(title="False Positive Rate", range=[0, 1],
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="True Positive Rate", range=[0, 1.02],
                   gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(
            x=0.55, y=0.08,
            bgcolor="rgba(0,0,0,0.4)",
            borderwidth=0,
            font=dict(size=13),
        ),
        margin=dict(l=60, r=30, t=30, b=60),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SHAP EXPLORER  (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.markdown(
        """
        <div class="placeholder-box">
            <h2>SHAP Explorer</h2>
            <p>This tab will provide interactive SHAP explanations for the model's predictions.<br>
            Coming soon.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — APPLICANT PREDICTOR  (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        """
        <div class="placeholder-box">
            <h2>Applicant Predictor</h2>
            <p>This tab will let you enter applicant details and get a real-time default probability.<br>
            Coming soon.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
