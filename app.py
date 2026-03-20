"""
app.py
------
Streamlit dashboard for the Credit-Risk XGBoost model.

Tabs
  1. Model Performance  - ROC curve, AUC-ROC, Gini, KS statistic
    2. SHAP Explorer       - global + local SHAP explanations
    3. Applicant Predictor - single-applicant scoring + narrative

Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from openai import OpenAI
import shap
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

DATASET_CANDIDATES = (
    "cs-training.csv",
    os.path.join("data", "cs-training.csv"),
)

FEATURE_META = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Revolving Utilization",
        "desc": "Credit card balance / credit limit ratio",
    },
    "age": {
        "label": "Age",
        "desc": "Borrower's age in years",
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "30-59 Days Past Due",
        "desc": "Times 30-59 days late in last 2 years",
    },
    "DebtRatio": {
        "label": "Debt Ratio",
        "desc": "Monthly debt payments / gross income",
    },
    "MonthlyIncome": {
        "label": "Monthly Income",
        "desc": "Borrower's monthly gross income",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open Credit Lines",
        "desc": "Number of open loans and credit lines",
    },
    "NumberOfTimes90DaysLate": {
        "label": "90+ Days Late",
        "desc": "Times 90+ days delinquent",
    },
    "NumberRealEstateLoansOrLines": {
        "label": "Real Estate Loans",
        "desc": "Number of mortgage and real estate loans",
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "60-89 Days Past Due",
        "desc": "Times 60-89 days late in last 2 years",
    },
    "NumberOfDependents": {
        "label": "Dependents",
        "desc": "Number of dependents in the family",
    },
}


def resolve_dataset_path():
    """Return the first dataset path that exists in the repository, else None."""
    base = os.path.dirname(__file__)
    for rel_path in DATASET_CANDIDATES:
        candidate = os.path.join(base, rel_path)
        if os.path.exists(candidate):
            return candidate
    return None


def dataset_source_signature(uploaded_file):
    """Build a lightweight cache key that changes with source updates."""
    if uploaded_file is not None:
        return f"upload:{uploaded_file.name}:{uploaded_file.size}"

    dataset_path = resolve_dataset_path()
    if dataset_path is None:
        return "missing"

    return f"path:{dataset_path}:{os.path.getmtime(dataset_path)}"


def read_dataset_dataframe(uploaded_file=None):
    """Load dataset from repository paths first, then uploaded CSV fallback."""
    dataset_path = resolve_dataset_path()
    if dataset_path is not None:
        return pd.read_csv(dataset_path)

    if uploaded_file is not None:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    raise FileNotFoundError(
        "Dataset file not found. Expected one of: cs-training.csv or data/cs-training.csv"
    )


def render_missing_dataset_help(error):
    """Show a user-facing recovery message when dataset is unavailable."""
    st.error("Dataset not found. The app cannot run without cs-training.csv.")
    st.caption(str(error))
    st.info(
        "For Streamlit Cloud, either commit the CSV as data/cs-training.csv, "
        "or upload it with the sidebar uploader."
    )

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon=os.path.join(os.path.dirname(__file__), "assets", "favicon.svg"),
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Prevent Material icon ligature text (e.g., keyboard_double_arrow_right) */
    .material-icons {
        font-family: 'Material Icons' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }

    /* Dark header bar */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0f0c29, #302b63, #24243e);
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 24px;
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
        border-radius: 32px;
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

uploaded_dataset = st.sidebar.file_uploader(
    "Upload cs-training.csv",
    type=["csv"],
    help="Use this on Streamlit Cloud if the dataset is not committed in the repository.",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING  (cached so it only runs once)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(dataset_signature, uploaded_file=None):
    """Load and prepare the dataset (same pipeline as train_model.py)."""
    del dataset_signature
    df = read_dataset_dataframe(uploaded_file)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    TARGET = "SeriousDlqin2yrs"
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Median imputation
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

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


@st.cache_resource
def get_shap_explainer():
    """Build one shared TreeExplainer instance for all tabs."""
    try:
        return shap.TreeExplainer(load_model())
    except Exception:
        # Some cloud runtime combinations (e.g., newer Python/XGBoost builds)
        # can fail in SHAP TreeExplainer internals.
        return None


@st.cache_resource
def is_shap_ready():
    """Expose SHAP availability as a shared flag for all tabs."""
    return get_shap_explainer() is not None


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
    dataset_signature = dataset_source_signature(uploaded_dataset)

    try:
        X_train, X_test, y_train, y_test = load_data(dataset_signature, uploaded_dataset)
    except FileNotFoundError as err:
        render_missing_dataset_help(err)
        st.stop()

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
        hoverlabel=dict(
            bgcolor="#1e1e2f",
            bordercolor="#7c6aff",
            font=dict(size=14, color="#e0e0ec", family="Inter, sans-serif"),
            align="left",
        ),
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Feature Importance Bar Chart ---
    st.markdown(
        '<p class="section-header">Feature Importance</p>'
        '<p class="section-sub">Which variables XGBoost weighted most when making predictions (built-in gain-based importance)</p>',
        unsafe_allow_html=True,
    )

    # Step 1: Extract importance scores stored inside the trained model
    importances = model.feature_importances_
    feature_names = X_test.columns.tolist()

    # Step 2: Map raw names to clean labels and descriptions
    clean_labels = [FEATURE_META.get(f, {}).get("label", f) for f in feature_names]
    descriptions = [FEATURE_META.get(f, {}).get("desc", "") for f in feature_names]

    # Step 3: Put into a DataFrame and sort ascending (top bar = most important)
    feat_imp = pd.DataFrame({
        "Feature": clean_labels,
        "Importance": importances,
        "Description": descriptions,
    })
    feat_imp = feat_imp.sort_values("Importance", ascending=True)

    # Step 4: Build horizontal bar chart with gradient coloring & hover descriptions
    fig_imp = go.Figure(
        go.Bar(
            x=feat_imp["Importance"],
            y=feat_imp["Feature"],
            orientation="h",
            marker=dict(
                color=feat_imp["Importance"],
                colorscale=[[0, "#312e81"], [0.5, "#7c6aff"], [1, "#c084fc"]],
            ),
            customdata=feat_imp["Description"],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata}<br><br>"
                "Importance: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig_imp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,47,1)",
        xaxis=dict(title="Importance Score",
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=30, t=20, b=50),
        height=420,
        hoverlabel=dict(
            bgcolor="#1e1e2f",
            bordercolor="#c084fc",  # Match the accent color
            font=dict(size=14, color="#e0e0ec", family="Inter, sans-serif"),
            align="left",
            namelength=-1,
        ),
    )

    st.plotly_chart(fig_imp, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SHAP EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    st.markdown(
        '<p class="section-header">SHAP Explorer</p>'
        '<p class="section-sub">Responsive global explanations with configurable sample size and feature depth</p>',
        unsafe_allow_html=True,
    )

    @st.cache_data
    def load_shap_sample(n_rows=500, dataset_signature=""):
        """Load and preprocess a sample for SHAP analysis."""
        del dataset_signature
        df = read_dataset_dataframe(uploaded_dataset)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        target_col = "SeriousDlqin2yrs"
        if target_col in df.columns:
            X_sample = df.drop(columns=[target_col]).head(n_rows).copy()
        else:
            X_sample = df.head(n_rows).copy()

        for col in X_sample.columns:
            if X_sample[col].isnull().any():
                X_sample[col] = X_sample[col].fillna(X_sample[col].median())

        return X_sample

    @st.cache_data
    def compute_shap_bundle(n_rows=500, dataset_signature=""):
        """Return model-aligned sample, SHAP values, and SHAP base value."""
        X_sample = load_shap_sample(n_rows, dataset_signature)
        model = load_model()

        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None:
            X_sample = X_sample.loc[:, model_features]

        explainer = get_shap_explainer()
        if explainer is None:
            raise RuntimeError("SHAP explainer is unavailable in this environment")
        shap_values_local = explainer.shap_values(X_sample)
        base_value_local = explainer.expected_value

        # Binary classifiers may return a list per class; use positive class effects.
        if isinstance(shap_values_local, list):
            shap_values_local = shap_values_local[-1]
        if isinstance(base_value_local, (list, np.ndarray)):
            base_value_local = base_value_local[-1]

        return X_sample, shap_values_local, float(base_value_local)

    c_rows, c_features, c_mode = st.columns(3)
    with c_rows:
        sample_rows = st.slider("Rows", min_value=200, max_value=2000, value=500, step=100)
    with c_features:
        max_display = st.slider("Top Features", min_value=5, max_value=20, value=12, step=1)
    with c_mode:
        plot_mode = st.selectbox("View", options=["Beeswarm", "Bar"], index=0)

    shap_ready = is_shap_ready()
    if shap_ready:
        try:
            with st.spinner("Computing SHAP values..."):
                X_shap, shap_values, base_value = compute_shap_bundle(
                    sample_rows,
                    dataset_source_signature(uploaded_dataset),
                )
        except FileNotFoundError as err:
            render_missing_dataset_help(err)
            st.stop()
    else:
        st.error(
            "SHAP is temporarily unavailable in this deployment environment. "
            "The rest of the dashboard is still fully functional."
        )

    if shap_ready:
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows Analyzed", f"{len(X_shap):,}")
        m2.metric("Features Used", f"{X_shap.shape[1]:,}")
        m3.metric("View", plot_mode)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
        top_features = X_shap.columns[top_idx].tolist()
        top_feature_labels = [FEATURE_META.get(f, {}).get("label", f) for f in top_features]

        if plot_mode == "Bar":
            bar_df = pd.DataFrame(
                {
                    "Feature": top_feature_labels,
                    "Mean |SHAP|": mean_abs_shap[top_idx],
                }
            ).sort_values("Mean |SHAP|", ascending=True)

            fig_shap = go.Figure(
                go.Bar(
                    x=bar_df["Mean |SHAP|"],
                    y=bar_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=bar_df["Mean |SHAP|"],
                        colorscale=[[0, "#312e81"], [0.5, "#7c6aff"], [1, "#c084fc"]],
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>"
                    ),
                )
            )

            fig_shap.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,30,47,1)",
                xaxis=dict(title="Mean Absolute SHAP Value", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=20, r=30, t=20, b=50),
                height=max(360, min(560, 34 * max_display + 80)),
                hoverlabel=dict(
                    bgcolor="#1e1e2f",
                    bordercolor="#c084fc",
                    font=dict(size=14, color="#e0e0ec", family="Inter, sans-serif"),
                    align="left",
                ),
            )
        else:
            rng = np.random.default_rng(42)

            x_all = []
            y_all = []
            c_all = []
            feature_all = []
            raw_all = []

            for row_idx, col_idx in enumerate(top_idx):
                shap_col = shap_values[:, col_idx]
                feat_col = pd.to_numeric(X_shap.iloc[:, col_idx], errors="coerce").to_numpy()
                feat_col = np.nan_to_num(feat_col, nan=np.nanmedian(feat_col))

            # Match SHAP-style coloring by scaling values within each feature.
                if np.ptp(feat_col) == 0:
                    feat_col_norm = np.full_like(feat_col, 0.5, dtype=float)
                else:
                    order = np.argsort(feat_col, kind="mergesort")
                    ranks = np.empty_like(order, dtype=float)
                    ranks[order] = np.linspace(0.0, 1.0, len(feat_col))
                    feat_col_norm = ranks

                jitter = rng.uniform(-0.28, 0.28, size=len(shap_col))
                y_pos = np.full(len(shap_col), row_idx) + jitter

                x_all.append(shap_col)
                y_all.append(y_pos)
                c_all.append(feat_col_norm)
                clean_label = FEATURE_META.get(X_shap.columns[col_idx], {}).get("label", X_shap.columns[col_idx])
                feature_all.append(np.repeat(clean_label, len(shap_col)))
                raw_all.append(feat_col)

            x_plot = np.concatenate(x_all)
            y_plot = np.concatenate(y_all)
            color_plot = np.concatenate(c_all)
            feature_plot = np.concatenate(feature_all)
            raw_plot = np.concatenate(raw_all)

            fig_shap = go.Figure(
                go.Scattergl(
                    x=x_plot,
                    y=y_plot,
                    mode="markers",
                    customdata=np.column_stack([feature_plot, raw_plot]),
                    marker=dict(
                        size=6,
                        opacity=0.75,
                        color=color_plot,
                        colorscale=[[0, "#1296f3"], [0.5, "#7c6aff"], [1, "#ff0d57"]],
                        colorbar=dict(title="Feature value", thickness=12),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>SHAP: %{x:.5f}<br>Value: %{customdata[1]:.4f}<extra></extra>"
                    ),
                )
            )

            fig_shap.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,30,47,1)",
                xaxis=dict(title="SHAP value (impact on model output)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(
                    title="",
                    tickmode="array",
                    tickvals=list(range(len(top_features))),
                    ticktext=top_feature_labels,
                    autorange="reversed",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                margin=dict(l=20, r=30, t=20, b=60),
                height=max(390, min(620, 34 * max_display + 110)),
                hoverlabel=dict(
                    bgcolor="#1e1e2f",
                    bordercolor="#c084fc",
                    font=dict(size=14, color="#e0e0ec", family="Inter, sans-serif"),
                    align="left",
                ),
                shapes=[
                    dict(
                        type="line",
                        x0=0,
                        x1=0,
                        y0=-0.5,
                        y1=len(top_features) - 0.5,
                        line=dict(color="rgba(255,255,255,0.4)", width=1.3, dash="dot"),
                    )
                ],
            )

        st.plotly_chart(fig_shap, width="stretch")

        st.markdown(
            '<p class="section-header">Applicant Waterfall</p>'
            '<p class="section-sub">Single-row SHAP breakdown for one applicant from the sampled rows</p>',
            unsafe_allow_html=True,
        )

        waterfall_row = st.number_input(
            "Row Index (0-499)",
            min_value=0,
            max_value=499,
            value=0,
            step=1,
        )

        selected_row = int(min(waterfall_row, len(X_shap) - 1))
        if selected_row != waterfall_row:
            st.info(f"Selected row adjusted to {selected_row} because current sample size is {len(X_shap)}.")

        clean_feature_names = [FEATURE_META.get(f, {}).get("label", f) for f in X_shap.columns]
        row_values = X_shap.iloc[selected_row].to_numpy()
        row_shap = shap_values[selected_row]
        pred_value = float(base_value + np.sum(row_shap))

        top_idx = np.argsort(np.abs(row_shap))[::-1][:max_display]
        remaining_contrib = float(np.sum(row_shap) - np.sum(row_shap[top_idx]))

        # Keep SHAP-like order: strongest absolute effects first.
        ordered_idx = top_idx

        y_labels = []
        x_values = []
        measures = []
        text_labels = []
        customdata = []

        for idx in ordered_idx:
            feat_val = row_values[idx]
            feat_name = clean_feature_names[idx]
            contrib = float(row_shap[idx])

            if float(feat_val).is_integer():
                feat_txt = f"{int(feat_val)}"
            elif abs(feat_val) >= 1000:
                feat_txt = f"{feat_val:.0f}"
            else:
                feat_txt = f"{feat_val:.3g}"

            y_labels.append(f"{feat_txt} = {feat_name}")
            x_values.append(contrib)
            measures.append("relative")
            text_labels.append(f"{contrib:+.2f}")
            customdata.append([feat_name, feat_val, contrib])

        if abs(remaining_contrib) > 1e-6:
            y_labels.append(f"{len(X_shap.columns) - len(top_idx)} other features")
            x_values.append(remaining_contrib)
            measures.append("relative")
            text_labels.append(f"{remaining_contrib:+.2f}")
            customdata.append(["other features", np.nan, remaining_contrib])

        fig_wf = go.Figure(
            go.Waterfall(
                orientation="h",
                y=y_labels,
                x=x_values,
                measure=measures,
                base=float(base_value),
                text=text_labels,
                textposition="outside",
                textfont={"size": 15},
                customdata=np.array(customdata, dtype=object),
                connector={"line": {"color": "rgba(210, 210, 230, 0.35)", "width": 1.4, "dash": "dot"}},
                increasing={"marker": {"color": "#ff005e"}},
                decreasing={"marker": {"color": "#119DFF"}},
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Contribution: %{customdata[2]:+.5f}<br>"
                    "Feature value: %{customdata[1]:.4f}<extra></extra>"
                ),
            )
        )

        fig_wf.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,1)",
            xaxis=dict(
                title="Model output",
                gridcolor="rgba(255,255,255,0.15)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.35)",
                zerolinewidth=1.2,
            ),
            yaxis=dict(
                title="",
                gridcolor="rgba(255,255,255,0.14)",
                automargin=True,
                autorange="reversed",
            ),
            margin=dict(l=300, r=35, t=18, b=65),
            height=max(440, min(760, 54 * len(y_labels) + 90)),
            hoverlabel=dict(
                bgcolor="#1e1e2f",
                bordercolor="#c084fc",
                font=dict(size=14, color="#e0e0ec", family="Inter, sans-serif"),
                align="left",
            ),
            shapes=[
                dict(
                    type="line",
                    x0=float(base_value),
                    x1=float(base_value),
                    y0=-0.5,
                    y1=max(0, len(y_labels) - 0.5),
                    line=dict(color="rgba(255,255,255,0.30)", width=1.2, dash="dot"),
                ),
                dict(
                    type="line",
                    x0=pred_value,
                    x1=pred_value,
                    y0=-0.5,
                    y1=max(0, len(y_labels) - 0.5),
                    line=dict(color="rgba(255,255,255,0.30)", width=1.2, dash="dot"),
                ),
            ],
            annotations=[
                dict(
                    x=float(base_value),
                    y=-0.14,
                    xref="x",
                    yref="paper",
                    text=f"E[f(X)] = {float(base_value):.2f}",
                    showarrow=False,
                    font=dict(size=14, color="rgba(220,220,235,0.95)"),
                ),
                dict(
                    x=pred_value,
                    y=-0.14,
                    xref="x",
                    yref="paper",
                    text=f"f(x) = {pred_value:.2f}",
                    showarrow=False,
                    font=dict(size=14, color="rgba(220,220,235,0.95)"),
                ),
            ],
        )

        st.plotly_chart(fig_wf, width="stretch")

        st.caption(
            "Tip: Reduce rows on slower machines or mobile for faster and cleaner rendering."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — APPLICANT PREDICTOR  (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        '<p class="section-header">Applicant Predictor</p>'
        '<p class="section-sub">Enter applicant attributes and score a single case with SHAP explanation</p>',
        unsafe_allow_html=True,
    )

    model = load_model()
    model_features = list(getattr(model, "feature_names_in_", []))

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        revolving_utilization_value = st.number_input(
            "Revolving Utilization",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01,
            format="%.2f",
            help="Credit card balance / credit limit ratio",
        )
        age_value = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=45,
            step=1,
            help="Borrower age in years",
        )
        past_due_30_59_value = st.number_input(
            "30-59 Days Past Due",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Times 30-59 days late in last 2 years",
        )
        debt_ratio_value = st.number_input(
            "Debt Ratio",
            min_value=0.0,
            max_value=5.0,
            value=0.80,
            step=0.01,
            format="%.2f",
            help="Monthly debt payments / gross income",
        )
        monthly_income_value = st.number_input(
            "Monthly Income",
            min_value=0,
            max_value=50000,
            value=5000,
            step=100,
            help="Borrower monthly gross income",
        )
        open_credit_lines_value = st.number_input(
            "Open Credit Lines",
            min_value=0,
            max_value=50,
            value=8,
            step=1,
            help="Number of open loans and credit lines",
        )
        past_due_90_value = st.number_input(
            "90+ Days Late",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Times 90+ days delinquent",
        )
        real_estate_loans_value = st.number_input(
            "Real Estate Loans",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of mortgage and real estate loans",
        )
        past_due_60_89_value = st.number_input(
            "60-89 Days Past Due",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Times 60-89 days late in last 2 years",
        )
        dependents_value = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Number of dependents in the family",
        )

        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

        input_payload = {
            "RevolvingUtilizationOfUnsecuredLines": float(revolving_utilization_value),
            "age": int(age_value),
            "NumberOfTime30-59DaysPastDueNotWorse": int(past_due_30_59_value),
            "DebtRatio": float(debt_ratio_value),
            "MonthlyIncome": int(monthly_income_value),
            "NumberOfOpenCreditLinesAndLoans": int(open_credit_lines_value),
            "NumberOfTimes90DaysLate": int(past_due_90_value),
            "NumberRealEstateLoansOrLines": int(real_estate_loans_value),
            "NumberOfTime60-89DaysPastDueNotWorse": int(past_due_60_89_value),
            "NumberOfDependents": int(dependents_value),
        }

        if predict_clicked:
            input_df = pd.DataFrame([input_payload])
            if model_features:
                input_df = input_df.loc[:, model_features]

            pd_prob = float(model.predict_proba(input_df)[0, 1])

            if not is_shap_ready():
                st.session_state["predictor_result"] = {
                    "pd_prob": pd_prob,
                    "input_row": input_df.iloc[0].to_dict(),
                    "shap_available": False,
                }
            else:
                explainer = get_shap_explainer()
                row_shap_values = explainer.shap_values(input_df)
                row_base_value = explainer.expected_value

                if isinstance(row_shap_values, list):
                    row_shap_values = row_shap_values[-1]
                if isinstance(row_base_value, (list, np.ndarray)):
                    row_base_value = row_base_value[-1]

                st.session_state["predictor_result"] = {
                    "pd_prob": pd_prob,
                    "input_row": input_df.iloc[0].to_dict(),
                    "shap_row": np.asarray(row_shap_values[0], dtype=float).tolist(),
                    "base_value": float(row_base_value),
                    "shap_available": True,
                }

    with col_right:
        if "predictor_result" not in st.session_state:
            st.info("Click Predict to score the applicant and view explanation.")
        else:
            predictor_result = st.session_state["predictor_result"]
            pd_prob = float(predictor_result["pd_prob"])

            if pd_prob < 0.10:
                risk_color = "#22c55e"
                risk_label = "Low Risk"
            elif pd_prob <= 0.30:
                risk_color = "#f59e0b"
                risk_label = "Moderate Risk"
            else:
                risk_color = "#ef4444"
                risk_label = "High Risk"

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #17192a 0%, #22263a 100%);
                    border: 1px solid {risk_color};
                    border-radius: 20px;
                    padding: 20px 24px;
                    margin-bottom: 16px;
                    box-shadow: 0 10px 24px rgba(0,0,0,0.25);
                ">
                    <div style="font-size:0.82rem;color:#9d9db5;letter-spacing:1.1px;text-transform:uppercase;">
                        Probability Of Default (PD)
                    </div>
                    <div style="font-size:2.2rem;font-weight:700;color:{risk_color};margin-top:6px;line-height:1;">
                        {pd_prob * 100:.2f}%
                    </div>
                    <div style="margin-top:8px;color:#c9ccda;font-size:0.95rem;">{risk_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not predictor_result.get("shap_available", False):
                st.warning(
                    "SHAP explanation is unavailable in this deployment environment, "
                    "but the PD score is valid."
                )
            else:
                input_row_series = pd.Series(predictor_result["input_row"])
                shap_row = np.asarray(predictor_result["shap_row"], dtype=float)
                base_value = float(predictor_result["base_value"])
                clean_feature_names = [FEATURE_META.get(f, {}).get("label", f) for f in input_row_series.index]

                row_explanation = shap.Explanation(
                    values=shap_row,
                    base_values=base_value,
                    data=input_row_series.to_numpy(),
                    feature_names=clean_feature_names,
                )

                plt.style.use("dark_background")
                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(row_explanation, max_display=10, show=False)
                fig_pred_wf = plt.gcf()
                fig_pred_wf.patch.set_facecolor("none")
                fig_pred_wf.tight_layout()
                st.pyplot(fig_pred_wf, use_container_width=True)
                plt.close(fig_pred_wf)

                top_driver_indices = np.argsort(np.abs(shap_row))[::-1][:3]
                driver_lines = []
                for driver_col_idx in top_driver_indices:
                    raw_feature = input_row_series.index[driver_col_idx]
                    feature_label = FEATURE_META.get(raw_feature, {}).get("label", raw_feature)
                    feature_value = float(input_row_series.iloc[driver_col_idx])
                    direction = "increases risk" if shap_row[driver_col_idx] >= 0 else "decreases risk"

                    if feature_value.is_integer():
                        feature_value_text = f"{int(feature_value)}"
                    elif abs(feature_value) >= 1000:
                        feature_value_text = f"{feature_value:.0f}"
                    else:
                        feature_value_text = f"{feature_value:.3f}".rstrip("0").rstrip(".")

                    driver_lines.append(
                        f"{feature_label}: value={feature_value_text}, {direction}"
                    )

                shap_drivers = "; ".join(driver_lines)
                pd_score = round(pd_prob * 100, 2)

                narrative_cache_key = f"{pd_score}|{shap_drivers}"
                cached_narrative_key = st.session_state.get("predictor_narrative_key")
                cached_narrative_text = st.session_state.get("predictor_narrative")

                if cached_narrative_key == narrative_cache_key and cached_narrative_text:
                    st.info(cached_narrative_text)
                else:
                    nim_api_key = st.secrets.get("NIM_API_KEY", "")
                    if not nim_api_key or nim_api_key == "your_key_here":
                        st.warning("Narrative unavailable: set a valid NIM_API_KEY in .streamlit/secrets.toml")
                    else:
                        with st.spinner("Generating analyst narrative..."):
                            prompt_messages = [
                                {
                                    "role": "system",
                                    "content": "You are a credit risk analyst. Write concise, professional assessments. Never invent data not provided to you.",
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"A loan applicant has a {pd_score}% probability of default. "
                                        f"The top risk drivers are: {shap_drivers}. "
                                        "Write a 3-sentence credit analyst assessment explaining the key risk factors and overall credit profile."
                                    ),
                                },
                            ]

                            try:
                                client = OpenAI(
                                    base_url="https://integrate.api.nvidia.com/v1",
                                    api_key=nim_api_key,
                                )

                                response = client.chat.completions.create(
                                    model="z-ai/glm5",
                                    messages=prompt_messages,
                                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                )
                                narrative_text = response.choices[0].message.content
                                st.session_state["predictor_narrative_key"] = narrative_cache_key
                                st.session_state["predictor_narrative"] = narrative_text
                                st.info(narrative_text)
                            except Exception:
                                try:
                                    client = OpenAI(
                                        base_url="https://integrate.api.nvidia.com/v1",
                                        api_key=nim_api_key,
                                    )
                                    response = client.chat.completions.create(
                                        model="moonshotai/kimi-k2-instruct",
                                        messages=prompt_messages,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                    )
                                    narrative_text = response.choices[0].message.content
                                    st.session_state["predictor_narrative_key"] = narrative_cache_key
                                    st.session_state["predictor_narrative"] = narrative_text
                                    st.info(narrative_text)
                                except Exception as fallback_err:
                                    st.warning("Narrative unavailable")
                                    st.caption(f"NIM error: {type(fallback_err).__name__}: {fallback_err}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INJECT JS — Round the Plotly hover tooltip boxes
#  CSS rx/ry does NOT work on SVG <path> elements. We must use JavaScript
#  to intercept the hover paths and redraw them as rounded rectangles.
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit.components.v1 as components

components.html(
    """
    <script>
    (function() {
        const pd = window.parent.document;

        // Redraw a sharp-cornered SVG path as a rounded rectangle
        function roundify(path, r) {
            try {
                const bb = path.getBBox();
                if (bb.width < 1 || bb.height < 1) return;
                r = Math.min(r, bb.width / 2, bb.height / 2);
                const x = bb.x, y = bb.y, w = bb.width, h = bb.height;
                path.setAttribute('d',
                    'M'+(x+r)+','+y+
                    ' H'+(x+w-r)+
                    ' a'+r+','+r+',0,0,1,'+r+','+r+
                    ' V'+(y+h-r)+
                    ' a'+r+','+r+',0,0,1,-'+r+','+r+
                    ' H'+(x+r)+
                    ' a'+r+','+r+',0,0,1,-'+r+',-'+r+
                    ' V'+(y+r)+
                    ' a'+r+','+r+',0,0,1,'+r+',-'+r+
                    ' Z'
                );
            } catch(e) {}
        }

        const obs = new MutationObserver(function() {
            pd.querySelectorAll('.hoverlayer .hovertext path').forEach(function(p) {
                const d = p.getAttribute('d');
                // Plotly's default rectangular tooltips don't use arc ('a') commands.
                // Our curved tooltips do. If an 'a' is missing, it means Plotly just 
                // redrew a sharp rectangle and we need to round it again.
                if (d && !d.includes('a')) {
                    roundify(p, 10);
                }
            });
        });
        obs.observe(pd.body, {childList: true, subtree: true, attributes: true});
    })();
    </script>
    """,
    height=0,
)



# Thank you for using the Credit Risk Dashboard