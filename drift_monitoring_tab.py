"""Streamlit drift monitoring tab visualization components."""

from __future__ import annotations

import pandas as pd
import streamlit as st


MOCK_DRIFT_SNAPSHOT = {
    "status": "Yellow",
    "psi": 0.18,
    "recommended_action": (
        "Review the CSI contributors, monitor the PSI trend daily, and prepare a retraining"
        " assessment if the signal persists."
    ),
}

MOCK_CSI_ROWS = [
    {"feature_name": "Revolving Utilization", "csi_value": 0.04, "status": "Stable"},
    {"feature_name": "Age", "csi_value": 0.03, "status": "Stable"},
    {"feature_name": "Debt Ratio", "csi_value": 0.11, "status": "Watch"},
    {"feature_name": "Monthly Income", "csi_value": 0.16, "status": "Elevated"},
    {"feature_name": "Open Credit Lines", "csi_value": 0.06, "status": "Stable"},
]

MOCK_PSI_HISTORY = [
    {"month": "Jan", "psi": 0.05},
    {"month": "Feb", "psi": 0.07},
    {"month": "Mar", "psi": 0.09},
    {"month": "Apr", "psi": 0.08},
    {"month": "May", "psi": 0.12},
    {"month": "Jun", "psi": 0.18},
]

MOCK_BUCKET_DISTRIBUTIONS = [
    {"bucket": "0.0-0.1", "expected": 0.28, "current": 0.22},
    {"bucket": "0.1-0.2", "expected": 0.24, "current": 0.20},
    {"bucket": "0.2-0.3", "expected": 0.18, "current": 0.19},
    {"bucket": "0.3-0.5", "expected": 0.17, "current": 0.21},
    {"bucket": "0.5+", "expected": 0.13, "current": 0.18},
]

STATUS_THEME = {
    "Green": {"accent": "#34d399", "background": "rgba(52, 211, 153, 0.12)", "label": "On track"},
    "Yellow": {"accent": "#fbbf24", "background": "rgba(251, 191, 36, 0.12)", "label": "Monitor closely"},
    "Red": {"accent": "#fb7185", "background": "rgba(251, 113, 133, 0.12)", "label": "Escalate"},
}


def inject_drift_monitoring_styles():
    """Add lightweight drift-specific styling on top of the shared dashboard theme."""

    st.markdown(
        """
        <style>
        .drift-hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.9fr);
            gap: 16px;
            margin-bottom: 18px;
        }
        .drift-status-card,
        .drift-psi-card,
        .drift-action-card {
            background: linear-gradient(135deg, rgba(30, 30, 47, 0.98) 0%, rgba(42, 42, 64, 0.98) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.24);
            padding: 22px 24px;
        }
        .drift-status-card__eyebrow,
        .drift-psi-card__eyebrow,
        .drift-action-card__eyebrow {
            color: #9d9db5;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .drift-status-card__pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .drift-status-card__pill-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: currentColor;
        }
        .drift-status-card__title {
            color: #f8fafc;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 8px;
        }
        .drift-status-card__subtitle,
        .drift-action-card__text {
            color: #c9ccda;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .drift-psi-card__value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1;
            margin-bottom: 8px;
        }
        .drift-psi-card__meta {
            color: #9d9db5;
            font-size: 0.9rem;
        }
        .drift-action-card {
            margin: 0 0 20px;
            border-left: 4px solid #fbbf24;
        }
        .drift-action-card__text strong {
            color: #f8fafc;
        }
        .drift-distribution-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        @media (max-width: 900px) {
            .drift-hero-grid,
            .drift-distribution-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_csi_summary_frame():
    """Return the mock CSI summary table used by the UI."""

    return pd.DataFrame(MOCK_CSI_ROWS).rename(
        columns={
            "feature_name": "Feature Name",
            "csi_value": "CSI Value",
            "status": "Status",
        }
    )


def build_csi_summary_frame_from_results(drift_results):
    """Return a display-ready CSI summary table from DriftResults."""

    if drift_results is None or not drift_results.csi:
        return build_csi_summary_frame()

    return pd.DataFrame(
        [
            {
                "Feature Name": feature_name,
                "CSI Value": metric.value,
                "Status": metric.status,
            }
            for feature_name, metric in drift_results.csi.items()
        ]
    )


def build_psi_history_frame():
    """Return mock monthly PSI history for the placeholder trend chart."""

    # Placeholder: replace this mock frame with real historical PSI values later.
    return pd.DataFrame(MOCK_PSI_HISTORY).rename(
        columns={
            "month": "Month",
            "psi": "PSI",
        }
    )


def build_psi_history_frame_from_results(psi_history):
    """Return chart-ready persisted PSI history, or mock fallback when unavailable."""

    if psi_history is None or psi_history.empty:
        return build_psi_history_frame()

    return psi_history.rename(
        columns={
            "Date": "Month",
            "PSI": "PSI",
        }
    ).loc[:, ["Month", "PSI"]]


def build_bucket_distribution_frame():
    """Return mock bucket distributions for expected vs current comparison."""

    # Placeholder: replace this mock frame with bucket distributions from the PSI engine later.
    return pd.DataFrame(MOCK_BUCKET_DISTRIBUTIONS).rename(
        columns={
            "bucket": "Bucket",
            "expected": "Expected Distribution",
            "current": "Current Distribution",
        }
    )


def build_bucket_distribution_frame_from_results(drift_results):
    """Return chart-ready PSI bucket distributions from DriftResults."""

    if drift_results is None or not drift_results.psi.expected_distribution:
        return build_bucket_distribution_frame()

    return pd.DataFrame(
        {
            "Bucket": drift_results.psi.bucket_labels,
            "Expected Distribution": drift_results.psi.expected_distribution,
            "Current Distribution": drift_results.psi.actual_distribution,
        }
    )


def style_csi_summary_frame(frame):
    """Apply simple presentation formatting to the mock CSI table."""

    def color_status(value):
        colors = {
            "Stable": "color: #34d399; font-weight: 700;",
            "Watch": "color: #fbbf24; font-weight: 700;",
            "Elevated": "color: #fb7185; font-weight: 700;",
            "Green": "color: #34d399; font-weight: 700;",
            "Yellow": "color: #fbbf24; font-weight: 700;",
            "Red": "color: #fb7185; font-weight: 700;",
        }

        return colors.get(value, "color: #e0e0ec; font-weight: 700;")

    return (
        frame.style.format({"CSI Value": "{:.3f}"})
        .applymap(color_status, subset=["Status"])
        .set_properties(**{"background-color": "rgba(255,255,255,0.015)", "color": "#e0e0ec"})
    )


def render_placeholder_panel(title, body):
    """Render a consistent empty-state panel for future monitoring visuals."""

    st.markdown(
        f"""
        <div class="placeholder-box">
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(status, psi_value):
    """Render the top-level monitoring status indicator."""

    theme = STATUS_THEME[status]
    st.markdown(
        f"""
        <div class="drift-status-card">
            <div class="drift-status-card__eyebrow">Monitoring Status</div>
            <div class="drift-status-card__pill" style="color: {theme['accent']}; background: {theme['background']};">
                <span class="drift-status-card__pill-dot"></span>
                {status} - {theme['label']}
            </div>
            <div class="drift-status-card__title">Current drift monitoring snapshot</div>
            <div class="drift-status-card__subtitle">
                PSI and CSI are calculated outside Streamlit, then passed into this visualization tab.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_psi_card(psi_value):
    """Render the current PSI value display."""

    st.markdown(
        f"""
        <div class="drift-psi-card">
            <div class="drift-psi-card__eyebrow">Current PSI</div>
            <div class="drift-psi-card__value">{psi_value:.3f}</div>
            <div class="drift-psi-card__meta">Score-level Population Stability Index.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_psi_trend_chart(psi_history=None):
    """Render the PSI trend line chart."""

    psi_history = build_psi_history_frame_from_results(psi_history).set_index("Month")
    st.line_chart(
        psi_history,
        y="PSI",
        use_container_width=True,
    )


def render_distribution_comparison_chart(drift_results=None):
    """Render the expected vs current PSI bucket distribution comparison."""

    bucket_distribution = build_bucket_distribution_frame_from_results(drift_results).set_index("Bucket")
    st.bar_chart(
        bucket_distribution,
        use_container_width=True,
    )


def render_drift_monitoring_tab(
    drift_results=None,
    monitoring_status=None,
    recommended_action=None,
    psi_history=None,
    simulation_mode=False,
    simulation_scenario_label=None,
):
    """Render the Streamlit drift monitoring tab using precomputed drift results."""

    inject_drift_monitoring_styles()

    st.markdown(
        '<p class="section-header">Drift Monitoring</p>'
        '<p class="section-sub">PSI and CSI monitoring for model stability review</p>',
        unsafe_allow_html=True,
    )

    if simulation_mode:
        scenario_text = simulation_scenario_label or "Unknown scenario"
        st.info(f"Simulation Mode Active - Scenario: {scenario_text}")

    status = monitoring_status or MOCK_DRIFT_SNAPSHOT["status"]
    psi_value = drift_results.psi.value if drift_results is not None else MOCK_DRIFT_SNAPSHOT["psi"]
    action_text = recommended_action or MOCK_DRIFT_SNAPSHOT["recommended_action"]

    hero_left, hero_right = st.columns([1.4, 0.9])
    with hero_left:
        render_status_card(status, psi_value)
    with hero_right:
        render_psi_card(psi_value)

    st.markdown(
        f"""
            <div class="drift-action-card">
                <div class="drift-action-card__eyebrow">Recommended Action</div>
            <div class="drift-action-card__text"><strong>{status}:</strong> {action_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-header">CSI Summary</p>'
        '<p class="section-sub">Feature-level stability view for monitored important features</p>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        style_csi_summary_frame(build_csi_summary_frame_from_results(drift_results)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<p class="section-header">PSI Trend Over Time</p>'
        '<p class="section-sub">Persisted score PSI values over time</p>',
        unsafe_allow_html=True,
    )
    render_psi_trend_chart(psi_history)

    st.markdown(
        '<p class="section-header">Expected vs Actual Distribution</p>'
        '<p class="section-sub">Reference score distribution compared with the current score distribution</p>',
        unsafe_allow_html=True,
    )
    render_distribution_comparison_chart(drift_results)
