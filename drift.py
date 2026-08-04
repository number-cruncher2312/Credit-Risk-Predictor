
import math
import os
from datetime import date

import numpy as np
import pandas as pd

from models import DriftMetric, DriftResults


def get_metric_status(value, config_section):
    """Return Green, Yellow, or Red for a metric value using threshold config."""

    warning_threshold = float(config_section.get("warning_threshold", 0.10))
    critical_threshold = float(config_section.get("critical_threshold", 0.25))

    if value >= critical_threshold:
        return "Red"
    if value >= warning_threshold:
        return "Yellow"
    return "Green"


def determine_monitoring_status(drift_results, monitoring_config):
    """Map PSI drift results to a traffic-light status and action."""

    if drift_results is None or not hasattr(drift_results, "psi"):
        raise ValueError("drift_results must provide a psi metric")

    psi_metric = drift_results.psi
    psi_value = getattr(psi_metric, "value", psi_metric)

    try:
        psi_value = float(psi_value)
    except (TypeError, ValueError) as error:
        raise ValueError("drift_results.psi must be numeric") from error

    psi_config = (monitoring_config or {}).get("psi", {})
    warning_threshold = psi_config.get("warning_threshold")
    critical_threshold = psi_config.get("critical_threshold")

    if warning_threshold is None or critical_threshold is None:
        raise ValueError("monitoring_config must define psi warning and critical thresholds")

    warning_threshold = float(warning_threshold)
    critical_threshold = float(critical_threshold)

    if warning_threshold < 0 or critical_threshold < 0:
        raise ValueError("PSI thresholds must be non-negative")
    if warning_threshold > critical_threshold:
        raise ValueError("warning_threshold must be less than or equal to critical_threshold")

    if psi_value >= critical_threshold:
        return (
            "Red",
            "Escalate for model risk review and assess whether retraining is warranted after reviewing additional evidence.",
        )
    elif psi_value >= warning_threshold:
        return (
            "Yellow",
            "Investigate drift and review CSI.",
        )
    else:
        return (
            "Green",
            "Continue monitoring.",
        )


def build_bucket_proportions(reference_scores, current_scores, n_buckets=10):
    """Bin raw scores using reference-derived bucket edges and return proportions."""
    reference_series = pd.Series(reference_scores, dtype="float64").dropna()
    current_series = pd.Series(current_scores, dtype="float64").dropna()

    if reference_series.empty:
        raise ValueError("reference_scores must contain at least one numeric value")
    if current_series.empty:
        raise ValueError("current_scores must contain at least one numeric value")

    _, bucket_edges = pd.qcut(reference_series, q=n_buckets, retbins=True, duplicates="drop")
    if len(bucket_edges) < 2:
        raise ValueError("reference_scores must contain at least two distinct values")

    bucket_edges[0] = float("-inf")
    bucket_edges[-1] = float("inf")

    reference_bins = pd.cut(reference_series, bins=bucket_edges, include_lowest=True)
    current_bins = pd.cut(current_series, bins=bucket_edges, include_lowest=True)
    bucket_categories = reference_bins.cat.categories

    expected = (
        reference_bins.value_counts(sort=False)
        .reindex(bucket_categories, fill_value=0)
        .div(len(reference_series))
        .tolist()
    )
    actual = (
        current_bins.value_counts(sort=False)
        .reindex(bucket_categories, fill_value=0)
        .div(len(current_series))
        .tolist()
    )

    return expected, actual


def build_bucket_distribution_frame(expected, actual, bucket_labels):
    """Return chart-ready expected/current bucket proportions."""

    return pd.DataFrame(
        {
            "Bucket": bucket_labels,
            "Expected Distribution": expected,
            "Current Distribution": actual,
        }
    )


def calculate_psi(expected, actual):
    """Calculate the Population Stability Index for two proportional buckets."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same length")

    epsilon = 1e-6
    psi_total = 0.0

    for expected_value, actual_value in zip(expected, actual):
        expected_safe = max(expected_value, epsilon)
        actual_safe = max(actual_value, epsilon)
        psi_total += (actual_safe - expected_safe) * math.log(actual_safe / expected_safe)

    return float(psi_total)


def select_monitoring_features(reference_data, model, explainer, config):
    """Select the most important features for CSI using SHAP importance when available."""

    csi_config = (config or {}).get("csi", {})
    top_n = int(csi_config.get("top_n_features", 5))
    model_features = list(getattr(model, "feature_names_in_", reference_data.columns))
    aligned_reference = reference_data.loc[:, model_features]

    if explainer is not None:
        try:
            shap_sample = aligned_reference.head(min(len(aligned_reference), 500))
            shap_values = explainer.shap_values(shap_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            mean_abs_shap = np.abs(np.asarray(shap_values, dtype=float)).mean(axis=0)
            ranked_indices = np.argsort(mean_abs_shap)[::-1]
            return [model_features[index] for index in ranked_indices[:top_n]]
        except Exception:
            pass

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        ranked_indices = np.argsort(np.asarray(importances, dtype=float))[::-1]
        return [model_features[index] for index in ranked_indices[:top_n]]

    return model_features[:top_n]


def calculate_feature_csi(reference_data, current_data, feature_name, config):
    """Calculate CSI for one monitored feature using PSI-style bucket comparison."""

    csi_config = (config or {}).get("csi", {})
    expected, actual = build_bucket_proportions(
        reference_data[feature_name],
        current_data[feature_name],
        n_buckets=int(csi_config.get("n_buckets", 10)),
    )
    csi_value = calculate_psi(expected, actual)
    return DriftMetric(
        name=feature_name,
        value=csi_value,
        status=get_metric_status(csi_value, csi_config),
        expected_distribution=expected,
        actual_distribution=actual,
        bucket_labels=[f"Bucket {index + 1}" for index in range(len(expected))],
    )


def compute_drift_metrics(
    current_data,
    reference_data,
    monitored_features,
    config,
    model=None,
):
    """Compute score-level PSI and feature-level CSI metrics outside the UI."""

    psi_config = (config or {}).get("psi", {})
    reference_frame = reference_data.copy()
    current_frame = current_data.copy()

    if model is not None:
        model_features = list(getattr(model, "feature_names_in_", reference_frame.columns))
        reference_scores = model.predict_proba(reference_frame.loc[:, model_features])[:, 1]
        current_scores = model.predict_proba(current_frame.loc[:, model_features])[:, 1]
    else:
        reference_scores = reference_frame.squeeze()
        current_scores = current_frame.squeeze()

    expected, actual = build_bucket_proportions(
        reference_scores,
        current_scores,
        n_buckets=int(psi_config.get("n_buckets", 10)),
    )
    psi_value = calculate_psi(expected, actual)
    psi_metric = DriftMetric(
        name="Population Stability Index",
        value=psi_value,
        status=get_metric_status(psi_value, psi_config),
        expected_distribution=expected,
        actual_distribution=actual,
        bucket_labels=[f"Bucket {index + 1}" for index in range(len(expected))],
    )

    csi_metrics = {}
    if (config or {}).get("csi", {}).get("enabled", False):
        for feature_name in monitored_features:
            if feature_name in reference_frame.columns and feature_name in current_frame.columns:
                csi_metrics[feature_name] = calculate_feature_csi(
                    reference_frame,
                    current_frame,
                    feature_name,
                    config,
                )

    return DriftResults(psi=psi_metric, csi=csi_metrics)


def resolve_psi_history_path(base_dir, config):
    """Resolve the configured PSI history CSV path for local persistence."""

    history_path = (config or {}).get("psi", {}).get("history_path", "data/psi_history.csv")
    if os.path.isabs(history_path):
        return history_path
    return os.path.join(base_dir, history_path)


def record_psi_history(psi_value, history_path, run_date=None):
    """Persist one PSI value per date in a lightweight CSV history file."""

    run_date = run_date or date.today().isoformat()
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    if os.path.exists(history_path):
        history = pd.read_csv(history_path)
    else:
        history = pd.DataFrame(columns=["Date", "PSI"])

    history = history[history["Date"].astype(str) != str(run_date)]
    history = pd.concat(
        [history, pd.DataFrame([{"Date": run_date, "PSI": float(psi_value)}])],
        ignore_index=True,
    )
    history["Date"] = pd.to_datetime(history["Date"]).dt.date.astype(str)
    history = history.sort_values("Date")
    history.to_csv(history_path, index=False)
    return history


def load_psi_history(history_path):
    """Load persisted PSI history as a chart-ready dataframe."""

    if not os.path.exists(history_path):
        return pd.DataFrame(columns=["Date", "PSI"])
    history = pd.read_csv(history_path)
    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"]).dt.date.astype(str)
    return history
