from types import SimpleNamespace

import pandas as pd

from drift import compute_drift_metrics, determine_monitoring_status
from drift_simulation import simulate_drift_dataset
from models import DriftResults


def test_determine_monitoring_status_green_yellow_red_boundaries():
    config = {
        "psi": {
            "warning_threshold": 0.10,
            "critical_threshold": 0.25,
        }
    }

    green_results = SimpleNamespace(psi=SimpleNamespace(value=0.05))
    yellow_results = SimpleNamespace(psi=SimpleNamespace(value=0.10))
    red_results = SimpleNamespace(psi=SimpleNamespace(value=0.25))

    assert determine_monitoring_status(green_results, config) == (
        "Green",
        "Continue monitoring.",
    )
    assert determine_monitoring_status(yellow_results, config) == (
        "Yellow",
        "Investigate drift and review CSI.",
    )
    assert determine_monitoring_status(red_results, config) == (
        "Red",
        "Escalate for model risk review and assess whether retraining is warranted after reviewing additional evidence.",
    )


def test_determine_monitoring_status_does_not_mutate_results():
    config = {
        "psi": {
            "warning_threshold": 0.10,
            "critical_threshold": 0.25,
        }
    }
    drift_results = SimpleNamespace(psi=SimpleNamespace(value=0.15))

    original_psi_value = drift_results.psi.value

    determine_monitoring_status(drift_results, config)

    assert drift_results.psi.value == original_psi_value


class FakeModel:
    feature_names_in_ = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "MonthlyIncome",
        "DebtRatio",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfDependents",
    ]

    def predict_proba(self, frame):
        score = (
            frame["RevolvingUtilizationOfUnsecuredLines"].astype(float) * 0.40
            + frame["DebtRatio"].astype(float) * 0.18
            + frame["NumberOfTime30-59DaysPastDueNotWorse"].astype(float) * 0.07
            + frame["NumberOfTime60-89DaysPastDueNotWorse"].astype(float) * 0.05
            + frame["NumberOfTimes90DaysLate"].astype(float) * 0.08
            + (1.0 - (frame["MonthlyIncome"].astype(float) / 10000.0).clip(0, 1)) * 0.14
            + (1.0 - (frame["age"].astype(float) / 100.0).clip(0, 1)) * 0.08
            + (1.0 - (frame["NumberOfOpenCreditLinesAndLoans"].astype(float) / 20.0).clip(0, 1)) * 0.05
            + (frame["NumberOfDependents"].astype(float) / 5.0).clip(0, 1) * 0.03
        ).clip(0, 1)
        return pd.DataFrame({0: 1 - score, 1: score}).to_numpy()


def build_base_frame():
    return pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
            "age": [60, 58, 55, 63, 57, 61],
            "MonthlyIncome": [6200, 6400, 6600, 6800, 7000, 7200],
            "DebtRatio": [0.22, 0.24, 0.21, 0.25, 0.23, 0.20],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 0, 0, 1, 0, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0, 0, 0, 0, 0],
            "NumberOfTimes90DaysLate": [0, 0, 0, 0, 0, 0],
            "NumberOfOpenCreditLinesAndLoans": [10, 11, 12, 9, 10, 11],
            "NumberOfDependents": [0, 1, 1, 0, 2, 1],
        }
    )


def test_compute_drift_metrics_returns_real_psi_csi_and_buckets():
    reference = build_base_frame()
    current = build_base_frame()
    config = {
        "psi": {"warning_threshold": 0.10, "critical_threshold": 0.25},
        "csi": {
            "enabled": True,
            "warning_threshold": 0.10,
            "critical_threshold": 0.25,
        },
    }

    results = compute_drift_metrics(
        current_data=current,
        reference_data=reference,
        monitored_features=["RevolvingUtilizationOfUnsecuredLines"],
        config=config,
        model=FakeModel(),
    )

    assert isinstance(results, DriftResults)
    assert results.psi.value >= 0
    assert results.psi.expected_distribution
    assert results.psi.actual_distribution
    assert set(results.csi) == {"RevolvingUtilizationOfUnsecuredLines"}
    assert results.csi["RevolvingUtilizationOfUnsecuredLines"].value >= 0


def test_simulation_baseline_keeps_current_data_unchanged():
    current = build_base_frame()

    simulated = simulate_drift_dataset(current, "baseline")

    pd.testing.assert_frame_equal(simulated, current)


def test_credit_utilization_shift_increases_feature_csi():
    reference = build_base_frame()
    current = build_base_frame()
    simulated = simulate_drift_dataset(current, "credit_utilization_shift")
    config = {
        "psi": {"warning_threshold": 0.01, "critical_threshold": 0.05},
        "csi": {
            "enabled": True,
            "warning_threshold": 0.01,
            "critical_threshold": 0.05,
        },
    }

    baseline_results = compute_drift_metrics(
        current_data=current,
        reference_data=reference,
        monitored_features=["RevolvingUtilizationOfUnsecuredLines"],
        config=config,
        model=FakeModel(),
    )
    simulated_results = compute_drift_metrics(
        current_data=simulated,
        reference_data=reference,
        monitored_features=["RevolvingUtilizationOfUnsecuredLines"],
        config=config,
        model=FakeModel(),
    )

    assert simulated_results.csi["RevolvingUtilizationOfUnsecuredLines"].value > baseline_results.csi[
        "RevolvingUtilizationOfUnsecuredLines"
    ].value


def test_population_shift_increases_psi_and_changes_status():
    reference = build_base_frame()
    current = build_base_frame()
    simulated = simulate_drift_dataset(current, "population_shift")
    config = {
        "psi": {"warning_threshold": 0.01, "critical_threshold": 0.05},
        "csi": {
            "enabled": True,
            "warning_threshold": 0.01,
            "critical_threshold": 0.05,
        },
    }

    baseline_results = compute_drift_metrics(
        current_data=current,
        reference_data=reference,
        monitored_features=["RevolvingUtilizationOfUnsecuredLines"],
        config=config,
        model=FakeModel(),
    )
    simulated_results = compute_drift_metrics(
        current_data=simulated,
        reference_data=reference,
        monitored_features=["RevolvingUtilizationOfUnsecuredLines"],
        config=config,
        model=FakeModel(),
    )
    simulated_status, simulated_action = determine_monitoring_status(simulated_results, config)

    assert simulated_results.psi.value > baseline_results.psi.value
    assert simulated_status in {"Yellow", "Red"}
    assert simulated_action != "Continue monitoring."
