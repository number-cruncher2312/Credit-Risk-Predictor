"""Scenario-based input data simulations for drift monitoring demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


SIMULATION_SCENARIOS = {
    "baseline": {
        "label": "Baseline / No Drift",
        "description": "Use the original current dataset without modification.",
    },
    "credit_utilization_shift": {
        "label": "Feature Drift - Credit Utilization Shift",
        "description": "Increase revolving utilization to demonstrate feature-level drift.",
    },
    "population_shift": {
        "label": "Population Shift - Multiple Feature Changes",
        "description": "Shift income, utilization, age, and delinquency signals together.",
    },
}


def _numeric_series(frame, column_name):
    series = pd.to_numeric(frame[column_name], errors="coerce")
    if series.notna().any():
        return series.fillna(series.median()).astype(float)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _set_series(frame, column_name, values):
    if column_name in frame.columns:
        frame[column_name] = values


def _shift_utilization(frame):
    if "RevolvingUtilizationOfUnsecuredLines" not in frame.columns:
        return frame

    utilization = _numeric_series(frame, "RevolvingUtilizationOfUnsecuredLines")
    shifted = (utilization * 1.75 + 0.15).clip(lower=0.0, upper=1.0)
    _set_series(frame, "RevolvingUtilizationOfUnsecuredLines", shifted)
    return frame


def _shift_population(frame):
    if "RevolvingUtilizationOfUnsecuredLines" in frame.columns:
        utilization = _numeric_series(frame, "RevolvingUtilizationOfUnsecuredLines")
        _set_series(
            frame,
            "RevolvingUtilizationOfUnsecuredLines",
            (utilization * 1.6 + 0.12).clip(lower=0.0, upper=1.0),
        )

    if "MonthlyIncome" in frame.columns:
        income = _numeric_series(frame, "MonthlyIncome")
        _set_series(frame, "MonthlyIncome", (income * 0.68).clip(lower=0.0))

    if "DebtRatio" in frame.columns:
        debt_ratio = _numeric_series(frame, "DebtRatio")
        _set_series(frame, "DebtRatio", (debt_ratio * 1.45 + 0.10).clip(lower=0.0, upper=5.0))

    if "age" in frame.columns:
        age = _numeric_series(frame, "age")
        _set_series(frame, "age", (age * 0.92 - 4.0).clip(lower=18.0, upper=95.0))

    delinquency_columns = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for column_name in delinquency_columns:
        if column_name in frame.columns:
            delinquency = _numeric_series(frame, column_name)
            shifted = np.clip(np.rint(delinquency * 1.5 + 1.0), 0, None)
            _set_series(frame, column_name, shifted)

    if "NumberOfOpenCreditLinesAndLoans" in frame.columns:
        open_lines = _numeric_series(frame, "NumberOfOpenCreditLinesAndLoans")
        shifted = np.clip(np.rint(open_lines * 0.85 - 1.0), 0, None)
        _set_series(frame, "NumberOfOpenCreditLinesAndLoans", shifted)

    if "NumberOfDependents" in frame.columns:
        dependents = _numeric_series(frame, "NumberOfDependents")
        shifted = np.clip(np.rint(dependents + 0.5), 0, None)
        _set_series(frame, "NumberOfDependents", shifted)

    return frame


def simulate_drift_dataset(current_data, scenario):
    """Return a modified copy of current_data for the selected simulation scenario."""

    simulated_frame = current_data.copy(deep=True)
    scenario_key = str(scenario or "baseline").strip().lower()

    if scenario_key in {"baseline", "no drift", "no_drift"}:
        return simulated_frame
    if scenario_key in {"credit_utilization_shift", "feature drift - credit utilization shift"}:
        return _shift_utilization(simulated_frame)
    if scenario_key in {"population_shift", "population shift - multiple feature changes"}:
        return _shift_population(simulated_frame)

    raise ValueError(f"Unknown drift simulation scenario: {scenario}")