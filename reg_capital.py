"""Basel IRB regulatory capital helpers.

This module is backend-only. It does not import Streamlit or any UI code.
Its job is to implement the Basel corporate IRB calculation chain in small,
testable functions that are easy to explain in an interview.

Basel sources referenced in comments below:
- CRE31: IRB approach: risk-weight functions
  https://www.bis.org/basel_framework/chapter/CRE/31.htm
- CRE32: IRB approach: risk components
  https://www.bis.org/basel_framework/chapter/CRE/32.htm
"""

from __future__ import annotations

from math import exp, isfinite, log, sqrt
from typing import Any

import pandas as pd
from scipy.stats import norm


BASEL_PD_FLOOR = 0.0005
BASEL_PD_CEILING = 1.0 - 1e-12
BASEL_CONFIDENCE_LEVEL = 0.999
BASEL_MATURITY_FLOOR = 1.0
BASEL_MATURITY_CEILING = 5.0
BASEL_RWA_MULTIPLIER = 12.5
BASEL_REFERENCE_MATURITY = 2.5


def _validate_probability(value: float, name: str) -> None:
    """Reject non-finite or out-of-range probabilities early."""

    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number.")

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive.")


def _effective_pd_for_basel(pd_value: float) -> float:
    """Clip PD into the Basel-compliant input range.

    Basel CRE32 requires corporate PD inputs used in the risk-weight formula
    to be no lower than 0.05%. Clipping also keeps inverse-normal and log
    calculations well-defined at the boundaries.
    """

    _validate_probability(pd_value, "pd")
    return min(max(pd_value, BASEL_PD_FLOOR), BASEL_PD_CEILING)


def _effective_maturity_for_basel(maturity: float) -> float:
    """Clamp maturity to the Basel corporate IRB range of 1 to 5 years."""

    if not isfinite(maturity):
        raise ValueError("maturity must be a finite number.")

    if maturity <= 0.0:
        raise ValueError("maturity must be greater than 0.")

    return min(max(maturity, BASEL_MATURITY_FLOOR), BASEL_MATURITY_CEILING)


def compute_asset_correlation(pd: float) -> float:
    """Compute the Basel corporate IRB asset correlation R.

    Why this exists:
        R is the systematic-risk correlation term in the Basel risk-weight
        function. It should be isolated so the Basel formula can be tested and
        explained without mixing it with maturity or exposure scaling.

    Inputs:
        pd: One-year probability of default for the corporate borrower.

    Outputs:
        The Basel asset correlation R as a decimal.

    Why this follows Basel:
        CRE31 defines the corporate risk-weight function and CRE32 defines PD.
        Basel also requires PD inputs used in the formula to respect the 0.05%
        floor, so the implementation clips the input before applying the formula.
    """

    effective_pd = _effective_pd_for_basel(pd)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # Basel Framework: CRE32 - IRB approach: risk components
    # https://www.bis.org/basel_framework/chapter/CRE/32.htm
    # The term below scales PD into the 0% to 100% range used by the Basel
    # corporate correlation curve.
    scaling_term = (1.0 - exp(-50.0 * effective_pd)) / (1.0 - exp(-50.0))

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # The correlation interpolates between 24% at very low PDs and 12% at
    # higher PDs, which is the corporate IRB parameterisation in Basel.
    asset_correlation = 0.12 * scaling_term + 0.24 * (1.0 - scaling_term)

    return asset_correlation


def compute_maturity_adjustment(pd: float) -> float:
    """Compute the Basel maturity adjustment parameter b.

    Why this exists:
        b is the PD-dependent term used in the maturity adjustment. Keeping it
        separate makes the Basel maturity logic easier to inspect and reuse.

    Inputs:
        pd: One-year probability of default for the corporate borrower.

    Outputs:
        The maturity adjustment term b.

    Why this follows Basel:
        CRE31 and CRE32 together define the corporate IRB risk-weight formula.
        The PD input is clipped to the Basel-compliant floor before the natural
        logarithm is applied so the formula stays numerically valid.
    """

    effective_pd = _effective_pd_for_basel(pd)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # Basel Framework: CRE32 - IRB approach: risk components
    # https://www.bis.org/basel_framework/chapter/CRE/32.htm
    # b is the PD-driven maturity sensitivity term used in the Basel formula.
    maturity_adjustment = (0.11852 - 0.05478 * log(effective_pd)) ** 2

    return maturity_adjustment


def compute_capital_requirement(pd: float, lgd: float, maturity: float) -> float:
    """Compute the Basel capital requirement K for one corporate loan.

    Why this exists:
        K is the unexpected-loss capital charge per unit of EAD. It depends on
        PD, LGD, maturity, and correlation, so it belongs in a dedicated
        function instead of being spread across the portfolio workflow.

    Inputs:
        pd: One-year probability of default.
        lgd: Loss given default as a decimal between 0 and 1.
        maturity: Effective maturity in years.

    Outputs:
        K, the Basel capital requirement per unit of EAD.

    Why this follows Basel:
        CRE31 defines the corporate IRB risk-weight function. The 99.9%
        confidence level is hardcoded because Basel fixes that stress level.
    """

    _validate_probability(lgd, "lgd")

    if not isfinite(maturity):
        raise ValueError("maturity must be a finite number.")

    if maturity <= 0.0:
        raise ValueError("maturity must be greater than 0.")

    effective_pd = _effective_pd_for_basel(pd)
    effective_maturity = _effective_maturity_for_basel(maturity)
    asset_correlation = compute_asset_correlation(effective_pd)
    maturity_adjustment = compute_maturity_adjustment(effective_pd)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # CRE32 - IRB approach: risk components
    # https://www.bis.org/basel_framework/chapter/CRE/32.htm
    # G(PD) converts the probability of default into a standard-normal score.
    inverse_default_quantile = norm.ppf(effective_pd)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # The 99.9% quantile is fixed by Basel and represents the capital stress
    # level for corporate IRB exposures.
    stress_quantile = norm.ppf(BASEL_CONFIDENCE_LEVEL)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # This is the exact Basel one-factor argument passed into the normal CDF.
    threshold = (
        inverse_default_quantile / sqrt(1.0 - asset_correlation)
        + sqrt(asset_correlation / (1.0 - asset_correlation)) * stress_quantile
    )

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # N(threshold) is the stressed default probability implied by the model.
    stressed_default_probability = norm.cdf(threshold)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # LGD scales the loss severity and PD*LGD removes expected loss.
    capital_before_maturity = lgd * stressed_default_probability - effective_pd * lgd

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # The maturity adjustment increases capital for longer-dated exposures and
    # reduces it for shorter-dated exposures, relative to the 2.5-year base.
    maturity_multiplier = (
        1.0 + (effective_maturity - BASEL_REFERENCE_MATURITY) * maturity_adjustment
    ) / (1.0 - 1.5 * maturity_adjustment)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # K remains a per-unit-of-EAD quantity until it is scaled by exposure.
    capital_requirement = capital_before_maturity * maturity_multiplier

    return capital_requirement


def compute_rwa(k: float, ead: float) -> float:
    """Convert Basel capital requirement K into risk-weighted assets.

    Why this exists:
        Basel reports RWA because it is the risk-weighted exposure measure used
        in regulatory capital ratios. K is the internal capital charge; RWA is
        the reporting output.

    Inputs:
        k: Basel capital requirement per unit of EAD.
        ead: Exposure at default in currency units.

    Outputs:
        Risk-weighted assets computed as K × 12.5 × EAD.

    Why this follows Basel:
        CRE31 shows that RWA is derived from K and EAD, and the 12.5 factor is
        the inverse of the 8% minimum capital ratio.
    """

    if not isfinite(k):
        raise ValueError("k must be a finite number.")

    if not isfinite(ead):
        raise ValueError("ead must be a finite number.")

    if ead < 0.0:
        raise ValueError("ead must be greater than or equal to 0.")

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # RWA is the capital requirement scaled by the exposure and the Basel 12.5
    # multiplier.
    risk_weighted_assets = k * BASEL_RWA_MULTIPLIER * ead

    return risk_weighted_assets


def compute_single_loan(
    *,
    pd: float,
    lgd: float,
    ead: float,
    maturity: float,
    loan_id: Any | None = None,
) -> dict[str, Any]:
    """Compute all Basel capital metrics for one loan.

    Why this exists:
        This function is the loan-level orchestration layer. It ties together
        the Basel inputs and returns every intermediate result needed for audit
       ability, debugging, and later dashboard display.

    Inputs:
        pd: One-year probability of default.
        lgd: Loss given default.
        ead: Exposure at default.
        maturity: Effective maturity.
        loan_id: Optional loan identifier.

    Outputs:
        A dictionary containing the input values plus R, b, K, Capital, and
        RWA.

    Why this follows Basel:
        The function uses the Basel corporate IRB formulas directly and keeps
        the calculation sequence explicit.
    """

    effective_pd = _effective_pd_for_basel(pd)
    effective_maturity = _effective_maturity_for_basel(maturity)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # R is the PD-dependent correlation parameter used inside the corporate IRB
    # risk-weight function.
    asset_correlation = compute_asset_correlation(effective_pd)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # b is the PD-dependent maturity sensitivity term.
    maturity_adjustment = compute_maturity_adjustment(effective_pd)

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # K is the Basel capital requirement per unit of EAD.
    capital_requirement = compute_capital_requirement(
        pd=effective_pd,
        lgd=lgd,
        maturity=effective_maturity,
    )

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # Capital converts K into a monetary amount by multiplying by EAD.
    capital_amount = capital_requirement * ead

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # RWA scales the capital charge by 12.5 and EAD for regulatory reporting.
    risk_weighted_assets = compute_rwa(capital_requirement, ead)

    return {
        "loan_id": loan_id,
        "pd": pd,
        "basel_pd_effective": effective_pd,
        "lgd": lgd,
        "ead": ead,
        "maturity": maturity,
        "basel_maturity_effective": effective_maturity,
        "basel_asset_correlation": asset_correlation,
        "basel_maturity_adjustment": maturity_adjustment,
        "basel_k": capital_requirement,
        "basel_capital": capital_amount,
        "basel_rwa": risk_weighted_assets,
    }


def compute_portfolio(loans: pd.DataFrame) -> pd.DataFrame:
    """Compute Basel capital metrics for a portfolio of loans.

    Why this exists:
        Portfolio users need a table with row-level Basel outputs and portfolio
        totals. This function keeps the loop out of the UI and returns a fully
        annotated DataFrame.

    Inputs:
        loans: A pandas DataFrame containing at least PD, LGD, EAD, and either
            maturity or maturity_years.

    Outputs:
        A copy of the input DataFrame with Basel columns added, plus portfolio
        totals stored as columns and DataFrame attributes.

    Why this follows Basel:
        Basel capital is computed at the loan level and then aggregated by
        summing RWA across exposures. Averaging PD first would change the math
        and break the nonlinear Basel formulas.
    """

    if not isinstance(loans, pd.DataFrame):
        raise TypeError("loans must be a pandas DataFrame.")

    updated_loans = loans.copy()

    required_columns = {"pd", "lgd", "ead"}
    missing_columns = required_columns.difference(updated_loans.columns)
    if missing_columns:
        raise KeyError(f"Missing required Basel input column(s): {sorted(missing_columns)}")

    if "maturity" in updated_loans.columns:
        maturity_column = "maturity"
    elif "maturity_years" in updated_loans.columns:
        maturity_column = "maturity_years"
    else:
        raise KeyError("Missing required Basel input column: maturity or maturity_years")

    row_results: list[dict[str, Any]] = []
    for row in updated_loans.itertuples(index=False):
        loan_id = getattr(row, "loan_id", None)
        row_results.append(
            compute_single_loan(
                pd=row.pd,
                lgd=row.lgd,
                ead=row.ead,
                maturity=getattr(row, maturity_column),
                loan_id=loan_id,
            )
        )

    metrics_frame = pd.DataFrame(row_results, index=updated_loans.index)

    # compute_single_loan() returns the original inputs alongside the Basel
    # outputs for traceability. The portfolio table already has the inputs, so
    # we only join the new Basel columns here to avoid column overlap.
    basel_metric_columns = [column for column in metrics_frame.columns if column.startswith("basel_")]
    updated_loans = updated_loans.join(metrics_frame[basel_metric_columns])

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # Portfolio RWA is the sum of the loan-level RWAs.
    total_portfolio_rwa = updated_loans["basel_rwa"].sum()

    # Basel Framework: CRE31 - IRB approach: risk-weight functions
    # https://www.bis.org/basel_framework/chapter/CRE/31.htm
    # Portfolio capital is the sum of the monetary capital amounts.
    total_regulatory_capital = updated_loans["basel_capital"].sum()

    updated_loans["basel_total_portfolio_rwa"] = total_portfolio_rwa
    updated_loans["basel_total_regulatory_capital"] = total_regulatory_capital
    updated_loans.attrs["total_portfolio_rwa"] = total_portfolio_rwa
    updated_loans.attrs["total_regulatory_capital"] = total_regulatory_capital

    return updated_loans


# Backward-compatible aliases for earlier versions of this module.
compute_basel_asset_correlation_from_pd = compute_asset_correlation
compute_basel_capital_requirement = compute_capital_requirement
compute_rwa_from_k_and_ead = compute_rwa
compute_regulatory_capital_for_loan = compute_single_loan
compute_regulatory_capital_for_portfolio = compute_portfolio