import pytest

from reg_capital import compute_single_loan


def test_basel_reference_case():
    """
    Regression test for the Basel IRB implementation.

    If this test ever fails after future code changes,
    investigate whether the Basel formulas were modified
    intentionally or a bug was introduced.
    """

    loan = compute_single_loan(
        pd=0.02,
        lgd=0.45,
        ead=100000,
        maturity=2.5,
    )

    assert loan["basel_asset_correlation"] == pytest.approx(
        0.16414553294057307,
        abs=1e-10,
    )

    assert loan["basel_maturity_adjustment"] == pytest.approx(
        0.11076956525517692,
        abs=1e-10,
    )

    assert loan["basel_k"] == pytest.approx(
        0.09188338300659993,
        abs=1e-10,
    )

    assert loan["basel_capital"] == pytest.approx(
        9188.338300659992,
        abs=1e-6,
    )

    assert loan["basel_rwa"] == pytest.approx(
        114854.22875824991,
        abs=1e-6,
    )