"""
RiskLens – unit tests for stress revaluation.
Run with: pytest tests/test_revaluation.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.modules.stress_test.portfolio import generate_portfolio
from src.modules.stress_test.revaluation import (
    revalue_bonds, revalue_loans, revalue_derivatives,
    revalue_equities, run_stress_test, reverse_stress_test,
)
from src.modules.stress_test.scenarios import ShockVector, get_scenario, scale_scenario


@pytest.fixture(scope="module")
def portfolio():
    return generate_portfolio(save=False)


@pytest.fixture
def zero_shock():
    return ShockVector()


@pytest.fixture
def credit_shock():
    return get_scenario("Credit Event", impact_score=8.0)


def test_zero_shock_no_change(portfolio, zero_shock):
    """With a zero shock, stressed value must equal mark-to-market."""
    result = run_stress_test(portfolio, zero_shock)
    assert abs(result["total_pnl"]) < 1e-3, "Zero shock should produce zero P&L"


def test_negative_credit_event(portfolio, credit_shock):
    """A credit event shock should produce a negative P&L (loss)."""
    result = run_stress_test(portfolio, credit_shock)
    assert result["total_pnl"] < 0, "Credit event should cause portfolio loss"


def test_pnl_pct_reasonable(portfolio, credit_shock):
    """P&L % should be within plausible bounds (not > 100% loss)."""
    result = run_stress_test(portfolio, credit_shock)
    assert -50.0 <= result["pnl_pct"] <= 50.0


def test_result_has_required_keys(portfolio, credit_shock):
    """Result dict must contain all required keys."""
    result = run_stress_test(portfolio, credit_shock)
    for key in ["before_value", "after_value", "total_pnl", "pnl_pct",
                "by_asset_class", "by_sector", "by_country", "top_losers"]:
        assert key in result


def test_top_losers_length(portfolio, credit_shock):
    """Top losers list must have at most 10 entries."""
    result = run_stress_test(portfolio, credit_shock)
    assert len(result["top_losers"]) <= 10


def test_scale_scenario_at_floor_is_zero():
    """At impact score = floor (5.0), shocks should be zero."""
    base = get_scenario("Geopolitical", impact_score=10.0)
    scaled = scale_scenario(base, impact_score=5.0, floor=5.0)
    assert scaled.equity_shock == pytest.approx(0.0)
    assert scaled.rate_shock_bps == pytest.approx(0.0)


def test_scale_scenario_increases_with_impact():
    """Higher impact score → larger shocks."""
    base = get_scenario("Credit Event", impact_score=10.0)
    s7 = scale_scenario(base, 7.0)
    s9 = scale_scenario(base, 9.0)
    assert abs(s9.equity_shock) > abs(s7.equity_shock)


def test_bond_revaluation_rate_rise(portfolio):
    """Rising rates should decrease bond values."""
    bonds = portfolio[portfolio["asset_class"] == "Bond"].copy()
    if bonds.empty:
        pytest.skip("No bonds in portfolio")
    shock = ShockVector(rate_shock_bps=100.0)
    result = revalue_bonds(bonds, shock)
    total_pnl = result["pnl"].sum()
    assert total_pnl < 0, "Rising rates should hurt bond portfolio"


def test_equity_beta_impact(portfolio):
    """Higher beta equities should lose more in an equity shock."""
    equities = portfolio[portfolio["asset_class"] == "Equity"].copy()
    if len(equities) < 2:
        pytest.skip("Need at least 2 equity positions")
    shock = ShockVector(equity_shock=-0.10)
    result = revalue_equities(equities, shock)
    # Correlation: pnl / market_value should be proportional to -beta
    result["pnl_per_mv"] = result["pnl"] / result["market_value"].replace(0, np.nan)
    corr = result[["beta", "pnl_per_mv"]].dropna().corr().iloc[0, 1]
    assert corr < 0, "Higher beta should correlate with more negative P&L in equity downturn"


def test_portfolio_has_all_asset_classes(portfolio):
    """Portfolio must contain all four asset classes."""
    classes = set(portfolio["asset_class"].unique())
    expected = {"Bond", "Corporate Loan", "IR Swap", "Equity"}
    assert expected.issubset(classes), f"Missing asset classes: {expected - classes}"


def test_portfolio_is_labeled_synthetic(portfolio):
    """All positions must be labeled synthetic=True."""
    assert portfolio["synthetic"].all(), "All positions must be labeled synthetic"


def test_sector_targeted_larger_loss_in_target_sector(portfolio):
    """Sector-targeted shock produces a larger percentage loss in the target sector than non-target sectors."""
    shock = get_scenario("Credit Event", impact_score=9.0)
    result = run_stress_test(portfolio, shock, target_sector="Financials")
    positions = result["positions"]
    fin_pnl_pct = positions[positions["sector"] == "Financials"]["pnl_pct"].mean()
    other_pnl_pct = positions[positions["sector"] != "Financials"]["pnl_pct"].mean()
    assert fin_pnl_pct < other_pnl_pct, "Financials positions should experience a larger negative PnL %"


def test_spillover_fraction_applied():
    """Verify spillover fraction is correctly applied to non-target positions."""
    loans = pd.DataFrame([
        {"asset_class": "Corporate Loan", "position_id": "L1", "rating": "BBB", "sector": "Financials",
         "ead": 10_000_000, "lgd": 0.45, "pd": 0.02, "mark_to_market": 10_000_000},
        {"asset_class": "Corporate Loan", "position_id": "L2", "rating": "BBB", "sector": "Technology",
         "ead": 10_000_000, "lgd": 0.45, "pd": 0.02, "mark_to_market": 10_000_000},
    ])
    shock = ShockVector(
        spread_shock_bps={"BBB": 100},
        pd_multiplier=2.0,
        target_sector_spread_extra_bps=0.0,
        spillover_fraction=0.25,
    )
    res = revalue_loans(loans, shock, target_sector="Financials")
    fin_pnl = res.loc[res["sector"] == "Financials", "pnl"].iloc[0]
    tech_pnl = res.loc[res["sector"] == "Technology", "pnl"].iloc[0]

    # Target: delta_el = (0.04 - 0.02) * 0.45 * 10M = 90,000; spread_loss = 3.0 * (100/10000) * 10M = 300,000 -> total loss = 390,000
    # Tech: eff_pd_mult = 1.0 + (2.0 - 1.0)*0.25 = 1.25 -> pd_new = 0.025 -> delta_el = 0.005 * 0.45 * 10M = 22,500
    #       spread_loss = 3.0 * (25/10000) * 10M = 75,000 -> total loss = 97,500 = 0.25 * 390,000
    assert tech_pnl == pytest.approx(0.25 * fin_pnl, rel=1e-3)


def test_reverse_stress_ignores_gains(portfolio):
    """Reverse stress test must not breach when scenario results in portfolio gains."""
    gain_shock = ShockVector(equity_shock=0.50)
    rst = reverse_stress_test(portfolio, gain_shock, loss_threshold_pct=5.0)
    assert rst["break_impact_score"] is None, "Gains should not trigger a loss breach"
    assert rst["break_monetary_loss"] is None


def test_financials_vs_utilities_credit_event_loss(portfolio):
    """A Credit Event targeting Financials hits Financials loans harder than Utilities loans."""
    shock = get_scenario("Credit Event", impact_score=9.0)
    result = run_stress_test(portfolio, shock, target_sector="Financials")
    loans = result["positions"][result["positions"]["asset_class"] == "Corporate Loan"]
    fin_loans = loans[loans["sector"] == "Financials"]
    util_loans = loans[loans["sector"] == "Utilities"]

    if not fin_loans.empty and not util_loans.empty:
        fin_mean_loss_pct = fin_loans["pnl_pct"].mean()
        util_mean_loss_pct = util_loans["pnl_pct"].mean()
        assert fin_mean_loss_pct < util_mean_loss_pct, "Financials loans should lose more than Utilities loans"
