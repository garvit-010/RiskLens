"""
RiskLens – portfolio revaluation engine.

Simplified first-order approximation (labeled as such):
  - Bonds:   ΔP = -Duration × Δrate  +  ½ × Convexity × (Δrate)²
             also spread-based loss: -Duration × Δspread_bps/10000
  - Loans:   ΔEL = (PD_new - PD_old) × LGD × EAD
             spread-based MTM: -Modified_duration_proxy × Δspread / 10000 × EAD
  - IR Swaps: ΔPNL = -DV01 × Δrate_bps  (negative because pay-fixed)
  - Equity:   ΔPNL = beta × equity_shock × market_value

DISCLAIMER: Not production-grade. For demonstration and regulatory
stress-testing illustration only.  See docs/ASSUMPTIONS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modules.stress_test.scenarios import ShockVector

# Assumed modified duration for loans (no explicit duration field)
_LOAN_DURATION_PROXY = 3.0   # years (ASSUMPTION)


def _spread_shock_for_rating(rating: str, shock: ShockVector) -> float:
    """Return the spread shock in bps for this rating, or 0 if not defined."""
    if not shock.spread_shock_bps:
        return 0.0
    return float(shock.spread_shock_bps.get(rating, 0.0))


def revalue_bonds(bonds: pd.DataFrame, shock: ShockVector) -> pd.DataFrame:
    """
    Apply rate and spread shocks to bond positions.
    Returns DataFrame with added columns: stressed_value, pnl, pnl_pct
    """
    bonds = bonds.copy()
    rate_delta  = shock.rate_shock_bps / 10_000   # to decimal

    spread_delta = bonds["rating"].apply(
        lambda r: _spread_shock_for_rating(str(r), shock) / 10_000
    )

    # Duration + convexity price change for rates
    dur  = bonds["duration"].fillna(5.0)
    conv = bonds["convexity"].fillna(0.5)
    dp_rate   = -dur * rate_delta + 0.5 * conv * rate_delta ** 2
    dp_spread = -dur * spread_delta

    total_dp = dp_rate + dp_spread
    mv = bonds["mark_to_market"].fillna(0)
    bonds["stressed_value"] = np.round(mv * (1 + total_dp), 0)
    bonds["pnl"] = np.round(bonds["stressed_value"] - mv, 0)
    bonds["pnl_pct"] = np.where(mv != 0, bonds["pnl"] / mv * 100, 0.0)
    bonds["shock_applied"] = (
        f"rate Δ{shock.rate_shock_bps:+.0f}bps, spread Δ by rating"
    )
    return bonds


def revalue_loans(loans: pd.DataFrame, shock: ShockVector) -> pd.DataFrame:
    """
    Apply PD multiplier and spread shock to corporate loan positions.
    ΔEL = (PD_new - PD_old) × LGD × EAD
    """
    loans = loans.copy()
    pd_col  = loans["pd"].fillna(0.01)
    lgd_col = loans["lgd"].fillna(0.45)
    ead_col = loans["ead"].fillna(0)

    pd_new = pd_col * shock.pd_multiplier
    delta_el = (pd_new - pd_col) * lgd_col * ead_col   # positive = more loss

    spread_delta = loans["rating"].apply(
        lambda r: _spread_shock_for_rating(str(r), shock) / 10_000
    )
    # Spread widening: mark-to-market loss ≈ -duration × Δspread × EAD
    # This is negative (a loss), so we add to delta_el before negating
    spread_loss = _LOAN_DURATION_PROXY * spread_delta * ead_col  # positive = loss

    total_pnl = -(delta_el + spread_loss)   # negative = net loss
    mv = loans["mark_to_market"].fillna(ead_col)
    loans["stressed_value"] = np.round(mv + total_pnl, 0)
    loans["pnl"] = np.round(total_pnl, 0)
    loans["pnl_pct"] = np.where(mv != 0, loans["pnl"] / mv * 100, 0.0)
    loans["shock_applied"] = f"PD×{shock.pd_multiplier:.2f}, spread by rating"
    return loans


def revalue_derivatives(derivs: pd.DataFrame, shock: ShockVector) -> pd.DataFrame:
    """
    Apply rate shock to IR swaps.
    Pay-fixed swap: higher rates → positive MTM for the bank as receiver.
    We model as: ΔPNL = DV01 × Δrate_bps (positive if rates rise, pay-fixed)
    """
    derivs = derivs.copy()
    dv01 = derivs["dv01"].fillna(0)
    pnl  = dv01 * shock.rate_shock_bps   # positive if rates rise
    mv   = derivs["mark_to_market"].fillna(0)
    derivs["stressed_value"] = np.round(mv + pnl, 0)
    derivs["pnl"] = np.round(pnl, 0)
    derivs["pnl_pct"] = np.where(
        mv.abs() > 0, derivs["pnl"] / mv.abs() * 100, 0.0
    )
    derivs["shock_applied"] = f"rate Δ{shock.rate_shock_bps:+.0f}bps"
    return derivs


def revalue_equities(equities: pd.DataFrame, shock: ShockVector) -> pd.DataFrame:
    """ΔPNL = beta × equity_shock × market_value"""
    equities = equities.copy()
    beta = equities["beta"].fillna(1.0)
    mv   = equities["market_value"].fillna(equities.get("mark_to_market", 0))
    pnl  = beta * shock.equity_shock * mv
    equities["stressed_value"] = np.round(mv + pnl, 0)
    equities["pnl"] = np.round(pnl, 0)
    equities["pnl_pct"] = np.where(mv != 0, equities["pnl"] / mv * 100, 0.0)
    equities["shock_applied"] = f"equity shock {shock.equity_shock:+.1%}"
    return equities


def run_stress_test(portfolio: pd.DataFrame, shock: ShockVector) -> dict:
    """
    Run the full stress test and return a structured result dict.

    Returns
    -------
    {
      "before_value": float,
      "after_value":  float,
      "total_pnl":    float,
      "pnl_pct":      float,
      "positions":    DataFrame (stressed),
      "by_asset_class": dict,
      "by_sector":    dict,
      "by_country":   dict,
      "top_losers":   list[dict],
    }
    """
    ac = portfolio["asset_class"]
    bonds   = portfolio[ac == "Bond"].copy()
    loans   = portfolio[ac == "Corporate Loan"].copy()
    derivs  = portfolio[ac == "IR Swap"].copy()
    equities = portfolio[ac == "Equity"].copy()

    results = []
    if len(bonds):
        results.append(revalue_bonds(bonds, shock))
    if len(loans):
        results.append(revalue_loans(loans, shock))
    if len(derivs):
        results.append(revalue_derivatives(derivs, shock))
    if len(equities):
        results.append(revalue_equities(equities, shock))

    stressed = pd.concat(results, ignore_index=True)

    before_val = float(stressed["mark_to_market"].sum())
    after_val  = float(stressed["stressed_value"].sum())
    total_pnl  = after_val - before_val
    pnl_pct    = (total_pnl / before_val * 100) if before_val != 0 else 0.0

    by_ac = (
        stressed.groupby("asset_class")["pnl"]
        .sum().round(0).to_dict()
    )
    by_sector = (
        stressed.groupby("sector")["pnl"]
        .sum().round(0).sort_values().head(10).to_dict()
    )
    by_country = (
        stressed.groupby("country")["pnl"]
        .sum().round(0).sort_values().head(10).to_dict()
    )

    top_losers = (
        stressed.nsmallest(10, "pnl")[
            ["position_id", "asset_class", "sector", "country", "pnl", "pnl_pct"]
        ].to_dict(orient="records")
    )

    return {
        "before_value": round(before_val, 0),
        "after_value":  round(after_val, 0),
        "total_pnl":    round(total_pnl, 0),
        "pnl_pct":      round(pnl_pct, 2),
        "positions":    stressed,
        "by_asset_class": by_ac,
        "by_sector":    by_sector,
        "by_country":   by_country,
        "top_losers":   top_losers,
        "disclaimer":   "SIMPLIFIED FIRST-ORDER APPROXIMATION. Not production-grade.",
    }


def reverse_stress_test(
    portfolio: pd.DataFrame,
    shock: ShockVector,
    loss_threshold_pct: float = 5.0,
    max_severity: float = 10.0,
    steps: int = 50,
) -> dict:
    """
    Sweep severity from 0 to max_severity and find the impact score
    where portfolio loss exceeds loss_threshold_pct.

    Returns
    -------
    {
      "threshold_pct":        float,
      "break_impact_score":   float | None,
      "sweep_results":        list[{impact_score, pnl_pct}]
    }
    """
    from src.modules.stress_test.scenarios import scale_scenario

    sweep = []
    break_score = None

    for impact in np.linspace(0, max_severity, steps):
        scaled = scale_scenario(shock, impact_score=impact)
        result = run_stress_test(portfolio, scaled)
        pnl_pct = result["pnl_pct"]
        sweep.append({"impact_score": round(float(impact), 2), "pnl_pct": round(pnl_pct, 2)})
        if break_score is None and abs(pnl_pct) >= loss_threshold_pct:
            break_score = round(float(impact), 2)

    return {
        "threshold_pct":      loss_threshold_pct,
        "break_impact_score": break_score,
        "sweep_results":      sweep,
    }
