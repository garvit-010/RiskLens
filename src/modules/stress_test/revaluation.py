"""
RiskLens – portfolio revaluation engine.

Simplified first-order approximation (labeled as such):
  - Bonds:   ΔP = -Duration × Δrate  +  ½ × Convexity × (Δrate)²
             also spread-based loss: -Duration × Δspread_bps/10000
  - Loans:   ΔEL = (PD_new - PD_old) × LGD × EAD
             spread-based MTM: -Modified_duration_proxy × Δspread / 10000 × EAD
  - IR Swaps: ΔPNL = DV01 × Δrate_bps  (positive when rates rise for pay-fixed swap)
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


def revalue_bonds(
    bonds: pd.DataFrame, shock: ShockVector, target_sector: str | None = None
) -> pd.DataFrame:
    """
    Apply rate and spread shocks to bond positions.
    When target_sector is specified:
      - target sector receives full rating spread shock + target_sector_spread_extra_bps
      - other sectors receive spillover_fraction of rating spread shock
    Returns DataFrame with added columns: stressed_value, pnl, pnl_pct
    """
    bonds = bonds.copy()
    rate_delta  = shock.rate_shock_bps / 10_000   # to decimal

    def _calc_spread_bps(row: pd.Series) -> float:
        base_spread = _spread_shock_for_rating(str(row.get("rating", "")), shock)
        if target_sector is not None:
            sector = str(row.get("sector", ""))
            if sector == target_sector:
                return base_spread + shock.target_sector_spread_extra_bps
            else:
                return base_spread * shock.spillover_fraction
        return base_spread

    spread_delta = bonds.apply(_calc_spread_bps, axis=1) / 10_000

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
    target_info = f", target={target_sector}" if target_sector else ""
    bonds["shock_applied"] = (
        f"rate Δ{shock.rate_shock_bps:+.0f}bps, spread Δ by rating{target_info}"
    )
    return bonds


def revalue_loans(
    loans: pd.DataFrame, shock: ShockVector, target_sector: str | None = None
) -> pd.DataFrame:
    """
    Apply PD multiplier and spread shock to corporate loan positions.
    When target_sector is specified:
      - target sector receives full pd_multiplier and rating spread + target_sector_spread_extra_bps
      - other sectors receive spillover_fraction applied to PD increase and spread shock
    ΔEL = (PD_new - PD_old) × LGD × EAD
    """
    loans = loans.copy()
    pd_col  = loans["pd"].fillna(0.01)
    lgd_col = loans["lgd"].fillna(0.45)
    ead_col = loans["ead"].fillna(0)

    def _calc_loan_shocks(row: pd.Series) -> tuple[float, float]:
        base_spread = _spread_shock_for_rating(str(row.get("rating", "")), shock)
        if target_sector is not None:
            sector = str(row.get("sector", ""))
            if sector == target_sector:
                eff_pd_mult = shock.pd_multiplier
                eff_spread = base_spread + shock.target_sector_spread_extra_bps
            else:
                eff_pd_mult = 1.0 + (shock.pd_multiplier - 1.0) * shock.spillover_fraction
                eff_spread = base_spread * shock.spillover_fraction
        else:
            eff_pd_mult = shock.pd_multiplier
            eff_spread = base_spread
        return eff_pd_mult, eff_spread

    shock_tuples = loans.apply(_calc_loan_shocks, axis=1)
    pd_mults = pd.Series([t[0] for t in shock_tuples], index=loans.index)
    spread_deltas = pd.Series([t[1] / 10_000 for t in shock_tuples], index=loans.index)

    pd_new = pd_col * pd_mults
    delta_el = (pd_new - pd_col) * lgd_col * ead_col   # positive = more loss

    # Spread widening: mark-to-market loss ≈ -duration × Δspread × EAD
    spread_loss = _LOAN_DURATION_PROXY * spread_deltas * ead_col  # positive = loss

    total_pnl = -(delta_el + spread_loss)   # negative = net loss
    mv = loans["mark_to_market"].fillna(ead_col)
    loans["stressed_value"] = np.round(mv + total_pnl, 0)
    loans["pnl"] = np.round(total_pnl, 0)
    loans["pnl_pct"] = np.where(mv != 0, loans["pnl"] / mv * 100, 0.0)
    target_info = f", target={target_sector}" if target_sector else ""
    loans["shock_applied"] = f"PD×{shock.pd_multiplier:.2f}, spread by rating{target_info}"
    return loans


def revalue_derivatives(derivs: pd.DataFrame, shock: ShockVector) -> pd.DataFrame:
    """
    Apply rate shock to IR swaps.
    Pay-fixed swap: bank pays fixed and receives floating.
    When rates rise (Δrate > 0), floating cash flows increase, yielding a positive MTM gain.
    Modeled as: ΔPNL = DV01 × Δrate_bps (positive when rates rise).
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
    derivs["shock_applied"] = f"rate Δ{shock.rate_shock_bps:+.0f}bps (pay-fixed)"
    return derivs


def revalue_equities(
    equities: pd.DataFrame, shock: ShockVector, target_sector: str | None = None
) -> pd.DataFrame:
    """
    ΔPNL = beta × equity_shock × market_value
    When target_sector is specified, non-target positions receive spillover_fraction of the equity shock.
    """
    equities = equities.copy()
    beta = equities["beta"].fillna(1.0)
    mv   = equities["market_value"].fillna(equities.get("mark_to_market", 0))

    if target_sector is not None:
        eq_shock = equities["sector"].apply(
            lambda s: shock.equity_shock if str(s) == target_sector else shock.equity_shock * shock.spillover_fraction
        )
    else:
        eq_shock = shock.equity_shock

    pnl  = beta * eq_shock * mv
    equities["stressed_value"] = np.round(mv + pnl, 0)
    equities["pnl"] = np.round(pnl, 0)
    equities["pnl_pct"] = np.where(mv != 0, equities["pnl"] / mv * 100, 0.0)
    target_info = f", target={target_sector}" if target_sector else ""
    equities["shock_applied"] = f"equity shock {shock.equity_shock:+.1%}{target_info}"
    return equities


def run_stress_test(
    portfolio: pd.DataFrame, shock: ShockVector, target_sector: str | None = None
) -> dict:
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
        results.append(revalue_bonds(bonds, shock, target_sector=target_sector))
    if len(loans):
        results.append(revalue_loans(loans, shock, target_sector=target_sector))
    if len(derivs):
        results.append(revalue_derivatives(derivs, shock))
    if len(equities):
        results.append(revalue_equities(equities, shock, target_sector=target_sector))

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
    target_sector: str | None = None,
) -> dict:
    """
    Sweep severity from 0 to max_severity and find the impact score
    where portfolio loss exceeds loss_threshold_pct (signed: pnl_pct <= -threshold).
    Refines break impact score using bisection search.

    Returns
    -------
    {
      "threshold_pct":        float,
      "break_impact_score":   float | None,
      "break_monetary_loss":  float | None,
      "scaled_shock":         dict | None,
      "sweep_results":        list[{impact_score, pnl_pct}]
    }
    """
    from src.modules.stress_test.scenarios import scale_scenario

    sweep = []
    break_low = None
    break_high = None

    severities = np.linspace(0, max_severity, steps)
    prev_impact = 0.0

    for impact in severities:
        scaled = scale_scenario(shock, impact_score=impact)
        result = run_stress_test(portfolio, scaled, target_sector=target_sector)
        pnl_pct = result["pnl_pct"]
        sweep.append({"impact_score": round(float(impact), 2), "pnl_pct": round(pnl_pct, 2)})

        # Signed check: only a loss (negative PnL) breaches the loss threshold
        if break_high is None and pnl_pct <= -loss_threshold_pct:
            break_low = prev_impact
            break_high = float(impact)
        prev_impact = float(impact)

    break_score = None
    break_monetary_loss = None
    scaled_shock_dict = None

    if break_high is not None:
        # Refine break point via bisection search (15 iterations gives < 0.001 precision)
        low, high = break_low, break_high
        for _ in range(15):
            mid = (low + high) / 2.0
            mid_scaled = scale_scenario(shock, impact_score=mid)
            mid_res = run_stress_test(portfolio, mid_scaled, target_sector=target_sector)
            if mid_res["pnl_pct"] <= -loss_threshold_pct:
                high = mid
            else:
                low = mid
        break_score = round(float(high), 2)
        refined_scaled = scale_scenario(shock, impact_score=break_score)
        refined_result = run_stress_test(portfolio, refined_scaled, target_sector=target_sector)
        break_monetary_loss = refined_result["total_pnl"]
        scaled_shock_dict = {
            "equity_shock": round(refined_scaled.equity_shock, 4),
            "rate_shock_bps": round(refined_scaled.rate_shock_bps, 2),
            "pd_multiplier": round(refined_scaled.pd_multiplier, 3),
            "target_sector_spread_extra_bps": round(refined_scaled.target_sector_spread_extra_bps, 2),
            "spillover_fraction": round(refined_scaled.spillover_fraction, 2),
        }

    return {
        "threshold_pct":        loss_threshold_pct,
        "break_impact_score":   break_score,
        "break_monetary_loss":  break_monetary_loss,
        "scaled_shock":         scaled_shock_dict,
        "sweep_results":        sweep,
    }
