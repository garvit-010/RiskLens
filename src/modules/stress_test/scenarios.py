"""
RiskLens – scenario library.

Defines shock vectors per event type.
Magnitude scales linearly with impact score between floor (5) and 10.
All values are ASSUMPTIONS documented in docs/ASSUMPTIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path

from src.config import SCENARIOS_DIR


@dataclass
class ShockVector:
    """Shock magnitudes at impact score = 10 (scaled down for lower scores)."""
    equity_shock:                   float = 0.0    # fractional change, e.g. -0.08 = -8%
    rate_shock_bps:                 float = 0.0    # parallel rate move in basis points
    spread_shock_bps:               dict  = field(default_factory=dict)  # per rating bucket
    pd_multiplier:                  float = 1.0    # × current PD
    fx_shock_em:                    float = 0.0    # EM FX depreciation
    oil_shock:                      float = 0.0    # oil price change
    sector_overrides:               dict  = field(default_factory=dict)  # sector-specific spread shocks
    target_sector_spread_extra_bps: float = 0.0    # extra spread widening in target sector
    spillover_fraction:             float = 1.0    # fraction of shock applied to non-target sectors


# ---------------------------------------------------------------------------
# Base scenario definitions (at impact score = 10)
# ---------------------------------------------------------------------------
BASE_SCENARIOS: dict[str, ShockVector] = {
    "Credit Event": ShockVector(
        equity_shock                   = -0.12,
        rate_shock_bps                 = 0.0,
        spread_shock_bps               = {"AAA": 20, "AA": 40, "A": 80, "BBB": 150, "BB": 300, "B": 500, "CCC": 800},
        pd_multiplier                  = 2.0,
        target_sector_spread_extra_bps = 100.0,
        spillover_fraction             = 0.25,
    ),
    "Geopolitical": ShockVector(
        equity_shock     = -0.08,
        rate_shock_bps   = 30.0,
        spread_shock_bps = {"AAA": 10, "AA": 20, "A": 40, "BBB": 80, "BB": 150, "B": 250, "CCC": 400},
        pd_multiplier    = 1.3,
        fx_shock_em      = -0.05,
        oil_shock        =  0.15,
    ),
    "Macroeconomic": ShockVector(
        equity_shock     = -0.05,
        rate_shock_bps   = 200.0,
        spread_shock_bps = {"AAA": 5, "AA": 15, "A": 30, "BBB": 60, "BB": 120, "B": 200, "CCC": 350},
        pd_multiplier    = 1.2,
    ),
    "Regulatory/Legal": ShockVector(
        equity_shock                   = -0.04,
        spread_shock_bps               = {"BBB": 40, "BB": 80, "B": 150, "CCC": 250},
        pd_multiplier                  = 1.1,
        target_sector_spread_extra_bps = 50.0,
        spillover_fraction             = 0.25,
    ),
    "Merger/Acquisition": ShockVector(
        equity_shock     = 0.05,   # typically positive
    ),
    "Earnings/Guidance": ShockVector(
        equity_shock     = -0.06,
        spread_shock_bps = {"BBB": 30, "BB": 60, "B": 100},
        pd_multiplier    = 1.15,
    ),
    "Product Launch": ShockVector(
        equity_shock     = 0.03,
    ),
    "Operational/Cyber": ShockVector(
        equity_shock                   = -0.05,
        spread_shock_bps               = {"BBB": 50, "BB": 100, "B": 180},
        pd_multiplier                  = 1.2,
        target_sector_spread_extra_bps = 50.0,
        spillover_fraction             = 0.25,
    ),
    "Supply Chain/Commodity": ShockVector(
        equity_shock     = -0.04,
        oil_shock        =  0.10,
        spread_shock_bps = {"BBB": 30, "BB": 70, "B": 120},
        pd_multiplier    = 1.1,
    ),
    "ESG/Climate": ShockVector(
        equity_shock     = -0.02,
        spread_shock_bps = {"BBB": 15, "BB": 30, "B": 60},
        pd_multiplier    = 1.05,
    ),
    "Other/Noise": ShockVector(
        equity_shock     = -0.01,
    ),
}


def scale_scenario(base: ShockVector, impact_score: float, floor: float = 5.0) -> ShockVector:
    """
    Scale shock magnitudes linearly from floor to 10.
    At impact = floor → 0% of base shocks.
    At impact = 10   → 100% of base shocks.
    """
    scale = max(0.0, (impact_score - floor) / (10.0 - floor))

    return ShockVector(
        equity_shock                   = base.equity_shock     * scale,
        rate_shock_bps                 = base.rate_shock_bps   * scale,
        spread_shock_bps               = {k: v * scale for k, v in base.spread_shock_bps.items()},
        pd_multiplier                  = 1.0 + (base.pd_multiplier - 1.0) * scale,
        fx_shock_em                    = base.fx_shock_em      * scale,
        oil_shock                      = base.oil_shock        * scale,
        sector_overrides               = base.sector_overrides,
        target_sector_spread_extra_bps = base.target_sector_spread_extra_bps * scale,
        spillover_fraction             = base.spillover_fraction,
    )


def get_scenario(event_type: str, impact_score: float) -> ShockVector:
    """Return a scaled ShockVector for the given event type and impact score."""
    base = BASE_SCENARIOS.get(event_type, BASE_SCENARIOS["Other/Noise"])
    return scale_scenario(base, impact_score)


def save_scenarios_yaml() -> None:
    """Persist the scenario definitions to YAML for inspection."""
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sv in BASE_SCENARIOS.items():
        out[name] = {
            "equity_shock":                   sv.equity_shock,
            "rate_shock_bps":                 sv.rate_shock_bps,
            "spread_shock_bps":               sv.spread_shock_bps,
            "pd_multiplier":                  sv.pd_multiplier,
            "target_sector_spread_extra_bps": sv.target_sector_spread_extra_bps,
            "spillover_fraction":             sv.spillover_fraction,
            "note":                           "ASSUMPTION – documented defaults. See docs/ASSUMPTIONS.md.",
        }
    path = SCENARIOS_DIR / "scenarios.yaml"
    with open(path, "w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=True)
    print(f"Scenarios saved to {path}")
