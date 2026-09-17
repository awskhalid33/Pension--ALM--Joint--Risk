from __future__ import annotations

import numpy as np
import pandas as pd


def pv_remaining_cashflows_at_time_k(
    cf: pd.DataFrame,
    k: int,
    curve_model,
    curve_factors,
) -> float:
    """
    Value at time step k (i.e., at t=k years from original valuation),
    of remaining expected cashflows.

    We treat cf['t'] as years from original valuation.
    At time k, a cashflow at time t has time-to-maturity = (t - k).

    curve_model must provide discount_factor(factors, ttm).
    """
    t = cf["t"].to_numpy(dtype=float)
    cf_t = cf["expected_cashflow"].to_numpy(dtype=float)
    mask = t >= float(k)
    if not np.any(mask):
        return 0.0

    ttm = t[mask] - float(k)
    dfs = np.array(
        [
            1.0 if m <= 0.0 else float(curve_model.discount_factor(curve_factors, m))
            for m in ttm
        ],
        dtype=float,
    )
    return float(np.sum(cf_t[mask] * dfs))


def duration_remaining_cashflows_at_time_k(
    cf: pd.DataFrame,
    k: int,
    curve_model,
    curve_factors,
) -> float:
    """
    Macaulay duration (in years from time k) of remaining expected cashflows.

    D(k) = sum_{t>=k} (ttm * CF_t * P_k(ttm)) / PV_k
    where ttm = t - k.
    """
    t = cf["t"].to_numpy(dtype=float)
    cf_t = cf["expected_cashflow"].to_numpy(dtype=float)
    mask = t >= float(k)
    if not np.any(mask):
        return float("nan")

    ttm = t[mask] - float(k)
    dfs = np.array(
        [
            1.0 if m <= 0.0 else float(curve_model.discount_factor(curve_factors, m))
            for m in ttm
        ],
        dtype=float,
    )
    pv = float(np.sum(cf_t[mask] * dfs))
    weighted = float(np.sum(ttm * cf_t[mask] * dfs))

    if pv <= 0.0:
        return float("nan")

    return float(weighted / pv)
