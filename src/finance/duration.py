from __future__ import annotations

import numpy as np
import pandas as pd


def macaulay_duration_from_curve(
    cf: pd.DataFrame,
    curve,
) -> float:
    """
    Macaulay duration of expected cashflows under a given curve.

    D = sum_t [ t * CF_t * P(0,t) ] / PV
    """
    t = cf["t"].to_numpy(dtype=float)
    cf_t = cf["expected_cashflow"].to_numpy(dtype=float)
    dfs = np.array([float(curve.discount_factor(tt)) for tt in t], dtype=float)

    pv = float(np.sum(cf_t * dfs))
    weighted_sum = float(np.sum(t * cf_t * dfs))

    if pv <= 0.0:
        raise ValueError("PV must be positive to compute duration")

    return float(weighted_sum / pv)


def modified_duration_from_curve(
    cf: pd.DataFrame,
    curve,
    shift: float = 1e-4,
) -> float:
    """
    Approximate modified duration via finite difference:
      D_mod ≈ - (PV_up - PV_down) / (2 * PV * shift)

    where rates are shifted in parallel by 'shift'.
    """
    # Base PV
    t = cf["t"].to_numpy(dtype=float)
    cf_t = cf["expected_cashflow"].to_numpy(dtype=float)
    pv0 = float(np.sum(cf_t * np.array([curve.discount_factor(tt) for tt in t], dtype=float)))
    if pv0 <= 0.0:
        raise ValueError("PV must be positive to compute modified duration")

    # Shift curve up
    class ShiftedCurve:
        def discount_factor(self, t):
            z = curve.zero_rate(t) + shift
            return float(np.exp(-z * t))

    shifted = ShiftedCurve()
    pv_up = float(np.sum(cf_t * np.array([shifted.discount_factor(tt) for tt in t], dtype=float)))
    pv_down = float(
        np.sum(cf_t * np.exp(-(np.array([curve.zero_rate(tt) for tt in t], dtype=float) - shift) * t))
    )

    return float(- (pv_up - pv_down) / (2.0 * pv0 * shift))
