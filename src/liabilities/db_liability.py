from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.liabilities.survival import one_year_survival_probs


@dataclass(frozen=True)
class DBLiabilitySpec:
    """
    Stylised single-cohort DB pension:

    - Member is age x0 at valuation date (base_year)
    - Retires at retirement_age (e.g. 65)
    - After retirement, pays an annual pension amount (in nominal terms for now)
      at the END of each year while alive (annuity-immediate).
    - We cap at max_age for practicality.

    Later we add:
    - inflation indexation
    - spouse benefits
    - improvements year-by-year
    """
    base_year: int = 2023
    x0: int = 55
    retirement_age: int = 65
    max_age: int = 110

    annual_pension: float = 10_000.0  # £ per year
    flat_discount_rate: float = 0.03  # flat yield for now


def build_expected_cashflows(
    df_mort: pd.DataFrame,
    spec: DBLiabilitySpec,
    kappa_shift: float = 0.0,
) -> pd.DataFrame:
    """
    Build expected pension cashflows by year.

    For payment at time t (in years from valuation):
      Expected CF_t = benefit * Pr(alive at payment time)
    where Pr(alive) uses period survival from base_year.

    Payment starts at retirement, end of each year.
    If x0=55 and retirement=65, first payment at t=10.
    """
    start_t = spec.retirement_age - spec.x0
    if start_t < 0:
        raise ValueError("retirement_age must be >= x0")

    # last payment time (cap by max_age)
    last_t = spec.max_age - spec.x0
    times = np.arange(0, last_t + 1)

    px_by_age = one_year_survival_probs(
        df_mort=df_mort,
        year=spec.base_year,
        kappa_shift=kappa_shift,
    )
    ages = spec.x0 + np.arange(last_t, dtype=int)
    px = px_by_age.reindex(ages)
    if px.isna().any():
        missing = ages[px.isna().to_numpy()]
        raise ValueError(
            f"Ages {missing.tolist()} missing in mortality table for year={spec.base_year}"
        )

    survival_probs = np.ones(last_t + 1, dtype=float)
    survival_probs[1:] = np.cumprod(px.to_numpy(dtype=float))

    benefit = np.where(times < start_t, 0.0, float(spec.annual_pension))
    expected_cf = benefit * survival_probs
    years = spec.base_year + times

    return pd.DataFrame(
        {
            "t": times.astype(int),
            "year": years.astype(int),
            "benefit": benefit,
            "survival_prob": survival_probs,
            "expected_cashflow": expected_cf,
        }
    )


def present_value_from_expected_cashflows(
    cf: pd.DataFrame,
    flat_discount_rate: float,
) -> float:
    """
    PV = sum_t expected_cashflow_t * v^t, where v = (1+i)^(-1)
    """
    i = float(flat_discount_rate)
    v = 1.0 / (1.0 + i)

    t = cf["t"].to_numpy(dtype=float)
    cf_t = cf["expected_cashflow"].to_numpy(dtype=float)
    return float(np.sum(cf_t * (v ** t)))


def present_value_from_curve(
    cf: pd.DataFrame,
    curve,
) -> float:
    """
    PV = sum_t expected_cashflow_t * P(0,t)

    curve must have method: discount_factor(t: float) -> float
    """
    t = cf["t"].to_numpy(dtype=float)
    cf_t = cf["expected_cashflow"].to_numpy(dtype=float)
    dfs = np.array([float(curve.discount_factor(tt)) for tt in t], dtype=float)
    return float(np.sum(cf_t * dfs))
