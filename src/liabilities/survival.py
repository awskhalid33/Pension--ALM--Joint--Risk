from __future__ import annotations

import numpy as np
import pandas as pd


def _get_mx_series(df_mort: pd.DataFrame, year: int) -> pd.Series:
    """
    Extract m_x for a specific calendar year.

    Returns a Series indexed by age with values m_x.
    """
    required_cols = {"year", "age", "mx"}
    missing_cols = required_cols - set(df_mort.columns)
    if missing_cols:
        raise ValueError(f"df_mort is missing columns: {sorted(missing_cols)}")

    sub = df_mort[df_mort["year"] == year].copy()
    if sub.empty:
        raise ValueError(f"No mortality data found for year={year}")

    sub = sub.sort_values("age")
    if sub["age"].duplicated().any():
        raise ValueError(f"Duplicate ages detected for year={year}")
    mx_by_age = pd.Series(sub["mx"].to_numpy(), index=sub["age"].to_numpy())
    if (mx_by_age <= 0).any():
        raise ValueError("Mortality rates mx must be > 0")
    return mx_by_age


def qx_from_mx(mx: np.ndarray) -> np.ndarray:
    """
    Convert central death rates m_x to one-year death probabilities q_x.
    A common approximation is:
        q_x = 1 - exp(-m_x)
    """
    mx = np.asarray(mx, dtype=float)
    if np.any(mx < 0.0):
        raise ValueError("mx must be >= 0")
    return 1.0 - np.exp(-mx)


def one_year_survival_probs(
    df_mort: pd.DataFrame,
    year: int,
    kappa_shift: float = 0.0,
) -> pd.Series:
    """
    Compute p_x = 1 - q_x for all ages in the given year.

    kappa_shift applies a multiplicative shift to mortality:
      log m_x -> log m_x + kappa_shift
      m_x -> m_x * exp(kappa_shift)
    """
    mx_by_age = _get_mx_series(df_mort, year)

    # apply shift in log space
    mx_shifted = mx_by_age.to_numpy() * np.exp(float(kappa_shift))

    qx = qx_from_mx(mx_shifted)
    px = 1.0 - qx
    return pd.Series(px, index=mx_by_age.index)



def t_year_survival(
    df_mort: pd.DataFrame,
    base_year: int,
    base_age: int,
    t: int,
    kappa_shift: float = 0.0,
) -> float:
    """
    Compute {}_t p_{base_age} using period survival probabilities from base_year,
    with a mortality shift kappa_shift.

    {}_t p_x = Π_{k=0}^{t-1} p_{x+k}
    """
    if t < 0:
        raise ValueError("t must be >= 0")
    if t == 0:
        return 1.0

    px_by_age = one_year_survival_probs(df_mort, base_year, kappa_shift=kappa_shift)
    ages = np.arange(base_age, base_age + t, dtype=int)
    px = px_by_age.reindex(ages)
    if px.isna().any():
        missing = ages[px.isna().to_numpy()]
        raise ValueError(
            f"Ages {missing.tolist()} missing in mortality table for year={base_year}"
        )
    return float(np.prod(px.to_numpy(dtype=float)))
