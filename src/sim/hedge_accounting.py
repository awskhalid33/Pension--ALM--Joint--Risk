"""Annual expected-cohort accounting, valued immediately before each payment."""
from __future__ import annotations

import numpy as np

from src.liabilities.db_liability import build_expected_cashflows
from src.liabilities.survival import one_year_survival_probs
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k
from src.sim.liability_setup import default_liability_spec


def projected_cashflows(df_mort, spec, kappas, k):
    """Past intervals use their observed start-of-year kappa; future ones use kappa[k].

    Survivors are expected cohort counts per initial member, not individual deaths.
    The future mortality level is held flat at today's level (no foresight).
    """
    if k < 0 or k >= len(kappas) or not np.all(np.isfinite(kappas[:k + 1])):
        raise ValueError("Need finite mortality history through valuation year")
    cf = build_expected_cashflows(df_mort, spec)
    last_t = spec.max_age - spec.x0
    if k > last_t:
        raise ValueError("Projection exceeds liability maximum age")
    px = one_year_survival_probs(df_mort, spec.base_year).reindex(
        spec.x0 + np.arange(last_t)
    ).to_numpy(dtype=float)
    shifts = np.full(last_t, kappas[k], dtype=float)
    shifts[:k] = kappas[:k]
    survival = np.r_[1.0, np.cumprod(px ** np.exp(shifts))]
    cf["survival_prob"] = survival
    cf["expected_cashflow"] = cf["benefit"].to_numpy() * survival
    return cf


def funding_path(*, df_mort, n_years, hedge_maturity, regimes, x_path,
                 kappas, models, ratio_policy=None):
    """Self-financing bond/cash account, with benefit outflows.

    Static holds initial bond notional, using the cash account for benefits.
    A policy returns bond value / liability value. Negative cash is borrowing
    at the same short rate as deposits; no costs or collateral constraints.
    The bond must mature after the experiment so no reinvestment is assumed.
    """
    spec = default_liability_spec()
    if n_years < 0 or n_years >= spec.max_age - spec.x0:
        raise ValueError("Horizon must be nonnegative and below liability maximum age")
    if not np.isfinite(hedge_maturity) or hedge_maturity <= n_years:
        raise ValueError("Hedge maturity must exceed projection horizon")
    if any(len(a) != n_years + 1 for a in (regimes, x_path, kappas)):
        raise ValueError("Scenario arrays must have n_years + 1 entries")
    fr = np.zeros(n_years + 1)
    cash = notional = 0.0
    previous_payment = 0.0
    for k in range(n_years + 1):
        if k:
            prev = models.rate_models[int(regimes[k - 1])]
            factors_prev = prev.factors_from_array(x_path[k - 1])
            cash = (cash - previous_payment) * np.exp(prev.zero_rate(factors_prev, 0.0))
        model = models.rate_models[int(regimes[k])]
        factors = model.factors_from_array(x_path[k])
        cf = projected_cashflows(df_mort, spec, kappas, k)
        liability = pv_remaining_cashflows_at_time_k(cf, k, model, factors)
        price = model.discount_factor(factors, hedge_maturity - k)
        if k == 0:
            cash = liability
        assets = cash + notional * price
        fr[k] = assets / liability if liability > 0 else np.nan
        if k == 0 or ratio_policy is not None:
            h = 1.0 if ratio_policy is None else ratio_policy(k, cf, model, factors)
            target = float(h) * liability / price
            cash -= (target - notional) * price
            notional = target
        previous_payment = float(cf.loc[cf.t == k, "expected_cashflow"].iloc[0])
    return fr
