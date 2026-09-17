import numpy as np
import pytest

from src.sim.hedge_accounting import funding_path, projected_cashflows
from src.sim.liability_setup import default_liability_spec, load_toy_mortality
from src.sim.scenario import build_default_joint_regime_models
from src.run_monte_carlo_ml_hedge import duration_features, simulate_ml_duration_hedge_fr
from src.finance.stochastic_ns import NSFactors


def inputs(n=15):
    return dict(df_mort=load_toy_mortality(), n_years=n, hedge_maturity=25,
                regimes=np.zeros(n + 1, dtype=int), x_path=np.zeros((n + 1, 3)),
                kappas=np.zeros(n + 1), models=build_default_joint_regime_models())


@pytest.mark.parametrize("policy", [None, lambda k, cf, model, factors: 0.7])
def test_benefits_do_not_create_funding_surplus(policy):
    # With no interest and no mortality revisions, paying a liability removes
    # exactly the same amount of assets. This includes five post-retirement years.
    fr = funding_path(**inputs(), ratio_policy=policy)
    np.testing.assert_allclose(fr, 1.0, atol=1e-12)


def test_new_mortality_information_does_not_rewrite_past_survivors():
    df = load_toy_mortality()
    spec = default_liability_spec()
    old = np.zeros(16)
    changed = old.copy()
    changed[10:] = -0.5
    cf_old = projected_cashflows(df, spec, old, 10)
    cf_new = projected_cashflows(df, spec, changed, 10)
    np.testing.assert_allclose(cf_old.survival_prob[:11], cf_new.survival_prob[:11])
    assert cf_new.survival_prob.iloc[20] > cf_old.survival_prob.iloc[20]
    np.testing.assert_allclose(projected_cashflows(df, spec, changed, 9).expected_cashflow,
                               projected_cashflows(df, spec, old, 9).expected_cashflow)


def test_duration_features_include_elapsed_time():
    f = NSFactors(0.03, 0, 0)
    assert duration_features(f, np.zeros(11), 10)[-1] == 10


def test_exact_duration_hedge_preserves_accounting():
    fr = simulate_ml_duration_hedge_fr(**inputs(), beta=None, h_floor=0.5, h_cap=1.5)
    np.testing.assert_allclose(fr, 1.0, atol=1e-12)


def test_rejects_bond_expiring_within_horizon():
    args = inputs()
    args['hedge_maturity'] = 10
    with pytest.raises(ValueError, match='maturity'):
        funding_path(**args)
