import numpy as np
import pandas as pd

from src.liabilities.valuation import (
    duration_remaining_cashflows_at_time_k,
    pv_remaining_cashflows_at_time_k,
)
from src.sim.paths import simulate_joint_paths
from src.sim.scenario import build_default_joint_regime_models


class FlatCurveModel:
    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def discount_factor(self, _factors, ttm: float) -> float:
        return float(np.exp(-self.rate * float(ttm)))


def test_pv_remaining_matches_manual_computation() -> None:
    cf = pd.DataFrame(
        {
            "t": [0, 1, 2],
            "expected_cashflow": [10.0, 10.0, 10.0],
        }
    )
    model = FlatCurveModel(rate=0.03)
    pv = pv_remaining_cashflows_at_time_k(cf=cf, k=1, curve_model=model, curve_factors=None)
    expected = 10.0 + 10.0 * np.exp(-0.03 * 1.0)
    assert np.isclose(pv, expected)


def test_duration_remaining_positive_for_positive_cashflows() -> None:
    cf = pd.DataFrame(
        {
            "t": [1, 2, 3],
            "expected_cashflow": [5.0, 5.0, 5.0],
        }
    )
    model = FlatCurveModel(rate=0.02)
    d = duration_remaining_cashflows_at_time_k(cf=cf, k=0, curve_model=model, curve_factors=None)
    assert 0.0 < d < 3.0


def test_simulate_joint_paths_reproducible() -> None:
    models = build_default_joint_regime_models()
    r1, x1, k1 = simulate_joint_paths(n_years=10, models=models, seed=12345)
    r2, x2, k2 = simulate_joint_paths(n_years=10, models=models, seed=12345)

    assert np.array_equal(r1, r2)
    assert np.allclose(x1, x2)
    assert np.allclose(k1, k2)


def test_simulate_joint_paths_shapes() -> None:
    models = build_default_joint_regime_models()
    regimes, x_path, kappas = simulate_joint_paths(n_years=7, models=models, seed=7)
    assert regimes.shape == (8,)
    assert x_path.shape == (8, 3)
    assert kappas.shape == (8,)
