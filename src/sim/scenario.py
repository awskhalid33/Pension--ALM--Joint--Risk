from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.mortality.regime_improvement import RegimeMortalityModel


@dataclass(frozen=True)
class JointRegimeModels:
    transition_matrix: np.ndarray
    rate_models: Mapping[int, StochasticNelsonSiegelModel]
    mortality_model: RegimeMortalityModel
    initial_factors: np.ndarray
    initial_regime: int = 0
    initial_kappa: float = 0.0

    def __post_init__(self) -> None:
        p = np.asarray(self.transition_matrix, dtype=float)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError("transition_matrix must be a square 2D array")
        n_states = p.shape[0]
        if n_states < 1:
            raise ValueError("transition_matrix must have at least one state")

        model_keys = set(self.rate_models.keys())
        expected_keys = set(range(n_states))
        if model_keys != expected_keys:
            raise ValueError(
                f"rate_models keys must match states {sorted(expected_keys)}, got {sorted(model_keys)}"
            )

        x0 = np.asarray(self.initial_factors, dtype=float)
        if x0.shape != (3,):
            raise ValueError("initial_factors must have shape (3,)")

        if self.initial_regime < 0 or self.initial_regime >= n_states:
            raise ValueError("initial_regime is out of bounds for transition_matrix")


def build_default_joint_regime_models(dt_years: float = 1.0) -> JointRegimeModels:
    p = np.array([[0.90, 0.10], [0.20, 0.80]], dtype=float)

    a0 = np.array(
        [
            [0.995, 0.000, 0.000],
            [0.000, 0.990, 0.000],
            [0.000, 0.000, 0.985],
        ],
        dtype=float,
    )
    sigma0 = np.diag([0.00003, 0.00005, 0.00005])

    a1 = np.array(
        [
            [0.990, 0.000, 0.000],
            [0.000, 0.985, 0.000],
            [0.000, 0.000, 0.975],
        ],
        dtype=float,
    )
    sigma1 = np.diag([0.00010, 0.00012, 0.00012])

    rate_models = {
        0: StochasticNelsonSiegelModel(A=a0, Sigma=sigma0, tau=2.5, dt_years=dt_years),
        1: StochasticNelsonSiegelModel(A=a1, Sigma=sigma1, tau=2.5, dt_years=dt_years),
    }

    mortality_model = RegimeMortalityModel(
        mu={0: -0.012, 1: -0.004},
        sigma={0: 0.008, 1: 0.012},
    )

    return JointRegimeModels(
        transition_matrix=p,
        rate_models=rate_models,
        mortality_model=mortality_model,
        initial_factors=np.array([0.030, -0.010, 0.010], dtype=float),
        initial_regime=0,
        initial_kappa=0.0,
    )
