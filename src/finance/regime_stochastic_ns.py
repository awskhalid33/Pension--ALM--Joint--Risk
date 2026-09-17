from __future__ import annotations

import numpy as np
from src.finance.stochastic_ns import StochasticNelsonSiegelModel


class RegimeStochasticNS:
    """
    Nelson–Siegel model with regime-dependent VAR parameters.
    """

    def __init__(self, models: dict[int, StochasticNelsonSiegelModel]):
        if not models:
            raise ValueError("models must be non-empty")
        self.models = models

    def simulate(
        self,
        regimes: np.ndarray,
        x0: np.ndarray,
        seed: int = 123,
    ) -> np.ndarray:
        r = np.asarray(regimes, dtype=int)
        if r.ndim != 1:
            raise ValueError("regimes must be a 1D array")
        if len(r) < 1:
            raise ValueError("regimes must contain at least one state")
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (3,):
            raise ValueError("x0 must have shape (3,)")
        missing_states = set(np.unique(r)) - set(self.models.keys())
        if missing_states:
            raise ValueError(f"missing rate models for states: {sorted(missing_states)}")

        rng = np.random.default_rng(seed)
        t_horizon = len(r) - 1
        x_path = np.zeros((t_horizon + 1, 3), dtype=float)
        x_path[0] = x0

        for t in range(t_horizon):
            model = self.models[int(r[t])]
            eps = rng.multivariate_normal(
                mean=np.zeros(3),
                cov=model.Sigma,
            )
            x_path[t + 1] = model.A @ x_path[t] + eps

        return x_path
