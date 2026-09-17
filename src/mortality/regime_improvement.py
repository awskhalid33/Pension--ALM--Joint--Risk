from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RegimeMortalityModel:
    """
    Mortality improvement with regime-dependent drift and volatility.
    """
    mu: dict[int, float]        # mu[state]
    sigma: dict[int, float]     # sigma[state]

    def __post_init__(self) -> None:
        mu_keys = set(self.mu.keys())
        sigma_keys = set(self.sigma.keys())
        if not mu_keys:
            raise ValueError("mu must be non-empty")
        if mu_keys != sigma_keys:
            raise ValueError("mu and sigma must have identical state keys")

        for state, value in self.mu.items():
            if not np.isfinite(value):
                raise ValueError(f"mu[{state}] must be finite")
        for state, value in self.sigma.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"sigma[{state}] must be finite and >= 0")

    def simulate(
        self,
        regimes: np.ndarray,
        kappa0: float = 0.0,
        seed: int = 123,
    ) -> np.ndarray:
        reg = np.asarray(regimes, dtype=int)
        if reg.ndim != 1:
            raise ValueError("regimes must be a 1D array")
        if len(reg) < 1:
            raise ValueError("regimes must contain at least one state")

        unknown = set(np.unique(reg)) - set(self.mu.keys())
        if unknown:
            raise ValueError(f"regimes contain unknown states: {sorted(unknown)}")

        rng = np.random.default_rng(seed)
        t_horizon = len(reg) - 1
        kappa = np.zeros(t_horizon + 1, dtype=float)
        kappa[0] = kappa0

        for t in range(t_horizon):
            s = int(reg[t])
            eps = rng.normal()
            kappa[t + 1] = kappa[t] + self.mu[s] + self.sigma[s] * eps

        return kappa
