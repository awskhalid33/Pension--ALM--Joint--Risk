from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NSFactors:
    level: float
    slope: float
    curvature: float


@dataclass(frozen=True)
class StochasticNelsonSiegelModel:
    """
    Stochastic Nelson–Siegel in factor form.

    Factors x_t = [level, slope, curvature]^T evolve as VAR(1):
      x_{t+1} = A x_t + eps_{t+1},  eps ~ N(0, Sigma)

    Mapping to zero rates uses standard NS loadings with fixed tau.
    """
    A: np.ndarray              # 3x3
    Sigma: np.ndarray          # 3x3 covariance
    tau: float = 2.5
    dt_years: float = 0.25     # quarterly steps

    def __post_init__(self) -> None:
        a = np.asarray(self.A, dtype=float)
        sigma = np.asarray(self.Sigma, dtype=float)

        if a.shape != (3, 3):
            raise ValueError("A must have shape (3, 3)")
        if sigma.shape != (3, 3):
            raise ValueError("Sigma must have shape (3, 3)")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(sigma)):
            raise ValueError("A and Sigma must contain finite values")
        if not np.allclose(sigma, sigma.T, atol=1e-12):
            raise ValueError("Sigma must be symmetric")
        eigvals = np.linalg.eigvalsh(sigma)
        if np.min(eigvals) < -1e-12:
            raise ValueError("Sigma must be positive semi-definite")
        if self.tau <= 0.0:
            raise ValueError("tau must be > 0")
        if self.dt_years <= 0.0:
            raise ValueError("dt_years must be > 0")

    def zero_rate(self, factors: NSFactors, ttm: float) -> float:
        """
        Zero rate for time-to-maturity (ttm) in years, continuous compounding.
        """
        ttm = float(ttm)
        if ttm <= 0.0:
            # limit t->0: level + slope
            return float(factors.level + factors.slope)

        x = ttm / float(self.tau)
        a = (1.0 - np.exp(-x)) / x
        b = a - np.exp(-x)

        z = factors.level + factors.slope * a + factors.curvature * b
        return float(z)

    def discount_factor(self, factors: NSFactors, ttm: float) -> float:
        z = self.zero_rate(factors, ttm)
        return float(np.exp(-z * float(ttm)))

    def step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        One VAR(1) step.
        """
        if x.shape != (3,):
            raise ValueError("x must have shape (3,)")
        eps = rng.multivariate_normal(mean=np.zeros(3), cov=self.Sigma)
        x_next = self.A @ x + eps
        return x_next

    def simulate_factors(
        self,
        x0: np.ndarray,
        n_steps: int,
        seed: int = 123,
    ) -> np.ndarray:
        """
        Returns array shape (n_steps+1, 3)
        """
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        if np.asarray(x0).shape != (3,):
            raise ValueError("x0 must have shape (3,)")

        rng = np.random.default_rng(seed)
        x_path = np.zeros((n_steps + 1, 3), dtype=float)
        x_path[0] = np.asarray(x0, dtype=float)

        for t in range(n_steps):
            x_path[t + 1] = self.step(x_path[t], rng)

        return x_path

    @staticmethod
    def factors_from_array(x: np.ndarray) -> NSFactors:
        return NSFactors(level=float(x[0]), slope=float(x[1]), curvature=float(x[2]))
