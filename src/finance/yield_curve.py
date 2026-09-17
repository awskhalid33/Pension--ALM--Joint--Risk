from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NelsonSiegelCurve:
    """
    Simple deterministic Nelson–Siegel zero rate curve.

    z(t) = beta0
         + beta1 * ((1 - exp(-t/tau)) / (t/tau))
         + beta2 * (((1 - exp(-t/tau)) / (t/tau)) - exp(-t/tau))

    where:
    - beta0 = long-run level
    - beta1 = slope
    - beta2 = curvature
    - tau   = decay parameter
    """
    beta0: float = 0.03
    beta1: float = -0.01
    beta2: float = 0.01
    tau: float = 2.5

    def zero_rate(self, t: float) -> float:
        """
        Annualised continuously-compounded zero rate z(t).
        """
        t = float(t)
        if t <= 0.0:
            # limit t -> 0 gives beta0 + beta1
            return float(self.beta0 + self.beta1)

        x = t / float(self.tau)
        a = (1.0 - np.exp(-x)) / x
        b = a - np.exp(-x)

        z = self.beta0 + self.beta1 * a + self.beta2 * b
        return float(z)

    def discount_factor(self, t: float) -> float:
        """
        Discount factor P(0,t) using continuous compounding:
          P(0,t) = exp(-z(t) * t)
        """
        zt = self.zero_rate(t)
        return float(np.exp(-zt * float(t)))
