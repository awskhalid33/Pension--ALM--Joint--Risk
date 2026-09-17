from __future__ import annotations

import numpy as np

from src.finance.regime_stochastic_ns import RegimeStochasticNS
from src.regimes.markov import MarkovRegime
from src.sim.scenario import JointRegimeModels


def simulate_joint_paths(
    *,
    n_years: int,
    models: JointRegimeModels,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_years < 0:
        raise ValueError("n_years must be >= 0")

    regime_proc = MarkovRegime(np.asarray(models.transition_matrix, dtype=float))
    regimes = regime_proc.simulate(n_steps=n_years, s0=models.initial_regime, seed=seed + 10)

    reg_ns = RegimeStochasticNS(models=models.rate_models)
    x_path = reg_ns.simulate(regimes=regimes, x0=np.asarray(models.initial_factors), seed=seed + 2)

    kappas = models.mortality_model.simulate(
        regimes=regimes,
        kappa0=models.initial_kappa,
        seed=seed + 20,
    )

    return regimes, x_path, kappas
