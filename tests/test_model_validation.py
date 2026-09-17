import numpy as np
import pytest

from src.finance.regime_stochastic_ns import RegimeStochasticNS
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.mortality.improvement import MortalityImprovementRW
from src.mortality.regime_improvement import RegimeMortalityModel
from src.regimes.markov import MarkovRegime
from src.sim.scenario import JointRegimeModels, build_default_joint_regime_models


def _valid_ns_model() -> StochasticNelsonSiegelModel:
    return StochasticNelsonSiegelModel(
        A=np.eye(3) * 0.99,
        Sigma=np.diag([1e-4, 2e-4, 3e-4]),
        tau=2.5,
        dt_years=1.0,
    )


def test_markov_rejects_invalid_transition_rows() -> None:
    bad_p = np.array([[0.9, 0.2], [0.1, 0.8]], dtype=float)
    with pytest.raises(ValueError, match="row must sum to 1"):
        MarkovRegime(bad_p)


def test_markov_rejects_negative_probabilities() -> None:
    bad_p = np.array([[1.1, -0.1], [0.2, 0.8]], dtype=float)
    with pytest.raises(ValueError, match="non-negative"):
        MarkovRegime(bad_p)


def test_markov_rejects_invalid_initial_state() -> None:
    proc = MarkovRegime(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float))
    with pytest.raises(ValueError, match="valid state index"):
        proc.simulate(n_steps=10, s0=2, seed=1)


def test_stochastic_ns_validates_sigma_psd() -> None:
    with pytest.raises(ValueError, match="positive semi-definite"):
        StochasticNelsonSiegelModel(
            A=np.eye(3),
            Sigma=np.array([[1.0, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        )


def test_stochastic_ns_validates_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        StochasticNelsonSiegelModel(A=np.eye(2), Sigma=np.eye(2))


def test_regime_mortality_rejects_unknown_states() -> None:
    model = RegimeMortalityModel(mu={0: -0.01, 1: -0.005}, sigma={0: 0.01, 1: 0.02})
    with pytest.raises(ValueError, match="unknown states"):
        model.simulate(regimes=np.array([0, 2]), kappa0=0.0, seed=1)


def test_mortality_rw_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        MortalityImprovementRW(mu=-0.01, sigma=-0.1)


def test_regime_stochastic_ns_rejects_missing_model() -> None:
    model0 = _valid_ns_model()
    reg_model = RegimeStochasticNS(models={0: model0})
    with pytest.raises(ValueError, match="missing rate models"):
        reg_model.simulate(regimes=np.array([0, 1]), x0=np.array([0.03, -0.01, 0.01]), seed=1)


def test_joint_regime_models_validate_keys() -> None:
    good = build_default_joint_regime_models()
    with pytest.raises(ValueError, match="keys must match states"):
        JointRegimeModels(
            transition_matrix=good.transition_matrix,
            rate_models={0: good.rate_models[0]},
            mortality_model=good.mortality_model,
            initial_factors=good.initial_factors,
        )
