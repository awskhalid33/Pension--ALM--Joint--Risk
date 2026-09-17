import numpy as np
from src.regimes.markov import MarkovRegime

def test_markov_simulation_shape_and_values():
    P = np.array([[0.9, 0.1],
                  [0.2, 0.8]], dtype=float)

    proc = MarkovRegime(P)
    states = proc.simulate(n_steps=10, s0=0, seed=123)

    assert states.shape == (11,)
    assert set(np.unique(states)).issubset({0, 1})

def test_markov_reproducible_with_seed():
    P = np.array([[0.9, 0.1],
                  [0.2, 0.8]], dtype=float)

    proc = MarkovRegime(P)
    s1 = proc.simulate(n_steps=25, s0=0, seed=999)
    s2 = proc.simulate(n_steps=25, s0=0, seed=999)

    assert np.array_equal(s1, s2)
