from __future__ import annotations

import numpy as np


class MarkovRegime:
    """
    Discrete-time Markov regime process.
    """

    def __init__(self, transition_matrix: np.ndarray):
        p = np.asarray(transition_matrix, dtype=float)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError("Transition matrix must be a square 2D array")
        if p.shape[0] < 1:
            raise ValueError("Transition matrix must have at least one state")
        if not np.all(np.isfinite(p)):
            raise ValueError("Transition matrix must contain finite values")
        if np.any(p < 0.0):
            raise ValueError("Transition probabilities must be non-negative")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("Each transition-matrix row must sum to 1")

        self.P = p
        self.n_states = p.shape[0]

    def simulate(self, n_steps: int, s0: int, seed: int = 123) -> np.ndarray:
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        if s0 < 0 or s0 >= self.n_states:
            raise ValueError("s0 must be a valid state index")

        rng = np.random.default_rng(seed)
        states = np.zeros(n_steps + 1, dtype=int)
        states[0] = s0

        for t in range(n_steps):
            states[t + 1] = rng.choice(
                self.n_states,
                p=self.P[states[t]],
            )
        return states
