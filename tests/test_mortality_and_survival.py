import numpy as np
from src.mortality.regime_improvement import RegimeMortalityModel
from src.datasets.toy_mortality import generate_toy_mortality
from src.liabilities.survival import t_year_survival

def test_regime_mortality_reproducible():
    regimes = np.array([0, 0, 1, 1, 0], dtype=int)
    model = RegimeMortalityModel(mu={0: -0.01, 1: -0.005}, sigma={0: 0.01, 1: 0.02})

    k1 = model.simulate(regimes=regimes, kappa0=0.0, seed=777)
    k2 = model.simulate(regimes=regimes, kappa0=0.0, seed=777)

    assert np.allclose(k1, k2)

def test_survival_decreases_with_time():
    df = generate_toy_mortality(seed=123)
    base_year = 2023
    base_age = 55

    s5 = t_year_survival(df, base_year=base_year, base_age=base_age, t=5, kappa_shift=0.0)
    s10 = t_year_survival(df, base_year=base_year, base_age=base_age, t=10, kappa_shift=0.0)

    assert 0.0 < s10 <= s5 <= 1.0

def test_negative_kappa_improves_survival():
    df = generate_toy_mortality(seed=123)
    base_year = 2023
    base_age = 55
    t = 10

    s_base = t_year_survival(df, base_year=base_year, base_age=base_age, t=t, kappa_shift=0.0)
    s_improved = t_year_survival(df, base_year=base_year, base_age=base_age, t=t, kappa_shift=-0.10)

    assert s_improved > s_base
