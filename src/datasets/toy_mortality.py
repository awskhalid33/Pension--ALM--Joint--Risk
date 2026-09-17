from __future__ import annotations

import numpy as np
import pandas as pd


def generate_toy_mortality(
    ages: range = range(50, 101),
    years: range = range(1970, 2024),
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate a synthetic mortality surface with:
    - Gompertz-like age pattern
    - gradual improvement over time
    - a macro-regime-like shock period (e.g. crisis / pandemic style)

    Output columns:
    year, age, mx, log_mx
    """
    rng = np.random.default_rng(seed)

    age_arr = np.array(list(ages), dtype=int)
    year_arr = np.array(list(years), dtype=int)

    # Gompertz-like baseline: log m_x ~ a + b * (age - 50)
    a = -6.8
    b = 0.085

    # Improvement trend: log mortality decreases over time
    # Around 1.5% per year in hazard terms is plausible-ish for a toy.
    improvement_per_year = -0.015

    # Regime shock window (toy): mortality temporarily higher
    shock_start, shock_end = 2008, 2010
    shock_log_add = 0.08

    rows = []
    for y in year_arr:
        t = y - year_arr[0]

        # time effect: improvement
        time_effect = improvement_per_year * t

        # shock
        shock = shock_log_add if (shock_start <= y <= shock_end) else 0.0

        # small year noise
        year_noise = rng.normal(0.0, 0.02)

        for x in age_arr:
            log_mx = a + b * (x - 50) + time_effect + shock + year_noise

            # add small idiosyncratic noise by age
            log_mx += rng.normal(0.0, 0.01)

            mx = float(np.exp(log_mx))

            rows.append((y, x, mx, log_mx))

    df = pd.DataFrame(rows, columns=["year", "age", "mx", "log_mx"])
    return df
