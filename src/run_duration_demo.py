from __future__ import annotations

from src.finance.duration import (
    macaulay_duration_from_curve,
    modified_duration_from_curve,
)
from src.finance.yield_curve import NelsonSiegelCurve
from src.liabilities.db_liability import build_expected_cashflows
from src.sim.liability_setup import default_liability_spec, load_toy_mortality


def main() -> None:
    df_mort = load_toy_mortality()
    spec = default_liability_spec()
    cf = build_expected_cashflows(df_mort, spec)

    curve = NelsonSiegelCurve()

    D_mac = macaulay_duration_from_curve(cf, curve)
    D_mod = modified_duration_from_curve(cf, curve)

    print("Macaulay duration (years):", round(D_mac, 2))
    print("Modified duration:", round(D_mod, 2))


if __name__ == "__main__":
    main()
