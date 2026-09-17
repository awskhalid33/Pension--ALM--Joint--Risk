from __future__ import annotations

from src.finance.yield_curve import NelsonSiegelCurve
from src.liabilities.db_liability import (
    build_expected_cashflows,
    present_value_from_curve,
    present_value_from_expected_cashflows,
)
from src.sim.liability_setup import default_liability_spec, load_toy_mortality


def main() -> None:
    df_mort = load_toy_mortality()
    spec = default_liability_spec()
    cf = build_expected_cashflows(df_mort, spec)

    # Flat rate PV (old way)
    pv_flat = present_value_from_expected_cashflows(cf, spec.flat_discount_rate)

    # Curve PV (new way)
    curve = NelsonSiegelCurve(
        beta0=0.03,   # long-run level
        beta1=-0.01,  # slope (short end lower/higher depending sign)
        beta2=0.01,   # curvature
        tau=2.5,
    )
    pv_curve = present_value_from_curve(cf, curve)

    print("PV (flat 3%):", round(pv_flat, 2))
    print("PV (Nelson–Siegel curve):", round(pv_curve, 2))

    # Print a few discount factors for intuition
    for t in [1, 5, 10, 20, 30, 40, 50]:
        print(f"t={t:>2}  z(t)={curve.zero_rate(t):.4f}  P(0,t)={curve.discount_factor(t):.6f}")


if __name__ == "__main__":
    main()
