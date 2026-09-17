from __future__ import annotations

from src.finance.bonds import ZeroCouponBond
from src.finance.duration import macaulay_duration_from_curve
from src.finance.yield_curve import NelsonSiegelCurve
from src.liabilities.db_liability import build_expected_cashflows
from src.sim.liability_setup import default_liability_spec, load_toy_mortality


def main() -> None:
    df_mort = load_toy_mortality()
    spec = default_liability_spec()
    cf = build_expected_cashflows(df_mort, spec)

    curve = NelsonSiegelCurve()
    D_L = macaulay_duration_from_curve(cf, curve)

    # Asset: single zero-coupon bond
    bond = ZeroCouponBond(maturity=25.0)
    D_A = bond.duration()

    print("Liability duration:", round(D_L, 2))
    print("Asset duration (zero):", round(D_A, 2))
    print("Duration mismatch:", round(D_A - D_L, 2))


if __name__ == "__main__":
    main()
