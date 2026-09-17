from src.datasets.toy_mortality import generate_toy_mortality
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows

def test_cashflows_zero_before_retirement():
    df = generate_toy_mortality(seed=123)
    spec = DBLiabilitySpec(base_year=2023, x0=55, retirement_age=65, max_age=80, annual_pension=10_000.0)

    cf = build_expected_cashflows(df, spec, kappa_shift=0.0)

    # retirement at t = 10
    pre = cf[cf["t"] < 10]
    post = cf[cf["t"] >= 10]

    assert (pre["benefit"] == 0.0).all()
    assert (post["benefit"] == 10_000.0).all()
