from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k
from src.mortality.improvement import MortalityImprovementRW
from src.plotting import get_pyplot
from src.runtime_utils import add_common_simulation_args, write_run_metadata
from src.sim.liability_setup import default_liability_spec, load_toy_mortality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-path demo: stochastic rates + longevity.")
    add_common_simulation_args(parser, default_n_years=10, default_seed=123)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--hedge-maturity", type=float, default=25.0)
    parser.add_argument("--kappa-mu", type=float, default=-0.01)
    parser.add_argument("--kappa-sigma", type=float, default=0.01)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    spec = default_liability_spec()

    if args.dt <= 0 or not np.isclose(1 / args.dt, round(1 / args.dt)):
        parser.error("dt must divide one year into whole steps")
    n_steps = int(round(args.n_years / args.dt))

    a = np.array(
        [
            [0.995, 0.000, 0.000],
            [0.000, 0.990, 0.000],
            [0.000, 0.000, 0.985],
        ],
        dtype=float,
    )
    sigma = np.diag([0.00005, 0.00008, 0.00008])
    model = StochasticNelsonSiegelModel(A=a, Sigma=sigma, tau=2.5, dt_years=args.dt)

    x0 = np.array([0.030, -0.010, 0.010], dtype=float)
    x_path = model.simulate_factors(x0=x0, n_steps=n_steps, seed=args.seed)

    mort_model = MortalityImprovementRW(mu=args.kappa_mu, sigma=args.kappa_sigma)
    kappas = mort_model.simulate(n_years=args.n_years, seed=args.seed + 876, kappa0=0.0)

    from types import SimpleNamespace
    from src.sim.hedge_accounting import funding_path, projected_cashflows
    annual_x = x_path[np.rint(np.arange(args.n_years + 1) / args.dt).astype(int)]
    funding = funding_path(df_mort=df_mort, n_years=args.n_years,
                           hedge_maturity=args.hedge_maturity,
                           regimes=np.zeros(args.n_years + 1, dtype=int),
                           x_path=annual_x, kappas=kappas,
                           models=SimpleNamespace(rate_models={0: model}))

    years = list(range(0, args.n_years + 1))
    records = []
    for k in years:
        idx = int(round(k / args.dt))
        factors = model.factors_from_array(x_path[idx])

        cf_k = projected_cashflows(df_mort, spec, kappas, k)
        l_val = pv_remaining_cashflows_at_time_k(cf=cf_k, k=k, curve_model=model, curve_factors=factors)
        fr = funding[k]
        a_val = fr * l_val
        records.append((k, kappas[k], a_val, l_val, fr))

    df = pd.DataFrame(records, columns=["year", "kappa", "asset", "liability", "funding_ratio"])
    print(df.to_string(index=False))

    if not args.no_plots:
        plt.plot(df["year"], df["funding_ratio"])
        plt.axhline(1.0)
        plt.title("Funding ratio with stochastic rates + longevity (static LDI)")
        plt.xlabel("Year")
        plt.ylabel("Funding ratio")
        out1 = settings.output_dir / "funding_ratio_joint_rates_longevity.png"
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", out1)

        plt.plot(df["year"], df["kappa"])
        plt.axhline(0.0)
        plt.title("Mortality improvement index kappa (negative = improving)")
        plt.xlabel("Year")
        plt.ylabel("kappa")
        out2 = settings.output_dir / "kappa_path.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", out2)

    summary = {
        "n_years": int(args.n_years),
        "dt": float(args.dt),
        "kappa_end": float(df["kappa"].iloc[-1]),
        "fr_end": float(df["funding_ratio"].iloc[-1]),
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="joint_rates_longevity_demo",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()

