from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import build_expected_cashflows
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k
from src.plotting import get_pyplot
from src.runtime_utils import add_common_simulation_args, write_run_metadata
from src.sim.liability_setup import default_liability_spec, load_toy_mortality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-path stochastic rates liability PV demo.")
    add_common_simulation_args(parser, default_n_years=10, default_seed=123)
    parser.add_argument("--dt", type=float, default=0.25)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    spec = default_liability_spec()
    cf = build_expected_cashflows(df_mort, spec)

    n_steps = int(args.n_years / args.dt)
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

    years = list(range(0, args.n_years + 1))
    pv_series = []
    lvl_series = []
    for k in years:
        idx = int(k / args.dt)
        factors = model.factors_from_array(x_path[idx])
        pv_k = pv_remaining_cashflows_at_time_k(
            cf=cf,
            k=k,
            curve_model=model,
            curve_factors=factors,
        )
        pv_series.append(pv_k)
        lvl_series.append(factors.level)

    out = pd.DataFrame({"year": years, "pv_liability": pv_series, "level_factor": lvl_series})
    print(out.to_string(index=False))

    if not args.no_plots:
        plt.plot(out["year"], out["pv_liability"])
        plt.title("Liability PV over time under stochastic interest rates (toy)")
        plt.xlabel("Year")
        plt.ylabel("PV (£)")
        out_path = settings.output_dir / "liability_pv_stochastic_rates.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", out_path)

    summary = {
        "n_years": int(args.n_years),
        "dt": float(args.dt),
        "pv_start": float(out["pv_liability"].iloc[0]),
        "pv_end": float(out["pv_liability"].iloc[-1]),
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="stochastic_rates_demo",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
