from __future__ import annotations

import argparse

import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.io_utils import ensure_dirs_exist
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k
from src.plotting import get_pyplot
from src.runtime_utils import add_common_simulation_args, write_run_metadata
from src.sim.liability_setup import default_liability_spec, load_toy_mortality
from src.sim.paths import simulate_joint_paths
from src.sim.scenario import build_default_joint_regime_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-path regime joint demo.")
    add_common_simulation_args(
        parser,
        default_n_years=10,
        include_n_paths=False,
        default_seed=42,
    )
    parser.add_argument("--hedge-maturity", type=float, default=25.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    spec = default_liability_spec()
    models = build_default_joint_regime_models(dt_years=1.0)

    regimes, x_path, kappas = simulate_joint_paths(
        n_years=args.n_years,
        models=models,
        seed=args.seed,
    )

    from src.sim.hedge_accounting import funding_path, projected_cashflows
    funding = funding_path(df_mort=df_mort, n_years=args.n_years,
                           hedge_maturity=args.hedge_maturity, regimes=regimes,
                           x_path=x_path, kappas=kappas, models=models)

    records = []
    for k in range(0, args.n_years + 1):
        s = int(regimes[k])
        mk = models.rate_models[s]
        fk = mk.factors_from_array(x_path[k])

        cf_k = projected_cashflows(df_mort, spec, kappas, k)
        lk = pv_remaining_cashflows_at_time_k(cf=cf_k, k=k, curve_model=mk, curve_factors=fk)
        fr = funding[k]
        ak = fr * lk

        records.append((k, s, kappas[k], fk.level, ak, lk, fr))

    out = pd.DataFrame(
        records,
        columns=["year", "regime", "kappa", "level", "asset", "liability", "funding_ratio"],
    )
    print(out.to_string(index=False))

    if not args.no_plots:
        plt.plot(out["year"], out["funding_ratio"])
        plt.axhline(1.0)
        plt.title("Funding ratio under regimes (rates + longevity, static hedge)")
        plt.xlabel("Year")
        plt.ylabel("Funding ratio")
        p1 = settings.output_dir / "funding_ratio_regime_joint.png"
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()

        plt.step(out["year"], out["regime"], where="post")
        plt.title("Regime path (0=normal, 1=stress)")
        plt.xlabel("Year")
        plt.ylabel("Regime")
        p2 = settings.output_dir / "regime_path.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()

        plt.plot(out["year"], out["kappa"])
        plt.axhline(0.0)
        plt.title("Kappa path (negative = improving mortality)")
        plt.xlabel("Year")
        plt.ylabel("kappa")
        p3 = settings.output_dir / "kappa_path_regime.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()

        print("Saved:", p1)
        print("Saved:", p2)
        print("Saved:", p3)

    summary = {
        "n_years": int(args.n_years),
        "seed": int(args.seed),
        "fr_start": float(out["funding_ratio"].iloc[0]),
        "fr_end": float(out["funding_ratio"].iloc[-1]),
        "regime1_fraction": float((out["regime"] == 1).mean()),
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="regime_joint_demo",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()

