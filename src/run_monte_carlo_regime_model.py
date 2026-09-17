from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.io_utils import ensure_dirs_exist
from src.plotting import get_pyplot
from src.runtime_utils import add_common_simulation_args, write_run_metadata
from src.sim.liability_setup import load_toy_mortality
from src.sim.paths import simulate_joint_paths
from src.sim.scenario import JointRegimeModels, build_default_joint_regime_models


def simulate_one_path(
    *,
    df_mort: pd.DataFrame,
    n_years: int,
    hedge_maturity: float,
    models: JointRegimeModels,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      funding_ratio_path (n_years+1,)
      regimes (n_years+1,)
      kappas (n_years+1,)
    """
    regimes, x_path, kappas = simulate_joint_paths(n_years=n_years, models=models, seed=seed)

    from src.sim.hedge_accounting import funding_path
    fr = funding_path(df_mort=df_mort, n_years=n_years, hedge_maturity=hedge_maturity,
                      regimes=regimes, x_path=x_path, kappas=kappas, models=models)

    return fr, regimes, kappas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo: joint regime rates + longevity model.")
    add_common_simulation_args(
        parser,
        default_n_years=10,
        include_n_paths=True,
        default_n_paths=1000,
        default_seed=10_000,
    )
    parser.add_argument(
        "--hedge-maturity",
        type=float,
        default=25.0,
        help="Maturity (years) of the hedging zero-coupon asset.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    models = build_default_joint_regime_models(dt_years=1.0)

    fr_t = np.zeros(args.n_paths, dtype=float)
    fr_min = np.zeros(args.n_paths, dtype=float)
    stress_frac = np.zeros(args.n_paths, dtype=float)

    for i in range(args.n_paths):
        fr, regimes, _ = simulate_one_path(
            df_mort=df_mort,
            n_years=args.n_years,
            hedge_maturity=args.hedge_maturity,
            models=models,
            seed=args.seed + i,
        )
        fr_t[i] = fr[-1]
        fr_min[i] = np.nanmin(fr)
        stress_frac[i] = float(np.mean(regimes == 1))

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    summary = {
        "n_paths": int(args.n_paths),
        "n_years": int(args.n_years),
        "hedge_maturity": float(args.hedge_maturity),
        "FR_T_mean": float(np.mean(fr_t)),
        "FR_T_p5": pct(fr_t, 5),
        "FR_T_p50": pct(fr_t, 50),
        "FR_T_p95": pct(fr_t, 95),
        "FR_min_mean": float(np.mean(fr_min)),
        "Pr(FR_T < 0.95)": float(np.mean(fr_t < 0.95)),
        "Pr(FR_T < 0.90)": float(np.mean(fr_t < 0.90)),
        "Pr(min FR < 0.95)": float(np.mean(fr_min < 0.95)),
        "Pr(min FR < 0.90)": float(np.mean(fr_min < 0.90)),
        "stress_frac_mean": float(np.mean(stress_frac)),
    }

    print("=== Monte Carlo Summary (Regime joint model) ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:>18}: {value:.4f}")
        else:
            print(f"{key:>18}: {value}")

    out_df = pd.DataFrame(
        {
            "fr_T": fr_t,
            "fr_min": fr_min,
            "stress_frac": stress_frac,
        }
    )
    csv_path = settings.output_dir / "mc_regime_joint_results.csv"
    out_df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    if not args.no_plots:
        plt.hist(fr_t, bins=40)
        plt.title(f"Funding ratio at year {args.n_years} (Regime joint model)")
        plt.xlabel("FR_T")
        plt.ylabel("Count")
        p1 = settings.output_dir / "hist_fr_T_regime_joint.png"
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", p1)

        plt.hist(fr_min, bins=40)
        plt.title(f"Minimum funding ratio over {args.n_years}y (Regime joint model)")
        plt.xlabel("min FR")
        plt.ylabel("Count")
        p2 = settings.output_dir / "hist_fr_min_regime_joint.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", p2)

    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="mc_regime_joint",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()

