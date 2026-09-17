from __future__ import annotations

import argparse

import numpy as np

from src.config import PROJECT_ROOT, settings
from src.io_utils import ensure_dirs_exist
from src.plotting import get_pyplot
from src.runtime_utils import add_common_simulation_args, write_run_metadata
from src.sim.liability_setup import load_toy_mortality
from src.sim.hedge_accounting import funding_path
from src.sim.paths import simulate_joint_paths
from src.sim.scenario import build_default_joint_regime_models




def simulate_static_hedge_path(**kwargs) -> np.ndarray:
    return funding_path(**kwargs)


def simulate_dynamic_hedge_path(*, h_normal, h_stress, **kwargs) -> np.ndarray:
    regimes = kwargs["regimes"]
    return funding_path(**kwargs, ratio_policy=lambda k, cf, model, factors:
                        h_stress if regimes[k] == 1 else h_normal)


def summarize(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "p5": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "Pr(<0.95)": float(np.mean(x < 0.95)),
        "Pr(<0.90)": float(np.mean(x < 0.90)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo: static vs regime-switching dynamic hedge.")
    add_common_simulation_args(
        parser,
        default_n_years=10,
        include_n_paths=True,
        default_n_paths=1000,
        default_seed=20_000,
    )
    parser.add_argument("--hedge-maturity", type=float, default=25.0)
    parser.add_argument("--h-normal", type=float, default=1.00, help="Dynamic hedge ratio in regime 0.")
    parser.add_argument("--h-stress", type=float, default=1.10, help="Dynamic hedge ratio in regime 1.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    models = build_default_joint_regime_models(dt_years=1.0)

    frt_static = np.zeros(args.n_paths, dtype=float)
    frmin_static = np.zeros(args.n_paths, dtype=float)
    frt_dyn = np.zeros(args.n_paths, dtype=float)
    frmin_dyn = np.zeros(args.n_paths, dtype=float)

    for i in range(args.n_paths):
        regimes, x_path, kappas = simulate_joint_paths(
            n_years=args.n_years,
            models=models,
            seed=args.seed + i,
        )

        fr_s = simulate_static_hedge_path(
            df_mort=df_mort,
            n_years=args.n_years,
            hedge_maturity=args.hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            models=models,
        )
        fr_d = simulate_dynamic_hedge_path(
            df_mort=df_mort,
            n_years=args.n_years,
            hedge_maturity=args.hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            models=models,
            h_normal=args.h_normal,
            h_stress=args.h_stress,
        )

        frt_static[i] = fr_s[-1]
        frmin_static[i] = np.nanmin(fr_s)
        frt_dyn[i] = fr_d[-1]
        frmin_dyn[i] = np.nanmin(fr_d)

    static_t = summarize(frt_static)
    dyn_t = summarize(frt_dyn)
    static_min = summarize(frmin_static)
    dyn_min = summarize(frmin_dyn)

    print("=== Static hedge: FR at horizon ===")
    for k, v in static_t.items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Dynamic hedge: FR at horizon ===")
    for k, v in dyn_t.items():
        print(f"{k:>10}: {v:.4f}")

    print(f"\n=== Static hedge: min FR over {args.n_years}y ===")
    for k, v in static_min.items():
        print(f"{k:>10}: {v:.4f}")

    print(f"\n=== Dynamic hedge: min FR over {args.n_years}y ===")
    for k, v in dyn_min.items():
        print(f"{k:>10}: {v:.4f}")

    if not args.no_plots:
        plt.hist(frt_static, bins=40, alpha=0.6, label="static")
        plt.hist(frt_dyn, bins=40, alpha=0.6, label="dynamic")
        plt.title(f"Funding ratio at year {args.n_years}: static vs dynamic hedge")
        plt.xlabel("FR_T")
        plt.ylabel("Count")
        plt.legend()
        p1 = settings.output_dir / "compare_hist_frT_static_vs_dynamic.png"
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print("\nSaved plot:", p1)

        plt.hist(frmin_static, bins=40, alpha=0.6, label="static")
        plt.hist(frmin_dyn, bins=40, alpha=0.6, label="dynamic")
        plt.title(f"Minimum funding ratio over {args.n_years}y: static vs dynamic hedge")
        plt.xlabel("min FR")
        plt.ylabel("Count")
        plt.legend()
        p2 = settings.output_dir / "compare_hist_frmin_static_vs_dynamic.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", p2)

    summary = {
        "static_horizon": static_t, "dynamic_horizon": dyn_t,
        "static_minimum": static_min, "dynamic_minimum": dyn_min,
        "n_paths": int(args.n_paths),
        "n_years": int(args.n_years),
        "h_normal": float(args.h_normal),
        "h_stress": float(args.h_stress),
        "static_frT_mean": static_t["mean"],
        "dynamic_frT_mean": dyn_t["mean"],
        "static_min_mean": static_min["mean"],
        "dynamic_min_mean": dyn_min["mean"],
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="mc_dynamic_vs_static",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()

