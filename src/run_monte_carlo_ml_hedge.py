from __future__ import annotations

import argparse

import numpy as np

from src.config import PROJECT_ROOT, settings
from src.io_utils import ensure_dirs_exist
from src.liabilities.valuation import (
    duration_remaining_cashflows_at_time_k,
)
from src.plotting import get_pyplot
from src.runtime_utils import add_common_simulation_args, write_run_metadata
from src.sim.liability_setup import default_liability_spec, load_toy_mortality
from src.sim.hedge_accounting import funding_path, projected_cashflows
from src.sim.paths import simulate_joint_paths
from src.sim.scenario import build_default_joint_regime_models


def fit_linear_regression(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    xtx = X.T @ X
    eye = np.eye(xtx.shape[0])
    return np.linalg.solve(xtx + ridge * eye, X.T @ y)


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ beta




def simulate_static_fr(**kwargs) -> np.ndarray:
    return funding_path(**kwargs)


def duration_features(factors, kappas, k):
    return np.array([1.0, factors.level, factors.slope, factors.curvature,
                     float(kappas[k]), 0.0 if k == 0 else float(kappas[k] - kappas[k - 1]),
                     float(k)])


def simulate_ml_duration_hedge_fr(*, beta, h_floor, h_cap, **kwargs) -> np.ndarray:
    def policy(k, cf, model, factors):
        # beta=None is the exact-duration benchmark, with identical limits.
        duration = (duration_remaining_cashflows_at_time_k(cf, k, model, factors)
                    if beta is None else float(predict(
                        beta, duration_features(factors, kwargs["kappas"], k))))
        return np.clip(duration / (kwargs["hedge_maturity"] - k), h_floor, h_cap)
    return funding_path(**kwargs, ratio_policy=policy)


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
    parser = argparse.ArgumentParser(description="Monte Carlo: static vs ML-duration hedge.")
    add_common_simulation_args(
        parser,
        default_n_years=10,
        include_n_paths=True,
        default_n_paths=1000,
        default_seed=40_000,
    )
    parser.add_argument("--hedge-maturity", type=float, default=25.0)
    parser.add_argument("--n-train-paths", type=int, default=400)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--h-floor", type=float, default=0.5)
    parser.add_argument("--h-cap", type=float, default=1.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_paths <= 0 or args.n_train_paths <= 0:
        parser.error("Training and evaluation path counts must be positive")
    if max(30_000, args.seed) < min(30_000 + args.n_train_paths, args.seed + args.n_paths):
        parser.error("Training and evaluation seed ranges must not overlap")
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    models = build_default_joint_regime_models(dt_years=1.0)

    spec = default_liability_spec()

    x_rows = []
    y_rows = []
    for i in range(args.n_train_paths):
        regimes, x_path, kappas = simulate_joint_paths(
            n_years=args.n_years,
            models=models,
            seed=30_000 + i,
        )
        for k in range(args.n_years + 1):
            s = int(regimes[k])
            mk = models.rate_models[s]
            fk = mk.factors_from_array(x_path[k])

            cfk = projected_cashflows(df_mort, spec, kappas, k)
            d_true = duration_remaining_cashflows_at_time_k(
                cf=cfk,
                k=k,
                curve_model=mk,
                curve_factors=fk,
            )

            x_rows.append(duration_features(fk, kappas, k))
            y_rows.append(d_true)

    x = np.array(x_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    beta = fit_linear_regression(x, y, ridge=args.ridge)
    print("Trained linear model coefficients (beta):")
    print(beta)

    frt_static = np.zeros(args.n_paths, dtype=float)
    frmin_static = np.zeros(args.n_paths, dtype=float)
    frt_ml = np.zeros(args.n_paths, dtype=float)
    frmin_ml = np.zeros(args.n_paths, dtype=float)

    frt_exact = np.zeros(args.n_paths)
    frmin_exact = np.zeros(args.n_paths)

    for i in range(args.n_paths):
        regimes, x_path, kappas = simulate_joint_paths(
            n_years=args.n_years,
            models=models,
            seed=args.seed + i,
        )

        fr_s = simulate_static_fr(
            df_mort=df_mort,
            n_years=args.n_years,
            hedge_maturity=args.hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            models=models,
        )
        fr_m = simulate_ml_duration_hedge_fr(
            df_mort=df_mort,
            n_years=args.n_years,
            hedge_maturity=args.hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            models=models,
            beta=beta,
            h_floor=args.h_floor,
            h_cap=args.h_cap,
        )

        fr_e = simulate_ml_duration_hedge_fr(
            df_mort=df_mort, n_years=args.n_years, hedge_maturity=args.hedge_maturity,
            regimes=regimes, x_path=x_path, kappas=kappas, models=models,
            beta=None, h_floor=args.h_floor, h_cap=args.h_cap,
        )
        frt_exact[i] = fr_e[-1]
        frmin_exact[i] = np.nanmin(fr_e)

        frt_static[i] = fr_s[-1]
        frmin_static[i] = np.nanmin(fr_s)
        frt_ml[i] = fr_m[-1]
        frmin_ml[i] = np.nanmin(fr_m)

    exact_t = summarize(frt_exact)
    exact_min = summarize(frmin_exact)
    print("Exact-duration benchmark:", exact_t, "minimum FR:", exact_min)
    static_t = summarize(frt_static)
    ml_t = summarize(frt_ml)
    static_min = summarize(frmin_static)
    ml_min = summarize(frmin_ml)

    print("\n=== Static hedge: FR at horizon ===")
    for k, v in static_t.items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== ML-duration hedge: FR at horizon ===")
    for k, v in ml_t.items():
        print(f"{k:>10}: {v:.4f}")

    print(f"\n=== Static hedge: min FR over {args.n_years}y ===")
    for k, v in static_min.items():
        print(f"{k:>10}: {v:.4f}")

    print(f"\n=== ML-duration hedge: min FR over {args.n_years}y ===")
    for k, v in ml_min.items():
        print(f"{k:>10}: {v:.4f}")

    if not args.no_plots:
        plt.hist(frmin_static, bins=40, alpha=0.6, label="static")
        plt.hist(frmin_ml, bins=40, alpha=0.6, label="ML-duration")
        plt.hist(frmin_exact, bins=40, alpha=0.4, label="exact-duration")
        plt.title(f"Minimum funding ratio over {args.n_years}y: static vs ML-duration hedge")
        plt.xlabel("min FR")
        plt.ylabel("Count")
        plt.legend()
        p1 = settings.output_dir / "compare_hist_frmin_static_vs_ml_duration.png"
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print("\nSaved plot:", p1)

    summary = {
        "n_paths": int(args.n_paths),
        "n_years": int(args.n_years),
        "n_train_paths": int(args.n_train_paths),
        "ridge": float(args.ridge),
        "static_horizon": static_t, "ml_horizon": ml_t, "exact_horizon": exact_t,
        "static_minimum": static_min, "ml_minimum": ml_min, "exact_minimum": exact_min,
        "static_frT_mean": static_t["mean"],
        "ml_frT_mean": ml_t["mean"],
        "static_min_mean": static_min["mean"],
        "ml_min_mean": ml_min["mean"],
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="mc_ml_hedge",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()

