from __future__ import annotations

import argparse

from src.config import PROJECT_ROOT, settings
from src.io_utils import ensure_dirs_exist
from src.plotting import get_pyplot
from src.runtime_utils import write_run_metadata
from src.sim.liability_setup import default_liability_spec, load_toy_mortality


def main() -> None:
    from src.liabilities.db_liability import (
        build_expected_cashflows,
        present_value_from_expected_cashflows,
    )

    parser = argparse.ArgumentParser(description="Deterministic DB liability cashflow demo.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--metadata-tag", type=str, default="")
    args = parser.parse_args()
    plt = None if args.no_plots else get_pyplot()

    ensure_dirs_exist([settings.output_dir])
    df_mort = load_toy_mortality()
    spec = default_liability_spec()

    cf = build_expected_cashflows(df_mort, spec)
    pv = present_value_from_expected_cashflows(cf, spec.flat_discount_rate)

    print("DB Liability Spec:", spec)
    print("Present Value (flat 3%):", round(pv, 2))
    print()
    print("Cashflow preview:")
    print(cf.head(15).to_string(index=False))
    print()
    print("Cashflow tail:")
    print(cf.tail(10).to_string(index=False))

    if not args.no_plots:
        plt.plot(cf["t"], cf["expected_cashflow"])
        plt.title("Expected DB pension cashflow (toy mortality)")
        plt.xlabel("t (years from valuation)")
        plt.ylabel("Expected cashflow (£)")
        out_path = settings.output_dir / "expected_db_cashflows_toy.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved plot:", out_path)

    summary = {
        "pv_flat_3pct": float(pv),
        "max_expected_cashflow": float(cf["expected_cashflow"].max()),
        "last_expected_cashflow": float(cf["expected_cashflow"].iloc[-1]),
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="liability_demo",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
