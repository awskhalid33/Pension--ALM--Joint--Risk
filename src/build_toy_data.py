from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.datasets.toy_mortality import generate_toy_mortality
from src.io_utils import ensure_dirs_exist
from src.runtime_utils import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate toy mortality dataset.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-year", type=int, default=settings.start_year)
    parser.add_argument("--end-year", type=int, default=settings.end_year)
    parser.add_argument("--min-age", type=int, default=50)
    parser.add_argument("--max-age", type=int, default=110)
    parser.add_argument("--metadata-tag", type=str, default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ensure_dirs_exist([settings.processed_dir, settings.output_dir])

    df = generate_toy_mortality(
        ages=range(args.min_age, args.max_age + 1),
        years=range(args.start_year, args.end_year + 1),
        seed=args.seed,
    )

    out_path: Path = settings.processed_dir / "toy_mortality_uk.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print(df.head(10).to_string(index=False))

    summary = {
        "rows": int(len(df)),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "min_age": int(args.min_age),
        "max_age": int(args.max_age),
    }
    metadata_path = write_run_metadata(
        output_dir=settings.output_dir,
        run_name="build_toy_data",
        args=args,
        summary=summary,
        project_root=PROJECT_ROOT,
    )
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
