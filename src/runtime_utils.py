from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def add_common_simulation_args(
    parser: argparse.ArgumentParser,
    *,
    default_n_years: int = 10,
    include_n_paths: bool = False,
    default_n_paths: int = 1000,
    default_seed: int = 10_000,
) -> argparse.ArgumentParser:
    parser.add_argument("--n-years", type=int, default=default_n_years, help="Projection horizon in years.")
    if include_n_paths:
        parser.add_argument("--n-paths", type=int, default=default_n_paths, help="Number of Monte Carlo paths.")
    parser.add_argument("--seed", type=int, default=default_seed, help="Base RNG seed.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run without generating plot files.",
    )
    parser.add_argument(
        "--metadata-tag",
        type=str,
        default="",
        help="Optional suffix for the metadata JSON filename.",
    )
    return parser


def _git_commit_hash(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except Exception:
        return ""


def write_run_metadata(
    *,
    output_dir: Path,
    run_name: str,
    args: argparse.Namespace,
    summary: dict[str, Any],
    project_root: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.metadata_tag}" if getattr(args, "metadata_tag", "") else ""
    out_path = output_dir / f"{run_name}_metadata{tag}.json"

    payload: dict[str, Any] = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(project_root),
        "args": vars(args),
        "summary": summary,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path
