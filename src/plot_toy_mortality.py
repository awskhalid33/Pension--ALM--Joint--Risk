from __future__ import annotations

from src.config import settings
from src.io_utils import ensure_dirs_exist
from src.plotting import get_pyplot
from src.sim.liability_setup import load_toy_mortality


def main() -> None:
    plt = get_pyplot()
    ensure_dirs_exist([settings.output_dir])

    df = load_toy_mortality()

    ages_to_plot = [55, 65, 75, 85]
    for age in ages_to_plot:
        sub = df[df["age"] == age].sort_values("year")
        plt.plot(sub["year"], sub["log_mx"], label=f"age {age}")

    plt.title("Toy mortality: log m_x over time")
    plt.xlabel("Year")
    plt.ylabel("log m_x")
    plt.legend()
    out_path = settings.output_dir / "toy_mortality_logmx_over_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
