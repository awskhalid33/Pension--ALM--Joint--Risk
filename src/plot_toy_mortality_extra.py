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
        plt.plot(sub["year"], sub["mx"], label=f"age {age}")

    plt.title("Toy mortality: m_x over time")
    plt.xlabel("Year")
    plt.ylabel("m_x")
    plt.legend()
    out1 = settings.output_dir / "toy_mortality_mx_over_time.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    for age in ages_to_plot:
        sub = df[df["age"] == age].sort_values("year").copy()
        sub["delta_log_mx"] = sub["log_mx"].diff()
        plt.plot(sub["year"], sub["delta_log_mx"], label=f"age {age}")

    plt.axhline(0.0)
    plt.title("Toy mortality: year-on-year improvement (delta log m_x)")
    plt.xlabel("Year")
    plt.ylabel("delta log m_x")
    plt.legend()
    out2 = settings.output_dir / "toy_mortality_improvement_delta_logmx.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plot:", out1)
    print("Saved plot:", out2)


if __name__ == "__main__":
    main()
