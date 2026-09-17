from __future__ import annotations

import pandas as pd

from src.config import settings
from src.liabilities.db_liability import DBLiabilitySpec


def load_toy_mortality() -> pd.DataFrame:
    return pd.read_csv(settings.processed_dir / "toy_mortality_uk.csv")


def default_liability_spec() -> DBLiabilitySpec:
    return DBLiabilitySpec(
        base_year=2023,
        x0=55,
        retirement_age=65,
        max_age=110,
        annual_pension=10_000.0,
        flat_discount_rate=0.03,
    )
