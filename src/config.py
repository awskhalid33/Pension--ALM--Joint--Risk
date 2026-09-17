from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # Default modelling choices
    country: str = "UK"
    start_year: int = 1970
    end_year: int = 2023

settings = Settings()
