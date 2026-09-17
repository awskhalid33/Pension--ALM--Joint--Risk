from src.config import settings
from src.io_utils import ensure_dirs_exist


def main() -> None:
    ensure_dirs_exist(
        [
            settings.data_dir,
            settings.raw_dir,
            settings.processed_dir,
            settings.output_dir,
        ]
    )

    print("Created/checked folders:")
    print("Raw:", settings.raw_dir)
    print("Processed:", settings.processed_dir)
    print("Outputs:", settings.output_dir)


if __name__ == "__main__":
    main()
