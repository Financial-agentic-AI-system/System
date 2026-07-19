"""CLI for loading datalake data into Postgres.

Examples:
    python -m src.db.load                      # create tables and load everything
    python -m src.db.load --datasets fundamentals prices
    python -m src.db.load --datalake ../Datalake --no-init
"""

import argparse

from src.db.init_db import init_db
from src.db.loaders import LOADERS, resolve_raw_data
from src.db.session import get_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the datalake into Postgres.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*LOADERS.keys(), "all"],
        default=["all"],
        help="Datasets to load (default: all).",
    )
    parser.add_argument(
        "--datalake",
        default=None,
        help="Path to the datalake directory (overrides autodetection).",
    )
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="Do not create tables before loading.",
    )
    args = parser.parse_args()

    raw_data = resolve_raw_data(args.datalake)
    print(f"Datalake: {raw_data}")

    if not args.no_init:
        init_db()
        print("Database schema ready.")

    datasets = list(LOADERS.keys()) if "all" in args.datasets else args.datasets

    session = get_session()
    try:
        for name in datasets:
            count = LOADERS[name](session, raw_data)
            print(f"  {name:<13} -> {count} rows")
    finally:
        session.close()

    print("Done.")


if __name__ == "__main__":
    main()
