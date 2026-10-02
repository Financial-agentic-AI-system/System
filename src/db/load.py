"""CLI for loading datalake data into Postgres.

Examples:
    python -m src.db.load                      # create tables and load everything
    python -m src.db.load --datasets fundamentals prices
    python -m src.db.load --datalake ../Datalake --no-init
    python -m src.db.load --datasets articles  # (re)load + embed new articles
    python -m src.db.load --no-embed           # skip the embedding step

Articles without an embedding are vectorized after loading; skipped when
GCP_PROJECT_ID is not set.
"""

import argparse

from dotenv import load_dotenv

from src.db.article_embeddings import embed_article_summaries
from src.db.init_db import init_db
from src.db.loaders import LOADERS, resolve_raw_data
from src.db.session import get_session
from src.retriever import embeddings


def embed_articles(session) -> None:
    """Vectorize articles that have no embedding yet."""
    if not embeddings.is_configured():
        print("  embeddings    -> skipped (GCP_PROJECT_ID is not set)")
        return
    print(f"  embeddings    -> calling {embeddings.model_name()} ...")
    try:
        counts = embed_article_summaries(session)
    except Exception as exc:
        session.rollback()
        print(f"  embeddings    -> FAILED: {exc}")
        print("                   Re-run with `--datasets articles` to resume.")
        return
    for column, count in counts.items():
        print(f"  {column:<24} -> {count} new vectors")


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
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Do not vectorize the article summaries after loading them.",
    )
    args = parser.parse_args()
    load_dotenv()

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
        if "articles" in datasets and not args.no_embed:
            embed_articles(session)
    finally:
        session.close()

    print("Done.")


if __name__ == "__main__":
    main()
