import json
from pathlib import Path

import click
import pandas as pd

DEFAULT_INPUT_FILE = Path("../datalake/raw_data/stock_prices_daily_from_2010.csv")
OUTPUT_BASE_DIR = Path("../datalake/raw_data/stock_price")


def load_stock_csv(input_path: Path) -> pd.DataFrame:
    """Load the source CSV produced by the stock prices downloader."""
    df = pd.read_csv(input_path, header=[0, 1, 2], index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        df.columns = pd.MultiIndex.from_tuples([(a, b) for a, b, _ in df.columns])

    return df


def save_stock_json(
    ticker: str, price_series: pd.Series, price_field: str, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for date, value in price_series.items():
        if pd.notna(value):
            reports.append(
                {"date": date.strftime("%Y-%m-%d"), price_field.lower(): str(value)}
            )

    payload = {"symbol": ticker, "period": "daily", "reports": reports}

    output_file = output_dir / f"{ticker}_daily.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    click.secho(f"  -> Saved: {output_file.resolve()}", fg="green")


@click.command()
@click.option(
    "--input",
    "input_file",
    default=DEFAULT_INPUT_FILE,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the source CSV file with downloaded stock prices.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=OUTPUT_BASE_DIR,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where per-stock JSON files will be written.",
)
@click.option(
    "--tickers",
    "tickers",
    default=None,
    help="Comma-separated list of tickers to export. By default all tickers are processed.",
)
@click.option(
    "--price-field",
    "price_field",
    default="Close",
    type=click.Choice(["Open", "High", "Low", "Close", "Volume"], case_sensitive=False),
    help="Which price field to export for each ticker.",
)
def main(input_file: Path, output_dir: Path, tickers: str, price_field: str):
    """Convert a multi-ticker stock prices CSV into per-symbol JSON files."""
    df = load_stock_csv(input_file)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(lambda tup: (tup[0].strip(), tup[1].strip()))
    else:
        raise click.ClickException(
            "Expected a MultiIndex CSV with price type and ticker in headers."
        )

    available_tickers = sorted({ticker for _, ticker in df.columns})
    click.secho(f"Found tickers: {', '.join(available_tickers)}", fg="cyan")

    if tickers:
        requested = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        requested = available_tickers

    for ticker in requested:
        matched_cols = [
            col
            for col in df.columns
            if col[0].lower() == price_field.lower() and col[1].upper() == ticker
        ]
        if not matched_cols:
            click.secho(
                f"  -> WARNING: {ticker} with field {price_field} not found",
                fg="yellow",
            )
            continue

        price_series = df[matched_cols[0]].copy()
        save_stock_json(ticker, price_series, price_field, output_dir)

    click.secho(
        "\nFinished converting CSV to per-stock JSON files.", fg="green", bold=True
    )


if __name__ == "__main__":
    main()
