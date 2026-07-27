import datetime
from pathlib import Path

import click
import yfinance as yf

DEFAULT_TICKERS = "AAPL,NVDA,GS,CAT,TSLA,META,NFLX,SMCI,META,INTC"
OUTPUT_BASE_DIR = Path("../datalake/raw_data")


def fetch_stock_data(
    tickers: list[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    interval: str = "1d",
) -> dict:
    """
    Fetches historical stock price data from Yahoo Finance.

    Args:
        tickers: List of stock tickers
        start_date: Start date for historical data
        end_date: End date for historical data
        interval: Data interval ('1d', '1h', '1m', etc.)

    Returns:
        Dictionary with ticker as key and DataFrame as value
    """
    click.echo(f"  Fetching stock data for {len(tickers)} tickers...")

    try:
        data = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
        )

        if data.empty:
            click.secho("  -> WARNING: No data retrieved", fg="yellow")
            return {}

        return data
    except Exception as e:
        click.secho(f"  -> ERROR fetching data: {str(e)}", fg="red")
        return {}


def save_to_csv(data, output_file: str):
    """
    Saves stock price data to CSV file.

    Args:
        data: DataFrame with stock data
        output_file: Path and filename for output CSV
    """
    try:
        output_path = OUTPUT_BASE_DIR / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(output_path)
        click.secho(f"  -> Saved: {output_path.resolve()}", fg="green")
    except Exception as e:
        click.secho(f"  -> ERROR saving file: {str(e)}", fg="red")


@click.command()
@click.option(
    "--tickers",
    "-t",
    default=DEFAULT_TICKERS,
    help="List of stock tickers (comma-separated)",
)
@click.option(
    "--start",
    "-s",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2024-01-01",
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end",
    "-e",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=datetime.date.today().strftime("%Y-%m-%d"),
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--interval",
    "-i",
    type=click.Choice(
        ["1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"], case_sensitive=False
    ),
    default="1d",
    help="Data interval (1d for daily, 1h for hourly, etc.)",
)
@click.option(
    "--output", "-o", default="stock_prices.csv", help="Output CSV filename in datalake"
)
def main(
    tickers: str,
    start: datetime.datetime,
    end: datetime.datetime,
    interval: str,
    output: str,
):
    """
    Fetch historical stock price data from Yahoo Finance and save to CSV.

    Examples:
        # Fetch default tickers for daily prices
        python yfinance_fetcher.py

        # Fetch specific tickers with custom dates
        python yfinance_fetcher.py -t "AAPL,MSFT,GOOGL" -s 2023-01-01 -e 2024-12-31

        # Fetch hourly data
        python yfinance_fetcher.py -i 1h -o stock_prices_hourly.csv
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    click.secho("=== Yahoo Finance Stock Data Fetcher ===", fg="cyan", bold=True)
    click.secho(
        f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", fg="cyan"
    )
    click.secho(f"Interval: {interval}", fg="cyan")
    click.secho(f"Tickers: {', '.join(ticker_list)}\n", fg="cyan")

    data = fetch_stock_data(ticker_list, start, end, interval)

    if not data.empty:
        save_to_csv(data, output)

        click.secho("\n✓ Data fetched successfully!", fg="green", bold=True)
        click.echo(f"  Records: {len(data)}")
        click.echo(f"  Columns: {list(data.columns)}")
    else:
        click.secho("\n✗ Failed to fetch data", fg="red", bold=True)


if __name__ == "__main__":
    main()
