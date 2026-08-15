import json
import os
import time
from pathlib import Path

import click
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

DEFAULT_TICKERS = "AAPL,CAT,GS,INTC,META,NFLX,NVDA,SMCI,TSLA"
OUTPUT_BASE_DIR = Path("../datalake/raw_data")


def fetch_data(ticker: str, function_name: str) -> dict | None:
    click.echo(f"  Fetching: {function_name} for {ticker}...")
    param_name = "tickers" if function_name == "NEWS_SENTIMENT" else "symbol"
    url = f"https://www.alphavantage.co/query?function={function_name}&{param_name}={ticker}&apikey={API_KEY}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


def save_to_json(data: dict, ticker: str, function_name: str, period: str = "all"):
    """
    Saves data. If fundamental data, allows filtering by annual/quarterly.
    """
    period_map = {"annual": "annualReports", "quarterly": "quarterlyReports"}

    dir_path = OUTPUT_BASE_DIR / function_name
    dir_path.mkdir(parents=True, exist_ok=True)

    if period in period_map and period_map[period] in data:
        data_to_save = {
            "symbol": data.get("symbol", ticker),
            "period": period,
            "reports": data[period_map[period]],
        }
        file_path = dir_path / f"{ticker}_{period}.json"
    else:
        data_to_save = data
        file_path = dir_path / f"{ticker}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)

    click.secho(f"  -> Saved: {file_path.resolve()}", fg="green")


@click.command()
@click.option(
    "--functions", "-f", required=True, help="API functions, e.g., INCOME_STATEMENT"
)
@click.option("--tickers", "-t", default=DEFAULT_TICKERS, help="Tickers list")
@click.option(
    "--period",
    "-p",
    type=click.Choice(["annual", "quarterly", "all"], case_sensitive=False),
    default="all",
    help="Filter for fundamental data",
)
def main(functions: str, tickers: str, period: str):
    if not API_KEY:
        raise click.ClickException("Missing API key!")

    func_list = [f.strip() for f in functions.split(",")]
    ticker_list = [t.strip() for t in tickers.split(",")]

    for ticker in ticker_list:
        click.secho(f"--- Stock: {ticker} ---", fg="cyan", bold=True)
        for func in func_list:
            data = fetch_data(ticker, func)

            if data and "Information" not in data and "Note" not in data:
                save_to_json(data, ticker, func, period)
            elif data:
                error_msg = data.get("Information", data.get("Note", "API error"))
                click.secho(f"  -> WARNING: {error_msg}", fg="red")

            time.sleep(15)


if __name__ == "__main__":
    main()
