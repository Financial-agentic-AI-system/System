import json
import os
import time
from pathlib import Path

import click
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

DEFAULT_TICKERS = "AAPL,CAT,GS,INTC,META,NFLX,NVDA,SMCI,TSLA,PLTR"
OUTPUT_BASE_DIR = Path("../datalake/raw_data")
FUNDAMENTAL_FUNCTIONS = {"BALANCE_SHEET", "CASH_FLOW", "INCOME_STATEMENT"}
REQUEST_PAUSE_S = 15


def fetch_data(ticker: str, function_name: str) -> dict | None:
    click.echo(f"  Fetching: {function_name} for {ticker}...")
    param_name = "tickers" if function_name == "NEWS_SENTIMENT" else "symbol"
    url = f"https://www.alphavantage.co/query?function={function_name}&{param_name}={ticker}&apikey={API_KEY}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


def fetch_reported_dates(ticker: str) -> dict[str, str]:
    """fiscalDateEnding -> reportedDate (publication date) from EARNINGS.

    The loader uses it to make a report visible only once it was published
    (src/db/queries.py, fetch_fundamentals).
    """
    data = fetch_data(ticker, "EARNINGS") or {}
    time.sleep(REQUEST_PAUSE_S)
    dates = {
        e["fiscalDateEnding"]: e["reportedDate"]
        for e in data.get("quarterlyEarnings", [])
        if e.get("fiscalDateEnding") and e.get("reportedDate")
    }
    if not dates:
        reason = data.get("Information") or data.get("Note") or "no quarterlyEarnings"
        click.secho(
            f"  -> WARNING: no reportedDate for {ticker} ({reason}); "
            "the loader will assume fiscal date + 90 days",
            fg="yellow",
        )
    return dates


def _write(data, file_path: Path):
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    click.secho(f"  -> Saved: {file_path.resolve()}", fg="green")


def save_to_json(
    data: dict,
    ticker: str,
    function_name: str,
    period: str = "all",
    reported_dates: dict[str, str] | None = None,
):
    """Saves {TICKER}_{annual|quarterly}.json (both for "all"), adds reportedDate."""
    period_map = {"annual": "annualReports", "quarterly": "quarterlyReports"}

    periods = list(period_map) if period == "all" else [period]
    periods = [p for p in periods if period_map[p] in data]
    if not periods:
        click.secho(
            f"  -> WARNING: no {period} reports in {function_name} for {ticker}, "
            "nothing saved",
            fg="red",
        )
        return

    dir_path = OUTPUT_BASE_DIR / function_name
    dir_path.mkdir(parents=True, exist_ok=True)

    for p in periods:
        reports = data[period_map[p]]
        for r in reports:
            r["reportedDate"] = (reported_dates or {}).get(r.get("fiscalDateEnding"))
        _write(
            {"symbol": data.get("symbol", ticker), "period": p, "reports": reports},
            dir_path / f"{ticker}_{p}.json",
        )


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
        reported_dates = (
            fetch_reported_dates(ticker)
            if FUNDAMENTAL_FUNCTIONS.intersection(func_list)
            else None
        )
        for func in func_list:
            data = fetch_data(ticker, func)

            if data and "Information" not in data and "Note" not in data:
                save_to_json(data, ticker, func, period, reported_dates)
            elif data:
                error_msg = data.get("Information", data.get("Note", "API error"))
                click.secho(f"  -> WARNING: {error_msg}", fg="red")

            time.sleep(REQUEST_PAUSE_S)


if __name__ == "__main__":
    main()
