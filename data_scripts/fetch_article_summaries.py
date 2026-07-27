"""
fetch_article_summaries.py

Each article is saved in the form:
{
    "title": ...,
    "url": ...,
    "time_published": ...,
    "source": ...,
    "summary": ...,
    "overall_sentiment_score": ...,
    "overall_sentiment_label": ...,
    "ticker_sentiment_score": ...,
    "ticker_sentiment_label": ...,
    "relevance_score": ...
}

Date range and relevance filter:
  * bounded by --start / --end in YYYY-MM format (server-side, time_from/time_to),
  * additionally drops articles with relevance_score < --min-relevance (default 0.9).

Examples:
    python fetch_article_summaries.py                          # 9 tickers, 2025-01..2026-06
    python fetch_article_summaries.py -t AAPL,NVDA
    python fetch_article_summaries.py --start 2025-01 --end 2026-06 --chunk-months 3
"""

import calendar
import json
import os
import time
from pathlib import Path

import click
import requests
from dotenv import load_dotenv

load_dotenv()

AV_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

DEFAULT_TICKERS = "AAPL,CAT,GS,INTC,META,NFLX,NVDA,SMCI,TSLA"
OUTPUT_BASE_DIR = Path("../datalake/raw_data")

AV_BASE = "https://www.alphavantage.co/query"

# Delay between requests
AV_SLEEP = 1


# ----------------------------------------------------------------------------- #
# HTTP layer
# ----------------------------------------------------------------------------- #
def _get_json(params: dict):
    """GET to Alpha Vantage, returns JSON or None on error."""
    try:
        resp = requests.get(
            AV_BASE, params={**params, "apikey": AV_API_KEY}, timeout=30
        )
    except requests.RequestException as exc:
        click.secho(f"  -> REQUEST ERROR: {exc}", fg="red")
        return None
    if resp.status_code != 200:
        click.secho(f"  -> HTTP {resp.status_code}", fg="red")
        return None
    try:
        return resp.json()
    except json.JSONDecodeError:
        click.secho("  -> Invalid JSON response", fg="red")
        return None


def av_error(data) -> str | None:
    """Rate-limit/error message from Alpha Vantage, if present."""
    if isinstance(data, dict):
        for key in ("Information", "Note", "Error Message"):
            if key in data:
                return data[key]
    return None


# ----------------------------------------------------------------------------- #
# Time windows (workaround for the 1000-articles-per-request limit)
# ----------------------------------------------------------------------------- #
def date_windows(
    start_year: int, start_month: int, end_year: int, end_month: int, chunk_months: int
):
    """Yields (time_from, time_to) in AV format: YYYYMMDDTHHMM."""
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        time_from = f"{y}{m:02d}01T0000"
        em = m + chunk_months - 1
        ey = y + (em - 1) // 12
        em = (em - 1) % 12 + 1
        # clamp the last window to end_year/end_month
        if (ey > end_year) or (ey == end_year and em > end_month):
            ey, em = end_year, end_month
        last_day = calendar.monthrange(ey, em)[1]
        time_to = f"{ey}{em:02d}{last_day:02d}T2359"
        yield time_from, time_to
        nm = em + 1
        ny = ey + (nm - 1) // 12
        nm = (nm - 1) % 12 + 1
        y, m = ny, nm


# ----------------------------------------------------------------------------- #
# Fetcher
# ----------------------------------------------------------------------------- #
def fetch_news_feed(
    ticker: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    chunk_months: int,
) -> list:
    """Fetches the NEWS_SENTIMENT feed for the whole range, by windows, deduped by URL."""
    feed, seen = [], set()
    for time_from, time_to in date_windows(
        start_year, start_month, end_year, end_month, chunk_months
    ):
        click.echo(
            f"  Fetching: NEWS_SENTIMENT {ticker} [{time_from[:8]}-{time_to[:8]}]..."
        )
        data = _get_json(
            {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_from": time_from,
                "time_to": time_to,
                "sort": "EARLIEST",
                "limit": 1000,
            }
        )
        err = av_error(data)
        if err:
            click.secho(f"  -> WARNING: {err}", fg="red")
        elif data:
            for item in data.get("feed", []):
                url = item.get("url")
                if url and url not in seen:
                    seen.add(url)
                    feed.append(item)
        time.sleep(AV_SLEEP)
    return feed


# ----------------------------------------------------------------------------- #
# Transforms / filters
# ----------------------------------------------------------------------------- #
def _ticker_relevance(item: dict, ticker: str) -> float:
    ts = next(
        (t for t in item.get("ticker_sentiment", []) if t.get("ticker") == ticker), {}
    )
    try:
        return (
            float(ts.get("relevance_score"))
            if ts.get("relevance_score") is not None
            else 0.0
        )
    except (TypeError, ValueError):
        return 0.0


def filter_by_relevance(feed: list, ticker: str, min_relevance: float) -> list:
    return [it for it in feed if _ticker_relevance(it, ticker) >= min_relevance]


def extract_summaries(feed: list, ticker: str) -> list:
    """Returns a LIST of article summaries from the (already filtered) feed."""
    articles = []
    for item in feed:
        ts = next(
            (t for t in item.get("ticker_sentiment", []) if t.get("ticker") == ticker),
            {},
        )
        articles.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "time_published": item.get("time_published"),
                "source": item.get("source"),
                "summary": item.get("summary"),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "ticker_sentiment_score": ts.get("ticker_sentiment_score"),
                "ticker_sentiment_label": ts.get("ticker_sentiment_label"),
                "relevance_score": ts.get("relevance_score"),
            }
        )
    return articles


# ----------------------------------------------------------------------------- #
# Saving
# ----------------------------------------------------------------------------- #
def save_json(data, ticker: str, folder: str):
    dir_path = OUTPUT_BASE_DIR / folder
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{ticker}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    click.secho(f"  -> Saved: {file_path.resolve()}", fg="green")


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def parse_ym(s: str):
    """Parses 'YYYY-MM' into (year, month)."""
    try:
        yr, mo = s.strip().split("-")
        yr, mo = int(yr), int(mo)
        if not 1 <= mo <= 12:
            raise ValueError
        return yr, mo
    except (ValueError, AttributeError):
        raise click.ClickException(f"Invalid date format: {s!r} (expected YYYY-MM)")


@click.command()
@click.option(
    "--tickers", "-t", default=DEFAULT_TICKERS, help="Comma-separated list of tickers"
)
@click.option("--start", default="2025-01", help="Range start in YYYY-MM format")
@click.option(
    "--end", default="2026-06", help="Range end in YYYY-MM format (inclusive)"
)
@click.option(
    "--min-relevance",
    default=0.9,
    type=float,
    help="relevance_score threshold (0 = no filter)",
)
@click.option(
    "--chunk-months", default=3, type=int, help="Time window size in months (3=quarter)"
)
def main(tickers, start, end, min_relevance, chunk_months):
    if not AV_API_KEY:
        raise click.ClickException("Missing ALPHA_VANTAGE_API_KEY in .env")

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    start_year, start_month = parse_ym(start)
    end_year, end_month = parse_ym(end)
    if (end_year, end_month) < (start_year, start_month):
        raise click.ClickException("--end cannot be earlier than --start")

    for ticker in ticker_list:
        click.secho(f"--- Stock: {ticker} ---", fg="cyan", bold=True)
        feed = fetch_news_feed(
            ticker, start_year, start_month, end_year, end_month, chunk_months
        )
        feed = filter_by_relevance(feed, ticker, min_relevance)
        click.echo(f"  Articles after filter (relevance>={min_relevance}): {len(feed)}")
        articles = extract_summaries(feed, ticker)
        save_json(articles, ticker, "ARTICLE_SUMMARIES")


if __name__ == "__main__":
    main()
