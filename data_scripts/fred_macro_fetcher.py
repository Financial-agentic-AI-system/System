import datetime
import click
from pathlib import Path
import pandas_datareader.data as web

DEFAULT_SERIES = "FEDFUNDS,GDP,CPIAUCSL,PPIACO,T10Y2Y,DGS10"
OUTPUT_BASE_DIR = Path("../datalake/raw_data/FRED_MACRO")


def fetch_and_save_fred_data(
    series_id: str, start_date: datetime.datetime, end_date: datetime.datetime
):
    click.echo(f"  Fetching time series: {series_id}...")

    try:
        df = web.DataReader(series_id, "fred", start_date, end_date)

        if not df.empty:
            dir_path = OUTPUT_BASE_DIR
            dir_path.mkdir(parents=True, exist_ok=True)

            df.index = df.index.strftime("%Y-%m-%d")
            file_path = dir_path / f"{series_id}.csv"
            df.to_csv(file_path, index=True)

            click.secho(f"  -> Saved: {file_path.resolve()}", fg="green")
        else:
            click.secho(f"  -> WARNING: No data for {series_id}", fg="yellow")

    except Exception as e:
        click.secho(f"  -> ERROR fetching {series_id}: {str(e)}", fg="red")


@click.command()
@click.option(
    "--series",
    "-s",
    default=DEFAULT_SERIES,
    help="List of FRED identifiers (comma-separated)",
)
@click.option(
    "--years", "-y", default=10, type=int, help="Number of years back to fetch"
)
def main(series: str, years: int):
    """
    Script fetching macroeconomic data from the FRED database to feed the Macroeconomic Agent.
    """
    series_list = [s.strip() for s in series.split(",")]

    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=years * 365)

    click.secho(
        f"=== Fetching macro data from the last {years} years ===", fg="cyan", bold=True
    )
    click.secho(
        f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n",
        fg="cyan",
    )

    for s_id in series_list:
        fetch_and_save_fred_data(s_id, start_date, end_date)


if __name__ == "__main__":
    main()
