# Multi-Agent Financial System (MAS)

An AI-driven investment analysis system using a Multi-Agent architecture to simulate hedge fund decision-making. This project leverages Large Language Models (LLMs) to process macroeconomic, fundamental, and sentiment data.

## Project Structure

For the system to work correctly, maintain the following directory structure:
```text
Thesis/
├── code/              <-- This repository (logic)
└── datalake/          <-- External data repository (JSON storage)
```
## Setup

1. **Install uv** (if you haven't already):
   Follow the instructions at [astral.sh/uv](https://astral.sh/uv/).

2. **Set up the environment and install dependencies**:
   ```bash
   git clone <repository_url>
   cd <repository_name>

   # Create a virtual environment
   uv venv

   # Activate the environment:
   # On macOS/Linux:
   source .venv/bin/activate 
   # On Windows:
   .venv\Scripts\activate

   # Sync dependencies from requirements.txt
   uv pip install -r requirements.txt

#### Data Ingestion
The system includes a generic CLI tool for fetching data from Alpha Vantage.

Basic usage:

```bash
python data_ingestion_scripts/alpha_vantage_fetcher.py --functions <FUNCTIONS> --tickers <TICKERS> --period <PERIOD>
```