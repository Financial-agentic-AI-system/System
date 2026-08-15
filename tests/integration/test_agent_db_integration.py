"""One-off smoke test: does a real agent node pull real DB data and produce
a real LLM call, end to end?

Exercises the actual production path used by the debate graph, no mocks:

    Postgres (fundamentals / stock_prices tables)
      -> fetch_fundamentals / fetch_prices        (src/agents/nodes.py)
      -> format_fundamentals_context / format_price_context
      -> render_prompt("financial_agent", ...)    (src/agents/prompt_loader.py)
      -> llm_client.generate_structured(...)      (src/agents/llm_client.py)

Uses financial_agent_node specifically because it only needs plain SQL
(fetch_fundamentals/fetch_prices) — sentiment_agent_node's fetch_recent_articles
also works the same way today, but the Sentiment agent is documented in
docs/ARCHITECTURE.md as eventually using retriever/pgvector similarity search,
which isn't built yet (src/retriever/ is still an empty scaffold).

NOT part of tests/unit — those mock generate_structured and never touch a
real database (see tests/unit/test_agents_nodes.py). This script needs:
  - Postgres running with data loaded:
        docker compose up -d postgres
        uv run python -m src.db.load
  - ADC + GCP_PROJECT_ID/GCP_LOCATION set, same prereqs as test_llm_client.py
  - a ticker/date that's actually covered by your local datalake — check what
    you ingested via data_scripts/alpha_vantage_fetcher.py

Usage:
    uv run python test_agent_db_integration.py TSLA 2025-06-01
"""

import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from src.agents.nodes import financial_agent_node  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: uv run python test_agent_db_integration.py <TICKER> <YYYY-MM-DD>")
        sys.exit(1)

    ticker, as_of_date_str = sys.argv[1], sys.argv[2]
    state = {
        "ticker": ticker,
        "horizon": "1W",
        "as_of_date": date.fromisoformat(as_of_date_str),
        "round_number": 1,
    }

    result = financial_agent_node(state)
    report = result["financial_report"]
    print("Opinion:", report.opinion)
    print("Reasoning:", report.reasoning)


if __name__ == "__main__":
    main()
