"""One-off smoke test: does the full compiled debate graph run end to end
against a real database and a real LLM?

Exercises `compiled_graph` from src/agents/graph.py directly — no mocks,
unlike this file's neighbor test_agent_db_integration.py (which only calls
one node). This runs the whole protocol from docs/agents.md: Financial +
Sentiment + Macro in parallel -> PM synthesizes -> Critic checks -> either
finalize or PM re-asks a subset of agents, up to MAX_ROUNDS (3).

Cost warning: unlike test_llm_client.py (one call) or
test_agent_db_integration.py (one node, one call), this can trigger many
LLM calls in a single run — 3 agents + PM + Critic per round, up to 3
rounds if the Critic keeps disagreeing. Worst case is roughly 3 + 1 + 1
(round 1) + up to 3 + 1 + 1 per re-ask round x2 more rounds. Cheap per
call (Llama 3.3 MaaS pricing, see docs/project_decisions.md) but not
"one call" cheap — don't loop this in a shell script without thinking
about it.

Uses `.invoke()`, which blocks until the whole debate finishes and returns
only the final state — no round-by-round output. `.stream(state,
stream_mode="updates")` (see docs/ARCHITECTURE.md §6) would give live
per-round deltas instead, but that's for the actual live-status feature,
not needed for a one-off smoke test.

Prereqs: same as test_agent_db_integration.py — Postgres running with data
loaded, ADC (or GEMINI_API_KEY) + GCP_PROJECT_ID/GCP_LOCATION set, and a
ticker/date actually covered by your local datalake.

Usage:
    uv run python tests/integration/test_full_debate.py TSLA 2025-06-01
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  

from dotenv import load_dotenv  

load_dotenv()

from src.agents.graph import compiled_graph  
from src.agents.state import DebateState  


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: uv run python tests/integration/test_full_debate.py "
            "<TICKER> <YYYY-MM-DD>"
        )
        sys.exit(1)

    ticker, as_of_date_str = sys.argv[1], sys.argv[2]
    initial_state: DebateState = {
        "ticker": ticker,
        "horizon": "1W",
        "as_of_date": date.fromisoformat(as_of_date_str),
        "round_number": 1,
        "active_agents": ["financial", "sentiment", "macro"],
        "financial_report": None,
        "sentiment_report": None,
        "macro_report": None,
        "pm_opinion": None,
        "critic_feedback": None,
        "final_result": None,
    }

    print(f"Running full debate for {ticker} as of {as_of_date_str} ...")
    result = compiled_graph.invoke(initial_state)
    final = result["final_result"]

    print(f"\nRounds used: {final.rounds_used}")
    print(f"Critic agreed: {final.critic_agreed}")
    print(f"Direction: {final.direction} (confidence {final.confidence})")
    print(f"Summary: {final.summary}\n")

    for name, report in (
        ("Financial", final.agent_reports.financial),
        ("Sentiment", final.agent_reports.sentiment),
        ("Macro", final.agent_reports.macro),
    ):
        print(f"--- {name} ---")
        print(f"Opinion: {report.opinion}")
        print(f"Reasoning: {report.reasoning}\n")


if __name__ == "__main__":
    main()
