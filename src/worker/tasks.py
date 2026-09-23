"""Celery task that runs the LangGraph debate and mirrors its progress into
src/cache/debate_cache.py.

`_run_debate` is a plain function, deliberately separate from the `@app.task`
wrapper — testable without Celery at all (same split as graph.py/nodes.py
already use for the graph itself). `run_debate` exists only to hand Celery's
own task_id (`self.request.id`) to `_run_debate`, since that id is what
FastAPI/Streamlit will later use to key into debate_cache.

Uses `.stream(state, stream_mode="updates")`, not `.invoke()` — see
docs/ARCHITECTURE.md §6: `.invoke()` blocks until the whole debate finishes
and returns only the final state; `.stream()` yields one delta per
completed node as the debate progresses, which is what makes the live
status/history feature possible at all. Confirmed empirically
(scripts/investigate_stream_shapes.py) that each delta carries exactly one
node's update, even for the parallel Financial/Sentiment/Macro fan-out.
"""

from datetime import date

from src.agents.graph import compiled_graph
from src.agents.state import DebateState, PredictResult
from src.cache import debate_cache
from src.worker.celery_app import app


def _run_debate(
    ticker: str, horizon: str, as_of_date: date, task_id: str
) -> PredictResult:
    initial_state: DebateState = {
        "ticker": ticker,
        "horizon": horizon,
        "as_of_date": as_of_date,
        "round_number": 1,
        "active_agents": ["financial", "sentiment", "macro"],
        "financial_report": None,
        "sentiment_report": None,
        "macro_report": None,
        "pm_opinion": None,
        "critic_feedback": None,
        "final_result": None,
    }
    debate_cache.set_status(task_id, "RUNNING", round_number=1)

    round_number = 1
    final_result: PredictResult | None = None
    try:
        for delta in compiled_graph.stream(initial_state, stream_mode="updates"):
            node_name, update = next(iter(delta.items()))
            if "round_number" in update:
                round_number = update["round_number"]
            debate_cache.append_history(task_id, node_name, round_number, update)
            debate_cache.set_status(
                task_id, "RUNNING", round_number, current_node=node_name
            )
            if node_name == "finalize":
                final_result = update["final_result"]
    except Exception as exc:
        debate_cache.set_status(task_id, "FAILED", round_number, error=str(exc))
        raise

    debate_cache.set_result(task_id, final_result)
    debate_cache.set_status(task_id, "DONE", round_number, current_node="finalize")
    return final_result


@app.task(bind=True)
def run_debate(self, ticker: str, horizon: str, as_of_date: date) -> dict:
    result = _run_debate(ticker, horizon, as_of_date, task_id=self.request.id)
    return result.model_dump(mode="json")
