"""Unit tests for src/agents/nodes.py and src/agents/volatility.py.

No real database and no real LLM calls here (`no_real_db`). `render_prompt` and
`llm_client.generate_structured` are replaced by the `run_node` fixture, so
these tests only check each node's own logic: which prompt key it picks, what
it passes, and which state key it returns.
"""

from datetime import date
from types import SimpleNamespace

import pytest

from src.agents import llm_client, nodes
from src.agents.state import (
    AgentReport,
    CriticFeedback,
    NextAgentsSelection,
    PMOpinion,
    PredictResult,
)
from src.agents.volatility import ewm_std, horizon_sigma
from src.db.models import Fundamentals, MacroSeries

pytestmark = pytest.mark.usefixtures("no_real_db")

REPORT = AgentReport(opinion="x", reasoning="y")
OPINION = PMOpinion(direction="BUY", confidence=0.8, summary="s", arguments=["a"])
DISAGREE = CriticFeedback(agree=False, feedback="not enough evidence")
# alternating +1% / -1% closes (newest first) -> non-zero daily sigma
CLOSES = [100.0 * (1.01 if i % 2 else 1.0) for i in range(30)]


@pytest.fixture
def run_node(monkeypatch):
    """Run a node with the LLM answering `answer`; returns the node's result
    and the `render_prompt` calls it made."""

    def run(node, state, answer):
        calls: list[dict] = []

        def fake_render(name, key="system_prompt", **kwargs):
            calls.append({"name": name, "key": key, "kwargs": kwargs})
            return f"PROMPT[{name}.{key}]"

        monkeypatch.setattr(nodes, "render_prompt", fake_render)
        monkeypatch.setattr(
            llm_client, "generate_structured", lambda prompt, model: answer
        )
        return node(state), calls

    return run


# --------------------------------------------------------------------------- #
# format_* — pure, no DB
# --------------------------------------------------------------------------- #
def test_format_fundamentals_context_drops_none_and_formats_key_value():
    row = Fundamentals(
        report_type="BALANCE_SHEET",
        fiscal_date=date(2025, 9, 30),
        data={"totalAssets": 100, "goodwill": None},
    )
    assert (
        nodes.format_fundamentals_context([row])
        == "BALANCE_SHEET (2025-09-30): totalAssets=100"
    )
    assert nodes.format_fundamentals_context([]) == "No fundamentals data available."


def test_format_macro_context_shows_latest_vs_earliest_trend():
    # Rows arrive pre-sorted series_id, date desc — as fetch_macro_series returns.
    rows = [
        MacroSeries(series_id="GDP", date=date(2025, 7, 1), value=100.0),
        MacroSeries(series_id="GDP", date=date(2020, 4, 1), value=80.0),
        MacroSeries(series_id="UNRATE", date=date(2025, 6, 1), value=4.0),
    ]
    # UNRATE has a single point in the window — no "was/change" clause.
    assert nodes.format_macro_context(rows) == (
        "GDP: 100.0 as of 2025-07-01 (was 80.0 on 2020-04-01, change +20.00)\n"
        "UNRATE: 4.0 as of 2025-06-01"
    )


def test_format_report_and_opinion_none_vs_value():
    assert nodes.format_report(None) == "(no report yet)"
    rendered = nodes.format_report(AgentReport(opinion="BUY-ish", reasoning="strong"))
    assert "BUY-ish" in rendered and "strong" in rendered

    assert nodes.format_opinion(None) == "(no opinion yet)"
    rendered = nodes.format_opinion(
        PMOpinion(
            direction="HOLD", confidence=0.5, summary="mixed", arguments=["a", "b"]
        )
    )
    assert "HOLD" in rendered and "mixed" in rendered and "a; b" in rendered


# --------------------------------------------------------------------------- #
# Worker agent nodes — prompt-key branching + LLM call
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("node", "prompt_name", "report_key"),
    [
        (nodes.financial_agent_node, "financial_agent", "financial_report"),
        (nodes.macro_agent_node, "macro_agent", "macro_report"),
    ],
)
def test_agent_node_picks_prompt_by_round(
    run_node, debate_state, node, prompt_name, report_key
):
    result, calls = run_node(node, debate_state(), REPORT)

    assert result == {report_key: REPORT}
    assert (calls[0]["name"], calls[0]["key"]) == (prompt_name, "system_prompt")

    previous = AgentReport(opinion="old", reasoning="old-r")
    state = debate_state(
        round_number=2, critic_feedback=DISAGREE, **{report_key: previous}
    )
    result, calls = run_node(node, state, REPORT)

    assert result == {report_key: REPORT}
    assert calls[0]["key"] == "revision_prompt"
    assert calls[0]["kwargs"]["critic_feedback"] == "not enough evidence"
    assert "old" in calls[0]["kwargs"]["previous_opinion"]


def test_sentiment_agent_node_round1_without_search(run_node, debate_state):
    # `no_real_db`: the model declines to search, so the retriever is not reached.
    result, calls = run_node(nodes.sentiment_agent_node, debate_state(), REPORT)

    assert result == {
        "sentiment_report": REPORT,
        "sentiment_queries": [],
        "sentiment_sources": [],
    }
    # Two prompts per round: the one that lets the model ask for a search, then
    # the one that produces the opinion.
    assert [c["name"] for c in calls] == ["sentiment_agent"] * 2
    assert [c["key"] for c in calls] == ["search_prompt", "system_prompt"]


# --------------------------------------------------------------------------- #
# PM / Critic nodes
# --------------------------------------------------------------------------- #
def test_pm_synthesize_node_first_round_uses_initial_prompt_with_hold_band(
    run_node, debate_state, monkeypatch
):
    rows = [SimpleNamespace(close=c) for c in CLOSES]
    monkeypatch.setattr(nodes, "fetch_prices", lambda *a, **k: rows)

    result, calls = run_node(nodes.pm_synthesize_node, debate_state(), OPINION)

    assert result == {"pm_opinion": OPINION}
    assert (calls[0]["name"], calls[0]["key"]) == (
        "portfolio_manager",
        "initial_prompt",
    )
    kwargs = calls[0]["kwargs"]
    assert kwargs["horizon_long"].startswith("1 week")
    assert kwargs["band_pct"].endswith("%")
    assert kwargs["vol_span"] == 100


def test_pm_synthesize_node_later_round_uses_revision_prompt(run_node, debate_state):
    state = debate_state(
        round_number=2,
        pm_opinion=OPINION,
        critic_feedback=DISAGREE,
        active_agents=["macro"],
    )
    revised = PMOpinion(direction="SELL", confidence=0.6, summary="s2", arguments=["b"])

    result, calls = run_node(nodes.pm_synthesize_node, state, revised)

    assert result == {"pm_opinion": revised}
    kwargs = calls[0]["kwargs"]
    assert calls[0]["key"] == "revision_prompt"
    assert kwargs["critic_feedback"] == "not enough evidence"
    assert kwargs["reasked_agents"] == "macro"
    # no prices in the database: the band falls back to words
    assert kwargs["band_pct"] == "its typical volatility"


def test_pm_select_next_agents_node_increments_round_and_sets_active_agents(
    run_node, debate_state
):
    state = debate_state(pm_opinion=OPINION, critic_feedback=DISAGREE)

    result, calls = run_node(
        nodes.pm_select_next_agents_node, state, NextAgentsSelection(agents=["macro"])
    )

    assert result == {"active_agents": ["macro"], "round_number": 2}
    assert (calls[0]["name"], calls[0]["key"]) == (
        "portfolio_manager",
        "select_agents_prompt",
    )


def test_critic_node_returns_feedback(run_node, debate_state):
    feedback = CriticFeedback(agree=True, feedback="looks solid")

    result, calls = run_node(
        nodes.critic_node, debate_state(pm_opinion=OPINION), feedback
    )

    assert result == {"critic_feedback": feedback}
    assert calls[0]["name"] == "critic"


def test_finalize_node_assembles_predict_result(debate_state):
    state = debate_state(
        round_number=2,
        pm_opinion=PMOpinion(
            direction="BUY", confidence=0.7, summary="s", arguments=["a"]
        ),
        critic_feedback=CriticFeedback(agree=True, feedback="ok"),
        financial_report=AgentReport(opinion="f-op", reasoning="f-r"),
        sentiment_report=AgentReport(opinion="s-op", reasoning="s-r"),
        macro_report=AgentReport(opinion="m-op", reasoning="m-r"),
    )

    final = nodes.finalize_node(state)["final_result"]

    assert isinstance(final, PredictResult)
    assert final.direction == "BUY"
    assert final.confidence == 0.7
    assert final.rounds_used == 2
    assert final.critic_agreed is True
    assert final.agent_reports.financial.opinion == "f-op"
    assert final.agent_reports.sentiment.opinion == "s-op"
    assert final.agent_reports.macro.opinion == "m-op"


# --------------------------------------------------------------------------- #
# volatility — the band shared by the PM prompt and the backtest labels
# --------------------------------------------------------------------------- #
def test_horizon_sigma_scales_with_sqrt_of_sessions():
    one_week = horizon_sigma(CLOSES, "1W")
    one_month = horizon_sigma(CLOSES, "1M")
    assert one_month / one_week == pytest.approx((21 / 5) ** 0.5)
    assert horizon_sigma([100.0], "1W") is None


def test_ewm_std_matches_pandas():
    pd = pytest.importorskip("pandas")

    returns_desc = [0.01, -0.02, 0.015, 0.003, -0.007, 0.02, -0.011] * 20
    expected = pd.Series(returns_desc[::-1]).ewm(span=100).std().iloc[-1]
    assert ewm_std(returns_desc, 100) == pytest.approx(expected)
