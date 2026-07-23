"""Unit tests for src/agents/nodes.py.

Convention (matches tests/unit/test_db_loaders.py): no real database and no
real LLM calls here. `fetch_*` (thin SQL) is verified manually against a
running Postgres, not here. `render_prompt` and `llm_client.generate_structured`
are monkeypatched so these tests only check each node's own logic: which
prompt key it picks, what it passes, and which state key it returns.
"""

from datetime import date

from src.agents import llm_client, nodes
from src.agents.state import (
    AgentReport,
    CriticFeedback,
    DebateState,
    NextAgentsSelection,
    PMOpinion,
    PredictResult,
)
from src.db.models import Fundamentals, MacroSeries


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
def _base_state(round_number: int = 1, **overrides) -> DebateState:
    state: DebateState = {
        "ticker": "TSLA",
        "horizon": "1W",
        "as_of_date": date(2025, 1, 1),
        "round_number": round_number,
        "active_agents": ["financial", "sentiment", "macro"],
        "financial_report": None,
        "sentiment_report": None,
        "macro_report": None,
        "pm_opinion": None,
        "critic_feedback": None,
        "final_result": None,
    }
    state.update(overrides)
    return state


class _FakeSessionCM:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


def _fake_render_prompt(calls: list[dict]):
    def fake(name, key="system_prompt", **kwargs):
        calls.append({"name": name, "key": key, "kwargs": kwargs})
        return f"PROMPT[{name}.{key}]"

    return fake


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


def test_format_fundamentals_context_empty():
    assert (
        nodes.format_fundamentals_context([]) == "No fundamentals data available."
    )


def test_format_macro_context_shows_latest_vs_earliest_trend():
    # Rows arrive pre-sorted series_id, date desc — as fetch_macro_series returns.
    rows = [
        MacroSeries(series_id="GDP", date=date(2025, 7, 1), value=100.0),
        MacroSeries(series_id="GDP", date=date(2020, 4, 1), value=80.0),
        MacroSeries(series_id="UNRATE", date=date(2025, 6, 1), value=4.0),
    ]
    result = nodes.format_macro_context(rows)
    assert "GDP: 100.0 as of 2025-07-01 (was 80.0 on 2020-04-01, change +20.00)" in result
    # UNRATE has a single point in the window — no "was/change" clause.
    assert "UNRATE: 4.0 as of 2025-06-01" in result
    assert "UNRATE: 4.0 as of 2025-06-01 (was" not in result


def test_format_macro_context_single_point_series_has_no_change_clause():
    rows = [MacroSeries(series_id="GDP", date=date(2025, 7, 1), value=100.0)]
    assert nodes.format_macro_context(rows) == "GDP: 100.0 as of 2025-07-01"


def test_format_report_none_vs_value():
    assert nodes.format_report(None) == "(no report yet)"
    report = AgentReport(opinion="BUY-ish", reasoning="strong revenue")
    assert "BUY-ish" in nodes.format_report(report)
    assert "strong revenue" in nodes.format_report(report)


def test_format_opinion_none_vs_value():
    assert nodes.format_opinion(None) == "(no opinion yet)"
    opinion = PMOpinion(direction="HOLD", confidence=0.5, summary="mixed", arguments=["a", "b"])
    rendered = nodes.format_opinion(opinion)
    assert "HOLD" in rendered
    assert "mixed" in rendered
    assert "a" in rendered and "b" in rendered


# --------------------------------------------------------------------------- #
# Worker agent nodes — DB fetch + prompt-key branching + LLM call, all mocked
# --------------------------------------------------------------------------- #
def test_financial_agent_node_round1_uses_system_prompt(monkeypatch):
    monkeypatch.setattr(nodes, "get_session", lambda: _FakeSessionCM())
    monkeypatch.setattr(nodes, "fetch_fundamentals", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_prices", lambda *a, **k: [])

    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))

    fake_report = AgentReport(opinion="x", reasoning="y")
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_report, raising=False
    )

    result = nodes.financial_agent_node(_base_state(round_number=1))

    assert result == {"financial_report": fake_report}
    assert render_calls[0]["name"] == "financial_agent"
    assert render_calls[0]["key"] == "system_prompt"


def test_financial_agent_node_round2_uses_revision_prompt(monkeypatch):
    monkeypatch.setattr(nodes, "get_session", lambda: _FakeSessionCM())
    monkeypatch.setattr(nodes, "fetch_fundamentals", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "fetch_prices", lambda *a, **k: [])

    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))

    fake_report = AgentReport(opinion="x2", reasoning="y2")
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_report, raising=False
    )

    state = _base_state(
        round_number=2,
        critic_feedback=CriticFeedback(agree=False, feedback="not enough evidence"),
        financial_report=AgentReport(opinion="old", reasoning="old-r"),
    )
    result = nodes.financial_agent_node(state)

    assert result == {"financial_report": fake_report}
    call = render_calls[0]
    assert call["key"] == "revision_prompt"
    assert call["kwargs"]["critic_feedback"] == "not enough evidence"
    assert "old" in call["kwargs"]["previous_opinion"]


def test_sentiment_agent_node_round1_uses_system_prompt(monkeypatch):
    monkeypatch.setattr(nodes, "get_session", lambda: _FakeSessionCM())
    monkeypatch.setattr(nodes, "fetch_recent_articles", lambda *a, **k: [])

    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))

    fake_report = AgentReport(opinion="x", reasoning="y")
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_report, raising=False
    )

    result = nodes.sentiment_agent_node(_base_state(round_number=1))

    assert result == {"sentiment_report": fake_report}
    assert render_calls[0]["name"] == "sentiment_agent"
    assert render_calls[0]["key"] == "system_prompt"


def test_macro_agent_node_round2_uses_revision_prompt(monkeypatch):
    monkeypatch.setattr(nodes, "get_session", lambda: _FakeSessionCM())
    monkeypatch.setattr(nodes, "fetch_macro_series", lambda *a, **k: [])

    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))

    fake_report = AgentReport(opinion="x", reasoning="y")
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_report, raising=False
    )

    state = _base_state(
        round_number=2,
        critic_feedback=CriticFeedback(agree=False, feedback="check rates again"),
        macro_report=AgentReport(opinion="old-macro", reasoning="old-r"),
    )
    result = nodes.macro_agent_node(state)

    assert result == {"macro_report": fake_report}
    call = render_calls[0]
    assert call["key"] == "revision_prompt"
    assert call["kwargs"]["critic_feedback"] == "check rates again"


# --------------------------------------------------------------------------- #
# PM / Critic nodes — no DB at all, only LLM call
# --------------------------------------------------------------------------- #
def test_pm_synthesize_node_first_round_uses_initial_prompt(monkeypatch):
    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))
    fake_opinion = PMOpinion(direction="BUY", confidence=0.8, summary="s", arguments=["a"])
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_opinion, raising=False
    )

    result = nodes.pm_synthesize_node(_base_state(round_number=1))

    assert result == {"pm_opinion": fake_opinion}
    assert render_calls[0]["name"] == "portfolio_manager"
    assert render_calls[0]["key"] == "initial_prompt"


def test_pm_synthesize_node_later_round_uses_revision_prompt(monkeypatch):
    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))
    fake_opinion = PMOpinion(direction="SELL", confidence=0.6, summary="s2", arguments=["b"])
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_opinion, raising=False
    )

    state = _base_state(
        round_number=2,
        pm_opinion=PMOpinion(direction="BUY", confidence=0.8, summary="prev", arguments=["a"]),
        critic_feedback=CriticFeedback(agree=False, feedback="too optimistic"),
        active_agents=["macro"],
    )
    result = nodes.pm_synthesize_node(state)

    assert result == {"pm_opinion": fake_opinion}
    call = render_calls[0]
    assert call["key"] == "revision_prompt"
    assert call["kwargs"]["critic_feedback"] == "too optimistic"
    assert call["kwargs"]["reasked_agents"] == "macro"


def test_pm_select_next_agents_node_increments_round_and_sets_active_agents(monkeypatch):
    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))
    fake_selection = NextAgentsSelection(agents=["macro"])
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_selection, raising=False
    )

    state = _base_state(
        round_number=1,
        pm_opinion=PMOpinion(direction="BUY", confidence=0.8, summary="s", arguments=["a"]),
        critic_feedback=CriticFeedback(agree=False, feedback="check macro"),
    )
    result = nodes.pm_select_next_agents_node(state)

    assert result == {"active_agents": ["macro"], "round_number": 2}
    assert render_calls[0]["name"] == "portfolio_manager"
    assert render_calls[0]["key"] == "select_agents_prompt"


def test_critic_node_returns_feedback(monkeypatch):
    render_calls: list[dict] = []
    monkeypatch.setattr(nodes, "render_prompt", _fake_render_prompt(render_calls))
    fake_feedback = CriticFeedback(agree=True, feedback="looks solid")
    monkeypatch.setattr(
        llm_client, "generate_structured", lambda prompt, model: fake_feedback, raising=False
    )

    state = _base_state(
        round_number=1,
        pm_opinion=PMOpinion(direction="BUY", confidence=0.8, summary="s", arguments=["a"]),
    )
    result = nodes.critic_node(state)

    assert result == {"critic_feedback": fake_feedback}
    assert render_calls[0]["name"] == "critic"


# --------------------------------------------------------------------------- #
# finalize_node — pure, no LLM, no DB
# --------------------------------------------------------------------------- #
def test_finalize_node_assembles_predict_result():
    state = _base_state(
        round_number=2,
        pm_opinion=PMOpinion(direction="BUY", confidence=0.7, summary="s", arguments=["a"]),
        critic_feedback=CriticFeedback(agree=True, feedback="ok"),
        financial_report=AgentReport(opinion="f-op", reasoning="f-r"),
        sentiment_report=AgentReport(opinion="s-op", reasoning="s-r"),
        macro_report=AgentReport(opinion="m-op", reasoning="m-r"),
    )

    result = nodes.finalize_node(state)

    final = result["final_result"]
    assert isinstance(final, PredictResult)
    assert final.direction == "BUY"
    assert final.confidence == 0.7
    assert final.rounds_used == 2
    assert final.critic_agreed is True
    assert final.agent_reports.financial.opinion == "f-op"
    assert final.agent_reports.sentiment.opinion == "s-op"
    assert final.agent_reports.macro.opinion == "m-op"
