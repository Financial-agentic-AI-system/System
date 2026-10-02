"""Unit tests for frontend/debate_model.py — pure logic, no Streamlit."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

from debate_model import build_rounds, describe_progress, stage_states  # noqa: E402
from models import DebateStatus, HistoryEntry  # noqa: E402

TS = datetime(2026, 1, 15, tzinfo=timezone.utc)


def _report(text: str = "x") -> dict:
    return {"opinion": text, "reasoning": text}


def _entry(node: str, rnd: int, delta: dict) -> HistoryEntry:
    return HistoryEntry(node=node, round_number=rnd, ts=TS, delta=delta)


def _pm(direction: str = "BUY") -> dict:
    return {
        "pm_opinion": {
            "direction": direction,
            "confidence": 0.7,
            "summary": "s",
            "arguments": ["a"],
        }
    }


def _critic(agree: bool) -> dict:
    return {"critic_feedback": {"agree": agree, "feedback": "f"}}


def _running() -> DebateStatus:
    return DebateStatus(status="RUNNING", round_number=1)


def test_build_rounds_empty():
    assert build_rounds([]) == []


def test_build_rounds_two_round_debate_with_partial_reask():
    history = [
        _entry("financial", 1, {"financial_report": _report()}),
        _entry("sentiment", 1, {"sentiment_report": _report()}),
        _entry("macro", 1, {"macro_report": _report()}),
        _entry("pm_synthesize", 1, _pm("HOLD")),
        _entry("critic", 1, _critic(False)),
        # the worker bumps round_number before writing the select entry
        _entry("pm_select_next_agents", 2, {"active_agents": ["macro"]}),
        _entry("macro", 2, {"macro_report": _report("revised")}),
        _entry("pm_synthesize", 2, _pm("BUY")),
        _entry("critic", 2, _critic(True)),
        _entry("finalize", 2, {"final_result": {}}),
    ]
    rounds = build_rounds(history)

    assert [r.number for r in rounds] == [1, 2]
    assert rounds[0].reasked == ["financial", "sentiment", "macro"]
    assert rounds[1].reasked == ["macro"]
    assert set(rounds[1].reports) == {"macro"}
    assert rounds[1].reports["macro"].opinion == "revised"
    assert rounds[0].pm_opinion.direction == "HOLD"
    assert rounds[1].pm_opinion.direction == "BUY"
    assert rounds[0].critic.agree is False
    assert rounds[1].critic.agree is True


def test_stage_states_live_round_in_progress():
    rounds = build_rounds([_entry("financial", 1, {"financial_report": _report()})])
    states = stage_states(rounds[0], live=True)
    assert states["financial"] == "done"
    assert states["sentiment"] == "running"
    assert states["macro"] == "running"
    assert states["pm"] == "waiting"  # agents not all done
    assert states["critic"] == "waiting"


def test_stage_states_pm_runs_once_all_agents_done():
    history = [
        _entry(a, 1, {f"{a}_report": _report()})
        for a in ("financial", "sentiment", "macro")
    ]
    states = stage_states(build_rounds(history)[0], live=True)
    assert states["pm"] == "running"
    assert states["critic"] == "waiting"


def test_stage_states_marks_agents_not_reasked_as_kept():
    history = [
        _entry("pm_select_next_agents", 2, {"active_agents": ["financial"]}),
    ]
    states = stage_states(build_rounds(history)[0], live=True)
    assert states["financial"] == "running"
    assert states["sentiment"] == "kept"
    assert states["macro"] == "kept"


def test_stage_states_never_running_when_not_live():
    states = stage_states(build_rounds([_entry("critic", 1, _critic(True))])[0], False)
    assert "running" not in states.values()


def test_describe_progress_variants():
    assert "Queued" in describe_progress(
        [], DebateStatus(status="PENDING", round_number=1)
    )
    assert "Starting" in describe_progress([], _running())

    one = build_rounds([_entry("financial", 1, {"financial_report": _report()})])
    assert "Sentiment, Macro" in describe_progress(one, _running())

    full = [
        _entry(a, 1, {f"{a}_report": _report()})
        for a in ("financial", "sentiment", "macro")
    ]
    assert "synthesizing" in describe_progress(build_rounds(full), _running())

    with_pm = build_rounds(full + [_entry("pm_synthesize", 1, _pm())])
    assert "Critic is reviewing" in describe_progress(with_pm, _running())

    disagree = build_rounds(
        full + [_entry("pm_synthesize", 1, _pm()), _entry("critic", 1, _critic(False))]
    )
    assert "re-ask" in describe_progress(disagree, _running())

    agree = build_rounds(
        full + [_entry("pm_synthesize", 1, _pm()), _entry("critic", 1, _critic(True))]
    )
    assert "final prediction" in describe_progress(agree, _running())


def test_describe_progress_terminal_states():
    failed = DebateStatus(status="FAILED", round_number=2, error="LLM timeout")
    assert "LLM timeout" in describe_progress([], failed)
    assert describe_progress([], DebateStatus(status="DONE")) == "Debate finished."
