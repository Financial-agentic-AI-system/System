"""Pure logic that turns the flat history stream into per-round view models.

No Streamlit imports here, so it is unit-testable (tests/unit). The history
comes from `GET /history/{task_id}`: one entry per finished graph node, in
order, where `delta` is that node's LangGraph update.

Round numbering follows the worker: `pm_select_next_agents` bumps
`round_number` *before* its history entry is written, so that entry and
everything after it belongs to the new round.
"""

from dataclasses import dataclass, field

import config
from models import (
    AgentReport,
    CriticFeedback,
    DebateStatus,
    HistoryEntry,
    PMOpinion,
)

AGENTS = ("financial", "sentiment", "macro")
AGENT_LABELS = {
    "financial": "Financial",
    "sentiment": "Sentiment",
    "macro": "Macro",
}
TERMINAL = {"DONE", "FAILED"}


@dataclass
class RoundView:
    number: int
    reasked: list[str] = field(default_factory=lambda: list(AGENTS))
    reports: dict[str, AgentReport] = field(default_factory=dict)
    pm_opinion: PMOpinion | None = None
    critic: CriticFeedback | None = None


def is_terminal(status: DebateStatus) -> bool:
    return status.status in TERMINAL


def build_rounds(history: list[HistoryEntry]) -> list[RoundView]:
    rounds: dict[int, RoundView] = {}
    for entry in history:
        view = rounds.setdefault(entry.round_number, RoundView(entry.round_number))
        if entry.node in AGENTS:
            view.reports[entry.node] = AgentReport.model_validate(
                entry.delta[f"{entry.node}_report"]
            )
        elif entry.node == "pm_synthesize":
            view.pm_opinion = PMOpinion.model_validate(entry.delta["pm_opinion"])
        elif entry.node == "critic":
            view.critic = CriticFeedback.model_validate(entry.delta["critic_feedback"])
        elif entry.node == "pm_select_next_agents":
            view.reasked = list(entry.delta["active_agents"])
        # "finalize" carries no per-round information.
    return [rounds[n] for n in sorted(rounds)]


def stage_states(view: RoundView, live: bool) -> dict[str, str]:
    """State of each stage of one round: done | running | waiting | kept.

    `kept` = the PM did not re-ask this agent, its previous report stands.
    `running` is only ever reported for a live round — a finished debate
    must not show spinners.
    """
    states: dict[str, str] = {}
    for agent in AGENTS:
        if agent not in view.reasked:
            states[agent] = "kept"
        elif agent in view.reports:
            states[agent] = "done"
        else:
            states[agent] = "running" if live else "waiting"

    agents_done = all(a in view.reports for a in view.reasked)
    if view.pm_opinion:
        states["pm"] = "done"
    else:
        states["pm"] = "running" if live and agents_done else "waiting"
    if view.critic:
        states["critic"] = "done"
    else:
        states["critic"] = "running" if live and view.pm_opinion else "waiting"
    return states


def describe_progress(rounds: list[RoundView], status: DebateStatus) -> str:
    """One-line 'what is happening right now' for the live view."""
    if status.status == "FAILED":
        return f"Debate failed: {status.error or 'unknown error'}"
    if status.status == "DONE":
        return "Debate finished."
    if not rounds:
        if status.status == "PENDING":
            return "Queued — waiting for the worker to pick up the task."
        return "Starting — the agents are loading their data."

    cur = rounds[-1]
    if cur.critic:
        if not cur.critic.agree and cur.number < config.MAX_ROUNDS:
            return "The Critic disagreed — the PM is choosing which agents to re-ask."
        return "Assembling the final prediction."
    if cur.pm_opinion:
        return "The Critic is reviewing the PM's opinion."
    pending = [AGENT_LABELS[a] for a in cur.reasked if a not in cur.reports]
    if pending:
        return f"Round {cur.number}: waiting for {', '.join(pending)}."
    return "The PM is synthesizing the agents' reports."
