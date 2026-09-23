"""In-process fake of the backend, so the UI can be built and demoed before
the FastAPI endpoints exist.

It replays a scripted debate against the wall clock and produces exactly
the shapes the real worker writes to Redis (`src/worker/tasks.py`): one
history entry per finished node, whose `delta` is that node's LangGraph
update (`financial_report`, `pm_opinion`, `critic_feedback`,
`active_agents` + `round_number`, `final_result`). The script is
deterministic per (ticker, horizon, as_of_date).
"""

import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import config
from models import (
    ApiError,
    DebateStatus,
    HistoryEntry,
    PredictResponse,
    PredictResult,
)

MODEL_VERSION = "meta/llama-3.3-70b-instruct-maas"
AGENTS = ("financial", "sentiment", "macro")
DIRECTIONS = ("BUY", "HOLD", "SELL")
QUEUE_DELAY_S = 1.5

_REPORTS = {
    "financial": {
        "BUY": (
            "{t}'s fundamentals are improving.",
            "Revenue growth accelerated over the last quarters, operating margin "
            "expanded and free cash flow covers capex comfortably. Leverage is "
            "stable, so the balance sheet does not constrain upside over the "
            "{h} horizon.",
        ),
        "HOLD": (
            "{t}'s fundamentals are mixed.",
            "Growth is steady but margins are flat and the valuation already "
            "discounts most of the expected improvement. Nothing in the "
            "statements argues for a directional bet over the {h} horizon.",
        ),
        "SELL": (
            "{t}'s fundamentals are deteriorating.",
            "Revenue growth is slowing, margins are compressing and debt service "
            "takes a growing share of operating cash flow. The latest quarters "
            "undershoot the trend, a headwind over the {h} horizon.",
        ),
    },
    "sentiment": {
        "BUY": (
            "News flow around {t} is clearly positive.",
            "Most recent articles carry bullish sentiment labels with high "
            "relevance scores; coverage focuses on product momentum and "
            "upgraded outlooks. No sign of a crowded or euphoric tone yet.",
        ),
        "HOLD": (
            "Sentiment on {t} is neutral and noisy.",
            "Bullish and bearish articles roughly cancel out, and relevance "
            "scores are low for most items. There is no sentiment edge either "
            "way for the {h} horizon.",
        ),
        "SELL": (
            "News flow around {t} is negative.",
            "Recent coverage is dominated by bearish labels: guidance worries, "
            "regulatory pressure and downgrades. The tone has worsened over "
            "the last weeks before the as-of date.",
        ),
    },
    "macro": {
        "BUY": (
            "The macro backdrop supports risk assets.",
            "Rates have stopped rising, inflation is trending down and credit "
            "conditions are easing versus the start of the lookback window. "
            "This favours {t} over the {h} horizon.",
        ),
        "HOLD": (
            "The macro backdrop is ambiguous.",
            "Growth indicators are stable but rates stay restrictive and "
            "inflation progress has stalled. The macro picture neither helps "
            "nor hurts {t} materially over the {h} horizon.",
        ),
        "SELL": (
            "The macro backdrop is a headwind.",
            "Yields are climbing, the yield curve is inverted and unemployment "
            "is ticking up. Higher discount rates weigh on {t}'s valuation "
            "over the {h} horizon.",
        ),
    },
}

_PM_SUMMARY = {
    "BUY": "Fundamentals, sentiment and macro context lean positive for {t}; "
    "the upside case outweighs the risks over the {h} horizon.",
    "HOLD": "Signals for {t} conflict and none dominates, so staying neutral "
    "over the {h} horizon is the best-supported call.",
    "SELL": "Deteriorating fundamentals and a weak backdrop make the downside "
    "case for {t} stronger over the {h} horizon.",
}

_AGENT_LABEL = {
    "financial": "Financial Agent",
    "sentiment": "Sentiment Agent",
    "macro": "Macro Agent",
}


@dataclass
class _Event:
    at: float  # seconds after start
    node: str
    round_number: int
    delta: dict


@dataclass
class _Task:
    task_id: str
    ticker: str
    horizon: str
    as_of_date: date
    started: datetime
    events: list[_Event]
    result: dict

    @property
    def duration(self) -> float:
        return self.events[-1].at


def _report(agent: str, stance: str, ticker: str, horizon: str, revised: bool) -> dict:
    opinion, reasoning = _REPORTS[agent][stance]
    fmt = {"t": ticker, "h": config.HORIZONS.get(horizon, horizon)}
    opinion, reasoning = opinion.format(**fmt), reasoning.format(**fmt)
    if revised:
        opinion = f"Revised: {opinion}"
        reasoning = (
            "Taking the Critic's feedback into account I re-checked the data. "
            + reasoning
        )
    return {"opinion": opinion, "reasoning": reasoning}


def _pm_opinion(
    direction: str,
    confidence: float,
    stances: dict[str, str],
    ticker: str,
    horizon: str,
) -> dict:
    fmt = {"t": ticker, "h": config.HORIZONS.get(horizon, horizon)}
    arguments = [
        f"{_AGENT_LABEL[a]}: {_REPORTS[a][s][0].format(**fmt)}"
        for a, s in stances.items()
    ]
    return {
        "direction": direction,
        "confidence": confidence,
        "summary": _PM_SUMMARY[direction].format(**fmt),
        "arguments": arguments,
    }


def build_script(
    ticker: str, horizon: str, as_of_date: date
) -> tuple[list[_Event], dict]:
    """Deterministic scripted debate: (timed events, final PredictResult dict)."""
    rng = random.Random(zlib.crc32(f"{ticker}|{horizon}|{as_of_date}".encode()))
    final_dir = rng.choice(DIRECTIONS)
    rounds = rng.choice([1, 2, 2, 3])
    last_round_agrees = rounds < 3 or rng.random() < 0.5

    stances = {
        a: final_dir if rng.random() < 0.7 else rng.choice(DIRECTIONS) for a in AGENTS
    }
    reports: dict[str, dict] = {}
    events: list[_Event] = []
    active = list(AGENTS)
    t = QUEUE_DELAY_S
    agree = False
    opinion: dict = {}

    for r in range(1, rounds + 1):
        if r > 1:
            active = rng.sample(AGENTS, k=rng.choice([1, 2]))
            t += 2.0
            events.append(
                _Event(
                    t,
                    "pm_select_next_agents",
                    r,
                    {"active_agents": active, "round_number": r},
                )
            )
        for agent in active:
            t += rng.uniform(1.0, 2.0)
            if r > 1:
                stances[agent] = final_dir
            reports[agent] = _report(agent, stances[agent], ticker, horizon, r > 1)
            events.append(_Event(t, agent, r, {f"{agent}_report": reports[agent]}))

        t += 2.5
        direction = (
            final_dir
            if r == rounds
            else rng.choice([d for d in DIRECTIONS if d != final_dir])
        )
        confidence = round(
            rng.uniform(0.65, 0.88) if r == rounds else rng.uniform(0.45, 0.62), 2
        )
        opinion = _pm_opinion(direction, confidence, stances, ticker, horizon)
        events.append(_Event(t, "pm_synthesize", r, {"pm_opinion": opinion}))

        t += 2.5
        agree = r == rounds and last_round_agrees
        feedback = (
            "The opinion is consistent with the agent reports and the confidence "
            "is proportionate to the evidence."
            if agree
            else "The opinion leans on one report and does not address the "
            "conflicting signals from the others. Justify the confidence level "
            "or ask the relevant agent to reconsider."
        )
        events.append(
            _Event(
                t,
                "critic",
                r,
                {"critic_feedback": {"agree": agree, "feedback": feedback}},
            )
        )

    result = {
        "direction": opinion["direction"],
        "confidence": opinion["confidence"],
        "summary": opinion["summary"],
        "rounds_used": rounds,
        "critic_agreed": agree,
        "agent_reports": reports,
    }
    events.append(_Event(t + 1.0, "finalize", rounds, {"final_result": result}))
    return events, result


class MockClient:
    """Same interface as `api_client.BackendClient`."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._tasks: dict[str, _Task] = {}

    def _task(self, task_id: str) -> _Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise ApiError("Unknown task_id (mock backend was restarted).") from None

    def _elapsed(self, task: _Task) -> float:
        return (self._clock() - task.started).total_seconds()

    def _finished(self, task: _Task) -> list[_Event]:
        elapsed = self._elapsed(task)
        return [e for e in task.events if e.at <= elapsed]

    def start(self, ticker: str, horizon: str, as_of_date: date) -> str:
        task_id = uuid4().hex
        events, result = build_script(ticker, horizon, as_of_date)
        self._tasks[task_id] = _Task(
            task_id, ticker, horizon, as_of_date, self._clock(), events, result
        )
        return task_id

    def status(self, task_id: str) -> DebateStatus:
        task = self._task(task_id)
        done = self._finished(task)
        now = self._clock()
        if len(done) == len(task.events):
            return DebateStatus(
                status="DONE",
                round_number=done[-1].round_number,
                current_node="finalize",
                updated_at=now,
            )
        if not done:
            state = "PENDING" if self._elapsed(task) < QUEUE_DELAY_S else "RUNNING"
            return DebateStatus(status=state, round_number=1, updated_at=now)
        return DebateStatus(
            status="RUNNING",
            round_number=done[-1].round_number,
            current_node=done[-1].node,
            updated_at=now,
        )

    def history(self, task_id: str) -> list[HistoryEntry]:
        task = self._task(task_id)
        return [
            HistoryEntry(
                node=e.node,
                round_number=e.round_number,
                ts=task.started + timedelta(seconds=e.at),
                delta=e.delta,
            )
            for e in self._finished(task)
        ]

    def result(self, task_id: str) -> PredictResponse:
        task = self._task(task_id)
        status = self.status(task_id)
        finished = status.status == "DONE"
        return PredictResponse(
            task_id=task_id,
            ticker=task.ticker,
            horizon=task.horizon,
            status=status.status,
            as_of_date=task.as_of_date,
            model_version=MODEL_VERSION,
            result=PredictResult.model_validate(task.result) if finished else None,
            created_at=task.started,
            finished_at=task.started + timedelta(seconds=task.duration)
            if finished
            else None,
        )


_MOCK = MockClient()


def get_mock_client() -> MockClient:
    """Process-wide singleton, so tasks survive Streamlit reruns."""
    return _MOCK
