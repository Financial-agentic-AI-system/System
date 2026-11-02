"""State shape for the LangGraph debate.

Financial/Sentiment/Macro agents are deliberately indistinguishable at the
type level — all three write an `AgentReport` to a different key of
`DebateState`. What differs between them lives in `nodes.py` (which data
they fetch) and `prompts/*.yaml` (what they're asked), not here.
"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Direction = Literal["BUY", "HOLD", "SELL"]
Stance = Literal["BULLISH", "NEUTRAL", "BEARISH"]

MAX_ROUNDS = 3


class AgentReport(BaseModel):
    reasoning: str
    opinion: str
    # Last, so the model reasons before it commits. No default on purpose: a
    # skipped field must fail validation, not silently become NEUTRAL.
    stance: Stance


class PMOpinion(BaseModel):
    arguments: list[str]
    summary: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)


class CriticFeedback(BaseModel):
    feedback: str
    agree: bool


class NextAgentsSelection(BaseModel):
    """Output of pm_select_next_agents_node: who to re-ask, and why."""

    agents: list[Literal["financial", "sentiment", "macro"]] = Field(min_length=1)


class AgentReports(BaseModel):
    financial: AgentReport
    sentiment: AgentReport
    macro: AgentReport


class RoundLog(BaseModel):
    """One debate round as seen by the Critic: who was asked, each agent's
    stance and the PM's call. Written by `critic_node`, used by the backtest.

    No Critic verdict here: every round but the last ended with a disagreement
    (that is why the debate went on), and the last one is
    `PredictResult.critic_agreed`.

    `stances` holds all three agents; the ones not in `active_agents` keep
    their stance from an earlier round.
    """

    round_number: int
    active_agents: list[str]
    stances: dict[str, Stance]
    pm_direction: Direction
    pm_confidence: float


class PredictResult(BaseModel):
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    rounds_used: int
    critic_agreed: bool
    agent_reports: AgentReports
    round_log: list[RoundLog] = []


class ArticleSource(TypedDict):
    """One article the sentiment agent found by searching; shown on the frontend."""

    id: int
    title: str | None
    url: str
    source: str | None
    published: str | None  # ISO timestamp
    similarity: float
    query: str  # the query that found it


class DebateState(TypedDict):
    ticker: str
    horizon: str
    as_of_date: date

    round_number: int  # not "round" — shadows the builtin
    active_agents: list[str]  # subset of ["financial", "sentiment", "macro"]

    financial_report: AgentReport | None
    sentiment_report: AgentReport | None
    macro_report: AgentReport | None

    sentiment_queries: Annotated[list[str], operator.add]
    sentiment_sources: Annotated[list[ArticleSource], operator.add]

    pm_opinion: PMOpinion | None
    critic_feedback: CriticFeedback | None
    round_log: Annotated[list[RoundLog], operator.add]
    final_result: PredictResult | None
