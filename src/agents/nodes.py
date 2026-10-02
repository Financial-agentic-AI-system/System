"""LangGraph nodes: row formatters, agent nodes, PM/Critic nodes, and the
final-result assembly node. The SQL they read lives in `src/db/queries.py`.
"""

from src.agents import llm_client
from src.agents.prompt_loader import render_prompt
from src.agents.state import (
    AgentReport,
    AgentReports,
    CriticFeedback,
    DebateState,
    NextAgentsSelection,
    PMOpinion,
    PredictResult,
)
from src.db.models import ArticleSummary, Fundamentals, MacroSeries, StockPrice
from src.db.queries import (
    fetch_fundamentals,
    fetch_macro_series,
    fetch_prices,
    fetch_recent_articles,
)
from src.db.session import get_session


# --------------------------------------------------------------------------- #
# Format (pure, unit-tested without a database)
# --------------------------------------------------------------------------- #
def format_fundamentals_context(rows: list[Fundamentals]) -> str:
    if not rows:
        return "No fundamentals data available."
    lines = []
    for r in rows:
        fields = ", ".join(f"{k}={v}" for k, v in r.data.items() if v is not None)
        lines.append(f"{r.report_type} ({r.fiscal_date}): {fields}")
    return "\n".join(lines)


def format_price_context(rows: list[StockPrice]) -> str:
    if not rows:
        return "No price data available."
    return "\n".join(f"{r.date}: {r.close}" for r in rows)


def format_macro_context(rows: list[MacroSeries]) -> str:
    """One line per series: latest value, plus the long-run trend against
    the earliest value in the fetched window (see fetch_macro_series).
    """
    if not rows:
        return "No macro data available."

    by_series: dict[str, list[MacroSeries]] = {}
    for row in rows:  # rows are ordered series_id, date desc within a series
        by_series.setdefault(row.series_id, []).append(row)

    lines = []
    for series_id, series_rows in by_series.items():
        latest = series_rows[0]
        if len(series_rows) == 1:
            lines.append(f"{series_id}: {latest.value} as of {latest.date}")
            continue
        earliest = series_rows[-1]
        change = latest.value - earliest.value
        lines.append(
            f"{series_id}: {latest.value} as of {latest.date} "
            f"(was {earliest.value} on {earliest.date}, change {change:+.2f})"
        )
    return "\n".join(lines)


def format_sentiment_context(rows: list[ArticleSummary]) -> str:
    if not rows:
        return "No recent articles available."
    return "\n".join(
        f"{r.time_published} [{r.overall_sentiment_label}] {r.title}: {r.summary}"
        for r in rows
    )


def format_report(report: AgentReport | None) -> str:
    if report is None:
        return "(no report yet)"
    return f"Opinion: {report.opinion}\nReasoning: {report.reasoning}"


def format_opinion(opinion: PMOpinion | None) -> str:
    if opinion is None:
        return "(no opinion yet)"
    return (
        f"Direction: {opinion.direction} (confidence {opinion.confidence})\n"
        f"Summary: {opinion.summary}\n"
        f"Arguments: {'; '.join(opinion.arguments)}"
    )


# --------------------------------------------------------------------------- #
# Worker agents — same shape, different data source + prompt file
# --------------------------------------------------------------------------- #
def financial_agent_node(state: DebateState) -> dict:
    with get_session() as session:
        fundamentals = fetch_fundamentals(session, state["ticker"], state["as_of_date"])
        prices = fetch_prices(session, state["ticker"], state["as_of_date"])

    common = dict(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        horizon=state["horizon"],
        fundamentals_context=format_fundamentals_context(fundamentals),
        price_context=format_price_context(prices),
    )

    if state["round_number"] == 1:
        prompt = render_prompt("financial_agent", **common)
    else:
        prompt = render_prompt(
            "financial_agent",
            key="revision_prompt",
            round_number=state["round_number"],
            critic_feedback=state["critic_feedback"].feedback,
            previous_opinion=format_report(state["financial_report"]),
            **common,
        )

    report = llm_client.generate_structured(prompt, AgentReport)
    return {"financial_report": report}


def sentiment_agent_node(state: DebateState) -> dict:
    with get_session() as session:
        articles = fetch_recent_articles(session, state["ticker"], state["as_of_date"])

    common = dict(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        horizon=state["horizon"],
        sentiment_context=format_sentiment_context(articles),
    )

    if state["round_number"] == 1:
        prompt = render_prompt("sentiment_agent", **common)
    else:
        prompt = render_prompt(
            "sentiment_agent",
            key="revision_prompt",
            round_number=state["round_number"],
            critic_feedback=state["critic_feedback"].feedback,
            previous_opinion=format_report(state["sentiment_report"]),
            **common,
        )

    report = llm_client.generate_structured(prompt, AgentReport)
    return {"sentiment_report": report}


def macro_agent_node(state: DebateState) -> dict:
    with get_session() as session:
        macro_rows = fetch_macro_series(session, state["as_of_date"])

    common = dict(
        ticker=state["ticker"],
        as_of_date=state["as_of_date"],
        horizon=state["horizon"],
        macro_context=format_macro_context(macro_rows),
    )

    if state["round_number"] == 1:
        prompt = render_prompt("macro_agent", **common)
    else:
        prompt = render_prompt(
            "macro_agent",
            key="revision_prompt",
            round_number=state["round_number"],
            critic_feedback=state["critic_feedback"].feedback,
            previous_opinion=format_report(state["macro_report"]),
            **common,
        )

    report = llm_client.generate_structured(prompt, AgentReport)
    return {"macro_report": report}


# --------------------------------------------------------------------------- #
# Portfolio Manager — two distinct jobs, two distinct prompts
# --------------------------------------------------------------------------- #
def pm_synthesize_node(state: DebateState) -> dict:
    reports = dict(
        financial_report=format_report(state["financial_report"]),
        sentiment_report=format_report(state["sentiment_report"]),
        macro_report=format_report(state["macro_report"]),
    )

    if state["pm_opinion"] is None:
        prompt = render_prompt(
            "portfolio_manager",
            key="initial_prompt",
            ticker=state["ticker"],
            horizon=state["horizon"],
            as_of_date=state["as_of_date"],
            **reports,
        )
    else:
        prompt = render_prompt(
            "portfolio_manager",
            key="revision_prompt",
            ticker=state["ticker"],
            horizon=state["horizon"],
            as_of_date=state["as_of_date"],
            round_number=state["round_number"],
            critic_feedback=state["critic_feedback"].feedback,
            previous_opinion=format_opinion(state["pm_opinion"]),
            reasked_agents=", ".join(state["active_agents"]),
            **reports,
        )

    opinion = llm_client.generate_structured(prompt, PMOpinion)
    return {"pm_opinion": opinion}


def pm_select_next_agents_node(state: DebateState) -> dict:
    prompt = render_prompt(
        "portfolio_manager",
        key="select_agents_prompt",
        ticker=state["ticker"],
        round_number=state["round_number"],
        critic_feedback=state["critic_feedback"].feedback,
        previous_opinion=format_opinion(state["pm_opinion"]),
        financial_report=format_report(state["financial_report"]),
        sentiment_report=format_report(state["sentiment_report"]),
        macro_report=format_report(state["macro_report"]),
    )
    selection = llm_client.generate_structured(prompt, NextAgentsSelection)
    return {
        "active_agents": selection.agents,
        "round_number": state["round_number"] + 1,
    }


def critic_node(state: DebateState) -> dict:
    prompt = render_prompt(
        "critic",
        ticker=state["ticker"],
        horizon=state["horizon"],
        as_of_date=state["as_of_date"],
        round_number=state["round_number"],
        pm_opinion=format_opinion(state["pm_opinion"]),
        financial_report=format_report(state["financial_report"]),
        sentiment_report=format_report(state["sentiment_report"]),
        macro_report=format_report(state["macro_report"]),
    )
    feedback = llm_client.generate_structured(prompt, CriticFeedback)
    return {"critic_feedback": feedback}


# --------------------------------------------------------------------------- #
# Final assembly — pure, no LLM call, fully unit-testable
# --------------------------------------------------------------------------- #
def finalize_node(state: DebateState) -> dict:
    opinion = state["pm_opinion"]
    return {
        "final_result": PredictResult(
            direction=opinion.direction,
            confidence=opinion.confidence,
            summary=opinion.summary,
            rounds_used=state["round_number"],
            critic_agreed=state["critic_feedback"].agree,
            agent_reports=AgentReports(
                financial=state["financial_report"],
                sentiment=state["sentiment_report"],
                macro=state["macro_report"],
            ),
        )
    }
