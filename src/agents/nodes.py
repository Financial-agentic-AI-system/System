"""LangGraph nodes: row formatters, agent nodes, PM/Critic nodes, and the
final-result assembly node. The SQL they read lives in `src/db/queries.py`.
"""

import logging
import os

from src.agents import llm_client
from src.agents.prompt_loader import render_prompt
from src.agents.state import (
    AgentReport,
    AgentReports,
    ArticleSource,
    CriticFeedback,
    DebateState,
    NextAgentsSelection,
    PMOpinion,
    PredictResult,
)
from src.agents.tools import SEARCH_ARTICLES_TOOL, extract_queries
from src.agents.volatility import (
    HISTORY_SESSIONS,
    HORIZON_LONG,
    VOL_SPAN,
    horizon_sigma,
)
from src.db.models import ArticleSummary, Fundamentals, MacroSeries, StockPrice
from src.db.queries import (
    count_articles_without_embedding,
    fetch_articles_by_ids,
    fetch_fundamentals,
    fetch_macro_series,
    fetch_prices,
    fetch_recent_articles,
)
from src.db.session import get_session

# Aliased: the model sees the *tool* as `search_articles`; this is the retriever.
from src.retriever.search import search_articles as retrieve_articles

logger = logging.getLogger(__name__)

# Articles kept per query. The retriever's similarity threshold often returns
# fewer, or none — see docs/project_decisions.md.
SEARCH_LIMIT = 5


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


def render_extra_section(rows: list[ArticleSummary]) -> str:
    """The "additional articles" block of a sentiment prompt, or an empty
    string when the search found nothing (the section is simply left out).
    """
    if not rows:
        return ""
    return render_prompt(
        "sentiment_agent",
        key="extra_articles_section",
        extra_articles=format_sentiment_context(rows),
    )


def build_source(
    article: ArticleSummary, similarity: float, query: str
) -> ArticleSource:
    """What the frontend shows about one article the search found."""
    published = article.time_published
    return ArticleSource(
        id=article.id,
        title=article.title,
        url=article.url,
        source=article.source,
        published=published.isoformat() if published else None,
        similarity=round(similarity, 3),
        query=query,
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


def retrieval_enabled() -> bool:
    return os.environ.get("SENTIMENT_RETRIEVAL", "1") != "0"


def sentiment_agent_node(state: DebateState) -> dict:
    """Assess sentiment from the most recent articles, optionally widened by a
    semantic search the model asks for.

    Per round: (1) read the recent articles plus those found in earlier rounds;
    (2) ask the model whether it wants to search — it may call the
    `search_articles` tool with up to MAX_QUERIES queries, or decline; (3) run
    the queries against pgvector, with `ticker` and `as_of_date` taken from the
    state, never from the model; (4) have the model write its opinion.

    Returns only this round's queries and sources — the state's reducers
    accumulate them (see `DebateState`).
    """
    ticker, as_of = state["ticker"], state["as_of_date"]
    first_round = state["round_number"] == 1
    search_key, final_key = (
        ("search_prompt", "system_prompt")
        if first_round
        else ("revision_search_prompt", "revision_prompt")
    )
    prior_ids = [source["id"] for source in state["sentiment_sources"]]

    # Short DB session: the LLM calls below can take a while.
    with get_session() as session:
        recent = fetch_recent_articles(session, ticker, as_of)
        recent_ids = {article.id for article in recent}
        earlier = [
            article
            for article in fetch_articles_by_ids(session, ticker, as_of, prior_ids)
            if article.id not in recent_ids
        ]

    common = dict(
        ticker=ticker,
        as_of_date=as_of,
        horizon=state["horizon"],
        sentiment_context=format_sentiment_context(recent),
    )
    if not first_round:
        common.update(
            round_number=state["round_number"],
            critic_feedback=state["critic_feedback"].feedback,
            previous_opinion=format_report(state["sentiment_report"]),
            previous_queries="; ".join(state["sentiment_queries"]) or "(none)",
        )

    # Step 2: the model decides whether the recent articles are enough.
    queries: list[str] = []
    if retrieval_enabled():
        search_prompt = render_prompt(
            "sentiment_agent",
            key=search_key,
            extra_section=render_extra_section(earlier),
            **common,
        )
        queries = extract_queries(
            llm_client.generate_tool_calls(search_prompt, [SEARCH_ARTICLES_TOOL])
        )

    # Step 3: retrieval. An article found by several queries is kept once,
    # with its best similarity. Infrastructure errors (embedding API, SQL)
    # propagate on purpose: a debate that silently skipped retrieval would
    # look valid but not be comparable in a backtest.
    new: list[ArticleSummary] = []
    sources: list[ArticleSource] = []
    if queries:
        seen = recent_ids | {article.id for article in earlier}
        best: dict[int, tuple[ArticleSummary, float, str]] = {}
        with get_session() as session:
            missing = count_articles_without_embedding(session, ticker, as_of)
            if missing:
                logger.warning(
                    "%s: %d article(s) have no embedding yet; the search covers "
                    "a partial corpus (run `python -m src.db.load`)",
                    ticker,
                    missing,
                )
            for query in queries:
                hits = retrieve_articles(
                    session,
                    ticker=ticker,
                    as_of_date=as_of,
                    query=query,
                    limit=SEARCH_LIMIT,
                )
                logger.info(
                    "sentiment search %r -> %d hit(s), best similarity %.2f",
                    query,
                    len(hits),
                    hits[0][1] if hits else 0.0,
                )
                for article, similarity in hits:
                    if article.id in seen:
                        continue
                    if article.id not in best or similarity > best[article.id][1]:
                        best[article.id] = (article, similarity, query)
        new = [article for article, _, _ in best.values()]
        sources = [
            build_source(article, similarity, query)
            for article, similarity, query in best.values()
        ]

    # Step 4: the opinion, written against everything the agent has seen.
    final_prompt = render_prompt(
        "sentiment_agent",
        key=final_key,
        extra_section=render_extra_section(earlier + new),
        **common,
    )
    report = llm_client.generate_structured(final_prompt, AgentReport)

    return {
        "sentiment_report": report,
        "sentiment_queries": queries,
        "sentiment_sources": sources,
    }


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
def _label_definition(state: DebateState) -> dict:
    """Placeholders that tell the PM what BUY/HOLD/SELL mean for this stock.

    HOLD = the price stays within one standard deviation (see volatility.py).
    """
    with get_session() as session:
        prices = fetch_prices(
            session, state["ticker"], state["as_of_date"], lookback_days=450
        )
    # Same truncation as the backtest labels (scripts/evaluation/labels.py).
    closes_desc = [float(p.close) for p in prices[:HISTORY_SESSIONS]]
    sigma = horizon_sigma(closes_desc, state["horizon"])
    band = f"{sigma * 100:.1f}%" if sigma is not None else "its typical volatility"
    return dict(
        horizon_long=HORIZON_LONG[state["horizon"]],
        band_pct=band,
        vol_span=VOL_SPAN,
    )


def pm_synthesize_node(state: DebateState) -> dict:
    label_definition = _label_definition(state)
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
            **label_definition,
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
            **label_definition,
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
