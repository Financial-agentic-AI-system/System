"""Tools the agents may ask the model to call.

The schema and the reading of the model's request live here. Execution is in
`nodes.py`, because it needs the ticker, the `as_of_date` and a DB session —
none of which the model may choose. That is why `SearchArticlesArgs` has no
`ticker` and no `as_of_date`: the point-in-time bound cannot be argued away.
"""

from pydantic import BaseModel, ValidationError

from src.agents.llm_client import ToolCall

SEARCH_TOOL_NAME = "search_articles"
MAX_QUERIES = 3


class SearchArticlesArgs(BaseModel):
    """Search news article summaries about the stock for context the recent
    articles do not cover (regulation, competition, earnings, supply chain).
    Pass up to 3 distinct, specific queries."""

    queries: list[str]


SEARCH_ARTICLES_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": SearchArticlesArgs.__doc__,
        "parameters": SearchArticlesArgs.model_json_schema(),
    },
}


def extract_queries(calls: list[ToolCall]) -> list[str]:
    """Clean queries from the model's tool calls: only our tool, validated by
    the same model that produced the schema, stripped, de-duplicated
    case-insensitively, capped at MAX_QUERIES.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for call in calls:
        if call.name != SEARCH_TOOL_NAME:
            continue
        try:
            args = SearchArticlesArgs.model_validate(call.arguments)
        except ValidationError:
            continue  # e.g. {"queries": "text"} or a missing key
        for query in (q.strip() for q in args.queries):
            if query and query.lower() not in seen:
                seen.add(query.lower())
                queries.append(query)
                if len(queries) == MAX_QUERIES:
                    return queries
    return queries
