"""Unit tests for src/agents/prompt_loader.py."""

import pytest

from src.agents import prompt_loader
from src.agents.prompt_loader import render_prompt


def test_render_prompt_substitutes_placeholders():
    rendered = render_prompt(
        "financial_agent",
        ticker="TSLA",
        as_of_date="2025-01-01",
        horizon="1W",
        fundamentals_context="FUND_X",
        price_context="PRICE_Y",
    )
    assert "TSLA" in rendered
    assert "FUND_X" in rendered
    assert "PRICE_Y" in rendered
    assert "{" not in rendered  # no leftover placeholder


def test_render_prompt_default_key_is_system_prompt():
    kwargs = dict(
        ticker="TSLA",
        as_of_date="2025-01-01",
        horizon="1W",
        fundamentals_context="x",
        price_context="y",
    )
    assert render_prompt("financial_agent", **kwargs) == render_prompt(
        "financial_agent", key="system_prompt", **kwargs
    )


def test_render_prompt_missing_key_raises_with_available_keys_listed():
    # critic.yaml only defines "system_prompt" — no revision_prompt.
    with pytest.raises(KeyError, match="revision_prompt"):
        render_prompt("critic", key="revision_prompt", ticker="TSLA")


def test_render_prompt_missing_placeholder_raises_clear_error():
    with pytest.raises(KeyError, match="Missing placeholder"):
        render_prompt("critic", ticker="TSLA")  # missing horizon, as_of_date, ...


def test_render_prompt_unknown_agent_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        render_prompt("not_a_real_agent")


def test_load_yaml_is_cached():
    first = prompt_loader._load_yaml("critic")
    second = prompt_loader._load_yaml("critic")
    assert first is second
