"""Unit tests for response parsing in src/agents/llm_client.py — no network."""

import json

import pytest
from pydantic import BaseModel

from src.agents import llm_client


class Selection(BaseModel):
    agents: list[str]


def _reply(answer: object) -> dict:
    return {"choices": [{"message": {"content": json.dumps(answer)}}]}


@pytest.mark.parametrize(
    "answer",
    [
        {"agents": ["financial"]},
        # Llama 4 wrapping the answer as a function call
        {
            "type": "function",
            "name": "Selection",
            "parameters": {"agents": ["financial"]},
        },
        {"type": "function", "name": "x", "arguments": '{"agents": ["financial"]}'},
    ],
)
def test_generate_structured_accepts_plain_and_wrapped_answers(monkeypatch, answer):
    monkeypatch.setattr(
        llm_client, "_call", lambda body, parse, *a, **k: parse(_reply(answer))
    )

    assert llm_client.generate_structured("p", Selection) == Selection(
        agents=["financial"]
    )


def test_unrelated_type_field_is_left_alone():
    answer = {"type": "BUY", "agents": []}
    assert llm_client._unwrap_function_call(answer) is answer
