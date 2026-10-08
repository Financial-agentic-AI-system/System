"""The frontend keeps its own copy of the backend's response models (it does
not ship `src/`). These tests fail when the two drift apart."""

import sys
from pathlib import Path
from typing import get_args

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

import models as frontend  # noqa: E402

from src.agents import state  # noqa: E402
from src.api.v1 import schemas  # noqa: E402
from src.cache import schemas as cache  # noqa: E402

PAIRS = [
    (frontend.AgentReport, state.AgentReport),
    (frontend.AgentReports, state.AgentReports),
    (frontend.PMOpinion, state.PMOpinion),
    (frontend.CriticFeedback, state.CriticFeedback),
    (frontend.PredictResult, state.PredictResult),
    (frontend.HistoryEntry, cache.HistoryEntry),
    (frontend.DebateStatus, schemas.DebateStatusResponse),
    (frontend.PredictResponse, schemas.PredictResultResponse),
    (frontend.Meta, schemas.MetaResponse),
]


@pytest.mark.parametrize(
    ("frontend_model", "backend_model"),
    PAIRS,
    ids=[f.__name__ for f, _ in PAIRS],
)
def test_same_fields(frontend_model, backend_model):
    assert set(frontend_model.model_fields) == set(backend_model.model_fields)


def test_same_directions():
    assert set(get_args(frontend.Direction)) == set(get_args(state.Direction))


def test_same_statuses():
    assert set(get_args(frontend.StatusValue)) == set(get_args(cache.DebateStatusValue))
