"""One-off smoke test for src/agents/llm_client.py.

NOT part of tests/unit — those mock the network entirely (see
tests/unit/test_agents_nodes.py), so they never actually exercise
generate_structured()'s HTTP call. This script makes real, cheap calls
to confirm:
  - Application Default Credentials resolve and a token is issued
  - the MaaS endpoint/model id in llm_client.py are still correct
  - structured output actually parses into a Pydantic model
  - the default temperature=0 (see docs/evaluation.md — backtests need
    reproducible output) actually produces the same answer twice for the
    same prompt

Prereqs (one-time):
    gcloud auth application-default login
    # in .env (see .env.example): GCP_PROJECT_ID=<your project>, GCP_LOCATION=<region>

Usage:
    uv run python tests/integration/test_llm_client.py

Delete this file (or move it into tests/unit as a proper, mocked test) once
llm_client.py has been exercised against a real backtest run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel  # noqa: E402 — must import after load_dotenv()

from src.agents.llm_client import MODEL_VERSION, generate_structured  # noqa: E402


class Ping(BaseModel):
    answer: str


def main() -> None:
    print(f"Calling {MODEL_VERSION} ...")
    prompt = (
        "Reply with a JSON object where the 'answer' field is exactly the "
        "word OK, nothing else."
    )
    result = generate_structured(prompt, Ping)
    print("SUCCESS:", result)

    print("Calling again with the same prompt to check temperature=0 determinism ...")
    result_repeat = generate_structured(prompt, Ping)
    if result == result_repeat:
        print("DETERMINISM OK:", result_repeat)
    else:
        print(f"DETERMINISM WARNING: got {result_repeat!r}, expected {result!r}")


if __name__ == "__main__":
    main()
