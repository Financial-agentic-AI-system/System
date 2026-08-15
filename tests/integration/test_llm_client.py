"""One-off smoke test for src/agents/llm_client.py.

NOT part of tests/unit — those mock the network entirely (see
tests/unit/test_agents_nodes.py), so they never actually exercise
generate_structured()'s HTTP call. This script makes one real, cheap call
to confirm:
  - Application Default Credentials resolve and a token is issued
  - the MaaS endpoint/model id in llm_client.py are still correct
  - structured output actually parses into a Pydantic model

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

from pydantic import BaseModel  

from src.agents.llm_client import MODEL_VERSION, generate_structured 


class Ping(BaseModel):
    answer: str


def main() -> None:
    print(f"Calling {MODEL_VERSION} ...")
    result = generate_structured(
        "Reply with a JSON object where the 'answer' field is exactly the "
        "word OK, nothing else.",
        Ping,
    )
    print("SUCCESS:", result)


if __name__ == "__main__":
    main()
