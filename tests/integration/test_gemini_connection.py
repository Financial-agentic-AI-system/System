"""Standalone connectivity check for Gemini Enterprise Agent Platform.

NOT part of the project's test suite (tests/unit) and not wired into
src/. One-off script to answer a single question: does the API key
still authenticate against the platform?

Makes at most a few tiny generateContent calls (short prompt, small/cheap
model, low max_output_tokens) against a short list of candidate regions
plus the "global" endpoint, stopping at the first success.

Usage:
    python test_gemini_connection.py <PROJECT_ID> <API_KEY>
"""

import sys
import urllib.error
import urllib.request
import json

MODEL = "gemini-2.5-flash-lite"
CANDIDATE_LOCATIONS = ["global", "europe-west1", "europe-west4", "europe-central2"]


def try_location(project_id: str, api_key: str, location: str) -> tuple[bool, str]:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    url = (
        f"https://{host}/v1/projects/{project_id}/locations/{location}"
        f"/publishers/google/models/{MODEL}:generateContent"
    )
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK."}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }
    ).encode()

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return True, resp.read().decode()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:500]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python test_gemini_connection.py <PROJECT_ID> <API_KEY>")
        sys.exit(1)

    project_id, api_key = sys.argv[1], sys.argv[2]

    for location in CANDIDATE_LOCATIONS:
        print(f"--- trying location={location} ---")
        ok, detail = try_location(project_id, api_key, location)
        if ok:
            print(f"SUCCESS at location={location}")
            print(detail)
            sys.exit(0)
        else:
            print(f"failed: {detail}")

    print("\nAll candidate locations failed. Most likely causes:")
    print("  - the API key is incorrect or has been deleted")
    print("  - the API key has restrictions (e.g., IP address limits) that block this request")
    print("  - the model isn't enabled/available in this project or region")
    sys.exit(1)


if __name__ == "__main__":
    main()