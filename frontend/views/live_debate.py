"""Page 2 — live view of a running debate (polls status + history)."""

import config
import streamlit as st
from api_client import get_client
from components import find_run, page_footer, render_rounds, render_transcript
from debate_model import build_rounds, describe_progress, is_terminal
from models import ApiError

st.title("Live debate")

task_id: str | None = st.session_state.task_id
if not task_id:
    st.info(
        "No debate selected. Start one from **New analysis** or open one from "
        "**History**."
    )
    st.stop()

client = get_client(st.session_state.mock_mode)


def _header() -> None:
    run = find_run(task_id)
    if not run:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticker", run["ticker"])
    c2.metric("Horizon", run["horizon"])
    c3.metric("As-of date", run["as_of_date"])


def _render_state() -> bool:
    """Draw the current debate state. Returns True once it is terminal."""
    try:
        status = client.status(task_id)
        history = client.history(task_id)
    except ApiError as exc:
        st.error(str(exc))
        return True  # stop polling; the user can navigate away and back

    rounds = build_rounds(history)
    live = not is_terminal(status)

    message = describe_progress(rounds, status)
    if status.status == "FAILED":
        st.error(message)
    elif live:
        st.info(f"{message}", icon=":material/hourglass_top:")
    st.progress(
        min(status.round_number, config.MAX_ROUNDS) / config.MAX_ROUNDS,
        text=f"Round {min(status.round_number, config.MAX_ROUNDS)} of "
        f"{config.MAX_ROUNDS} (max)",
    )
    render_rounds(rounds, live)

    st.subheader("Transcript")
    render_transcript(history)
    return is_terminal(status)


_header()

if task_id in st.session_state.finished:
    # Already seen finishing: successful runs go to the result page,
    # failed ones stay here as a static (non-polling) view.
    try:
        finished_ok = client.status(task_id).status == "DONE"
    except ApiError as exc:
        st.error(str(exc))
        st.stop()
    if finished_ok:
        st.switch_page("views/result.py")
    _render_state()
else:

    @st.fragment(run_every=config.POLL_SECONDS)
    def _live_panel() -> None:
        if _render_state():
            st.session_state.finished.add(task_id)
            st.rerun()

    _live_panel()

page_footer()
