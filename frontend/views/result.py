"""Page 3 — final prediction of a finished debate."""

import json

import streamlit as st
from api_client import get_client
from components import (
    direction_badge,
    md,
    page_footer,
    render_rounds,
    render_transcript,
)
from debate_model import AGENT_LABELS, AGENTS, build_rounds
from models import ApiError

st.title("Result")

task_id: str | None = st.session_state.task_id
if not task_id:
    st.info(
        "No debate selected. Start one from **New analysis** or open one from "
        "**History**."
    )
    st.stop()

client = get_client(st.session_state.mock_mode)
try:
    resp = client.result(task_id)
    try:
        history = client.history(task_id)
    except ApiError:
        history = []  # transcript expired; the prediction may still be there
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if resp.status in {"PENDING", "RUNNING"}:
    st.info("This debate is still running.", icon=":material/hourglass_top:")
    if st.button("Open live view", type="primary"):
        st.switch_page("views/live_debate.py")
    st.stop()

if resp.status == "FAILED" or resp.result is None:
    st.error(resp.error or "The debate failed and produced no prediction.")
    st.stop()

result = resp.result
st.caption(
    f"{resp.ticker} · horizon {resp.horizon} · as of "
    f"{resp.as_of_date.isoformat() if resp.as_of_date else '?'}"
)

left, right = st.columns([1, 2], vertical_alignment="center")
left.markdown(direction_badge(result.direction, big=True), unsafe_allow_html=True)
with right:
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{result.confidence:.0%}")
    m2.metric("Rounds used", f"{result.rounds_used} / 3")
    m3.metric(
        "Critic",
        "Agreed" if result.critic_agreed else "Did not agree",
        help=None
        if result.critic_agreed
        else "The round limit was reached before the Critic agreed.",
    )
st.progress(result.confidence, text="PM's self-reported confidence")

st.subheader("Summary")
st.markdown(md(result.summary))

rounds = build_rounds(history)
final_opinion = next((r.pm_opinion for r in reversed(rounds) if r.pm_opinion), None)
if final_opinion and final_opinion.arguments:
    st.subheader("Key arguments")
    for arg in final_opinion.arguments:
        st.markdown(f"- {md(arg)}")

st.subheader("Agent reports")
tabs = st.tabs([f"{AGENT_LABELS[a]}" for a in AGENTS])
for tab, agent in zip(tabs, AGENTS):
    report = getattr(result.agent_reports, agent)
    with tab:
        st.markdown(f"**{md(report.opinion)}**")
        st.markdown(md(report.reasoning))

if rounds:
    st.subheader("Debate")
    render_rounds(rounds, live=False)
    with st.expander("Full transcript"):
        render_transcript(history)

st.subheader("Run details")
d1, d2, d3 = st.columns(3)
d1.caption(f"Model: `{resp.model_version or 'n/a'}`")
d2.caption(f"Task: `{task_id}`")
if resp.created_at and resp.finished_at:
    secs = (resp.finished_at - resp.created_at).total_seconds()
    d3.caption(f"Duration: {secs:.0f} s")

st.download_button(
    "Download result (JSON)",
    data=json.dumps(resp.model_dump(mode="json"), indent=2),
    file_name=f"prediction_{resp.ticker}_{task_id[:8]}.json",
    mime="application/json",
)
page_footer()
