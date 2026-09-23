"""Page 4 — debates started or opened in this browser session."""

import streamlit as st
from api_client import get_client
from components import open_task, page_footer
from models import ApiError

st.title("History")
st.caption(
    "There is no 'list debates' endpoint and Redis keeps debates for 24 h only, "
    "so this lists the debates opened in this session."
)

client = get_client(st.session_state.mock_mode)

runs = st.session_state.runs
if not runs:
    st.info("Nothing here yet — start an analysis first.")
else:
    header = st.columns([2, 1, 1, 1, 1, 1])
    for col, title in zip(
        header, ["Started", "Ticker", "Horizon", "As-of date", "Status", ""]
    ):
        col.markdown(f"**{title}**")
    for run in runs[:30]:
        try:
            status = client.status(run["task_id"]).status
        except ApiError:
            status = "EXPIRED"
        cols = st.columns([2, 1, 1, 1, 1, 1])
        cols[0].write(run["started_at"])
        cols[1].write(run["ticker"])
        cols[2].write(run["horizon"])
        cols[3].write(run["as_of_date"])
        cols[4].write(status)
        if cols[5].button("Open", key=f"open_{run['task_id']}"):
            open_task(client, run["task_id"])

st.divider()
st.subheader("Open by task ID")
with st.form("open_by_id"):
    typed = st.text_input("Task ID", placeholder="c3f1…")
    go = st.form_submit_button("Open")
if go and typed.strip():
    open_task(client, typed.strip())

page_footer()
