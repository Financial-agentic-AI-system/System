"""Streamlit entrypoint: page routing + sidebar.

Run from the repo root:  streamlit run frontend/app.py
Talks to the FastAPI backend at $API_URL, or to the built-in mock backend
(sidebar toggle) when the API is not up yet.
"""

import config
import streamlit as st
from components import inject_css

st.set_page_config(
    page_title="MAS — Multi-Agent Financial System",
    page_icon="📈",
    layout="wide",
)

st.session_state.setdefault("mock_mode", config.default_mock())
st.session_state.setdefault("runs", [])  # tasks started/opened in this session
st.session_state.setdefault("task_id", None)
st.session_state.setdefault("finished", set())  # task_ids seen in a terminal state


def _on_mode_change() -> None:
    # task ids of one backend mean nothing to the other
    st.session_state.task_id = None
    st.session_state.runs = []
    st.session_state.finished = set()


pages = [
    st.Page(
        "views/new_analysis.py",
        title="New analysis",
        icon=":material/add_circle:",
        default=True,
    ),
    st.Page("views/live_debate.py", title="Live debate", icon=":material/forum:"),
    st.Page("views/result.py", title="Result", icon=":material/task_alt:"),
    st.Page("views/history.py", title="History", icon=":material/history:"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.toggle(
        "Mock backend",
        key="mock_mode",
        on_change=_on_mode_change,
        help="Simulated debates generated inside the frontend — no API needed.",
    )
    if st.session_state.mock_mode:
        st.caption("Simulated debates, no backend needed.")
    else:
        st.caption(f"API: {config.API_URL}")

inject_css()
nav.run()
