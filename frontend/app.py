"""Streamlit entrypoint: page routing + sidebar.

Run from the repo root:  streamlit run frontend/app.py
Talks to the FastAPI backend at $API_URL (default http://localhost:8000).
"""

import config
import streamlit as st
from api_client import get_client
from components import inject_css

st.set_page_config(
    page_title="MAS — Multi-Agent Financial System",
    page_icon="📈",
    layout="wide",
)

st.session_state.setdefault("runs", [])  # tasks started/opened in this session
st.session_state.setdefault("task_id", None)
st.session_state.setdefault("finished", set())  # task_ids seen in a terminal state


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
    st.caption(f"API: {config.API_URL}")
    if get_client().health():
        st.caption(":green[●] Backend online")
    else:
        st.error(
            "Backend is not reachable. Start the stack with "
            "`docker compose up --build`.",
            icon=":material/cloud_off:",
        )

inject_css()
nav.run()
