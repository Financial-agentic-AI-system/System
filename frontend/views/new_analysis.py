"""Page 1 — start a new debate."""

from datetime import date

import config
import streamlit as st
from api_client import get_client
from components import page_footer, remember_run
from models import ApiError

st.title("New analysis")
st.caption(
    "Three specialist agents (Financial, Sentiment, Macro) debate a stock under a "
    "Portfolio Manager and a Critic. The result is a BUY / HOLD / SELL call."
)

DEFAULT_AS_OF = date(2026, 1, 15)


@st.cache_data(ttl=300, show_spinner=False)
def _load_tickers() -> list[str]:
    return get_client().tickers()


try:
    tickers = _load_tickers()
except ApiError as exc:
    st.error(f"Cannot load the ticker list. {exc}")
    st.stop()
if not tickers:
    st.warning("The database has no tickers yet. Load the datalake first.")
    st.stop()

with st.form("new_analysis"):
    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Ticker", tickers)
    horizon = c2.selectbox(
        "Investment horizon",
        list(config.HORIZONS),
        format_func=lambda h: f"{h} — {config.HORIZONS[h]}",
    )
    as_of = c3.date_input(
        "As-of date",
        value=DEFAULT_AS_OF,
        min_value=config.BACKTEST_START,
        max_value=config.BACKTEST_END,
        help="The simulated 'today'. Agents only see data up to this date.",
    )
    submitted = st.form_submit_button("Start debate", type="primary")

st.info(
    "**Point-in-time analysis.** Every query the agents make is filtered to "
    "`date <= as-of date`, and the LLM's training cutoff (Dec 2023) precedes the "
    f"backtest window ({config.BACKTEST_START:%Y-%m-%d} – "
    f"{config.BACKTEST_END:%Y-%m-%d}), so the debate cannot see the future.",
    icon=":material/schedule:",
)

with st.expander("How the debate works"):
    st.markdown(
        """
1. The **Portfolio Manager** asks the Financial, Sentiment and Macro agents.
2. The agents answer in parallel with an opinion and their reasoning.
3. The PM synthesizes one opinion (direction, confidence, arguments).
4. The **Critic** checks the PM's opinion against the agents' reports.
5. If the Critic agrees, or after **3 rounds**, the prediction is returned.
   Otherwise the PM re-asks only the agents relevant to the Critic's objection.
"""
    )

if submitted:
    client = get_client()
    try:
        task_id = client.start(ticker, horizon, as_of)
    except ApiError as exc:
        st.error(str(exc))
    else:
        remember_run(task_id, ticker, horizon, as_of.isoformat())
        st.session_state.task_id = task_id
        st.switch_page("views/live_debate.py")

page_footer()
