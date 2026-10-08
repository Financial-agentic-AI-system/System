"""Reusable Streamlit building blocks shared by the pages."""

from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import config
import streamlit as st
from api_client import get_client
from debate_model import AGENT_LABELS, AGENTS, RoundView, stage_states
from models import ApiError, HistoryEntry, Meta

CSS = """
<style>
.mas-badge {
  display: inline-block; padding: .3rem 1rem; border-radius: .6rem;
  font-weight: 700; letter-spacing: .06em; border: 1px solid transparent;
}
.mas-badge.big { font-size: 2.6rem; padding: .3rem 1.6rem; line-height: 1.3; }
.mas-badge.BUY {
  background: rgba(26,157,90,.15); color: #1a9d5a; border-color: rgba(26,157,90,.5);
}
.mas-badge.HOLD {
  background: rgba(217,154,0,.15); color: #c98d00; border-color: rgba(217,154,0,.5);
}
.mas-badge.SELL {
  background: rgba(214,69,69,.15); color: #d64545; border-color: rgba(214,69,69,.5);
}
.mas-pill {
  display: inline-block; padding: .2rem .75rem; margin: 0 .3rem .3rem 0;
  border-radius: 999px; font-size: .85rem; border: 1px solid rgba(128,128,128,.4);
}
.mas-pill.done {
  background: rgba(26,157,90,.14); border-color: rgba(26,157,90,.55);
}
.mas-pill.bad {
  background: rgba(214,69,69,.14); border-color: rgba(214,69,69,.55);
}
.mas-pill.running {
  background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.65);
  animation: mas-pulse 1.4s ease-in-out infinite;
}
.mas-pill.waiting { opacity: .55; }
.mas-pill.kept { opacity: .6; border-style: dashed; }
.mas-arrow { opacity: .45; margin-right: .3rem; }
@keyframes mas-pulse { 50% { opacity: .55; } }
</style>
"""

AVATARS = {
    "financial": "📊",
    "sentiment": "📰",
    "macro": "🌍",
    "pm": "🧑‍💼",
    "critic": "🧐",
}
_ICONS = {"done": "✓", "running": "●", "waiting": "○", "kept": "↺"}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def md(text: str) -> str:
    """Escape `$` — financial text otherwise turns into LaTeX."""
    return text.replace("$", r"\$")


def direction_badge(direction: str, big: bool = False) -> str:
    size = " big" if big else ""
    return f'<span class="mas-badge {direction}{size}">{direction}</span>'


def _pill(label: str, state: str) -> str:
    suffix = " (unchanged)" if state == "kept" else ""
    return (
        f'<span class="mas-pill {state}">{_ICONS.get(state, "")} {label}{suffix}</span>'
    )


def render_rounds(rounds: list[RoundView], live: bool) -> None:
    """One pipeline row per round: agents -> PM -> Critic."""
    for view in rounds:
        is_current = view is rounds[-1]
        states = stage_states(view, live and is_current)
        parts = [_pill(AGENT_LABELS[a], states[a]) for a in AGENTS]
        parts.append('<span class="mas-arrow">→</span>')
        parts.append(_pill("PM", states["pm"]))
        parts.append('<span class="mas-arrow">→</span>')
        if view.critic:
            verdict = "agrees" if view.critic.agree else "disagrees"
            css = "done" if view.critic.agree else "bad"
            parts.append(_pill(f"Critic {verdict}", css))
        else:
            parts.append(_pill("Critic", states["critic"]))
        st.markdown(f"**Round {view.number}**")
        st.markdown("".join(parts), unsafe_allow_html=True)


def user_tz() -> tzinfo:
    """The viewer's time zone, as reported by their browser.

    `datetime.astimezone()` / `datetime.now()` would use the *server's* zone,
    which is UTC inside the `frontend` container. Falls back to `config.
    TIMEZONE` ($TZ), then UTC.
    """
    for name in (st.context.timezone, config.TIMEZONE):
        if name:
            try:
                return ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                continue
    return timezone.utc


def _fmt_time(ts: datetime | None) -> str:
    if ts is None:
        return ""
    if ts.tzinfo is None:  # the backend always sends UTC
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(user_tz()).strftime("%H:%M:%S")


def render_transcript(history: list[HistoryEntry]) -> None:
    """Chat-style transcript of the whole debate, in order."""
    if not history:
        st.caption("No messages yet.")
        return
    for entry in history:
        node, delta = entry.node, entry.delta
        stamp = f"round {entry.round_number} · {_fmt_time(entry.ts)}"
        if node in AGENTS:
            report = delta[f"{node}_report"]
            with st.chat_message("assistant", avatar=AVATARS[node]):
                st.markdown(f"**{AGENT_LABELS[node]} Agent** · {stamp}")
                st.markdown(md(report["opinion"]))
                with st.expander("Reasoning"):
                    st.markdown(md(report["reasoning"]))
        elif node == "pm_synthesize":
            op = delta["pm_opinion"]
            with st.chat_message("assistant", avatar=AVATARS["pm"]):
                st.markdown(f"**Portfolio Manager** · {stamp}")
                st.markdown(
                    f"{direction_badge(op['direction'])} &nbsp; confidence "
                    f"**{op['confidence']:.0%}**",
                    unsafe_allow_html=True,
                )
                st.markdown(md(op["summary"]))
                for arg in op.get("arguments", []):
                    st.markdown(f"- {md(arg)}")
        elif node == "critic":
            fb = delta["critic_feedback"]
            with st.chat_message("assistant", avatar=AVATARS["critic"]):
                verdict = (
                    "agrees with the PM" if fb["agree"] else "disagrees with the PM"
                )
                st.markdown(f"**Critic** · {stamp}")
                st.markdown(f"**{verdict.capitalize()}.** {md(fb['feedback'])}")
        elif node == "pm_select_next_agents":
            names = ", ".join(AGENT_LABELS[a] for a in delta["active_agents"])
            with st.chat_message("assistant", avatar=AVATARS["pm"]):
                st.markdown(f"**Portfolio Manager** · {stamp}")
                st.markdown(f"Re-asking: **{names}**.")


def find_run(task_id: str) -> dict | None:
    return next((r for r in st.session_state.runs if r["task_id"] == task_id), None)


def remember_run(task_id: str, ticker: str, horizon: str, as_of: str) -> None:
    if find_run(task_id):
        return
    st.session_state.runs.insert(
        0,
        {
            "task_id": task_id,
            "ticker": ticker,
            "horizon": horizon,
            "as_of_date": as_of,
            "started_at": datetime.now(user_tz()).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def open_task(client, task_id: str) -> None:
    """Make `task_id` the current one and jump to the right page."""
    try:
        status = client.status(task_id)
        resp = client.result(task_id)
    except ApiError as exc:
        st.error(str(exc))
        return
    remember_run(
        task_id,
        resp.ticker or "?",
        resp.horizon or "?",
        resp.as_of_date.isoformat() if resp.as_of_date else "?",
    )
    st.session_state.task_id = task_id
    finished = status.status in {"DONE", "FAILED"}
    st.switch_page("views/result.py" if finished else "views/live_debate.py")


@st.cache_data(ttl=300, show_spinner=False)
def load_meta() -> Meta:
    """`GET /meta`, cached. Raises `ApiError` when the backend is down."""
    return get_client().meta()


def page_footer() -> None:
    text = "Academic research prototype (engineering thesis). Not financial advice."
    try:
        meta = load_meta()
    except ApiError:
        pass
    else:
        text += (
            " Data is limited to the "
            f"{meta.backtest_start:%Y-%m-%d} – {meta.backtest_end:%Y-%m-%d} "
            "backtest window."
        )
    st.caption(text)
