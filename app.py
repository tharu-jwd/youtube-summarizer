"""
YouTube Summarizer & Q&A — Streamlit App
=========================================
Multi-agent AI system: LangGraph · Groq · Llama 3.1 8B · youtube-transcript-api
"""
import os

import streamlit as st

from graph import VideoState, graph

# ─────────────────────────────── Page config ────────────────────────────── #

st.set_page_config(
    page_title="YouTube Summarizer & Q&A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── CSS ────────────────────────────────────── #

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
.hero { background: linear-gradient(135deg,#6366f1,#8b5cf6);
        border-radius:14px; padding:2rem; color:#fff; text-align:center; margin-bottom:1.5rem; }
.hero h1 { margin:0; font-size:2rem; }
.hero p  { margin:.4rem 0 0; opacity:.85; font-size:.95rem; }
.card { background:#fff; border-radius:12px; padding:1.5rem;
        box-shadow:0 1px 4px rgba(0,0,0,.08); margin-bottom:1rem; }
.kp { background:#eef2ff; border-left:4px solid #6366f1;
           padding:.5rem .9rem; border-radius:0 8px 8px 0; margin:.35rem 0; }
.tag { display:inline-block; background:#6366f1; color:#fff; font-size:.78rem;
       padding:.2rem .65rem; border-radius:20px; margin:.2rem .15rem; }
.tx-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    font-family: monospace;
    font-size: .82rem;
    max-height: 500px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────── Session state ──────────────────────────── #

_DEFAULTS: dict = {
    "transcript":           None,
    "chunks":               [],
    "summary":              None,
    "key_points":           [],
    "topics":               [],
    "conversation_history": [],
    "video_id":             None,
    "processed_url":        None,
    "error":                None,
    "groq_api_key":         "",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────── API key helper ──────────────────────────── #

def _load_api_key() -> str:
    """Priority: environment variable → Streamlit secrets → user input."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not key:
        key = st.session_state.groq_api_key
    return key


# ─────────────────────────────── Sidebar ────────────────────────────────── #

with st.sidebar:
    st.header("Settings")

    api_key = _load_api_key()
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("Groq API key ready")
    else:
        user_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free at [console.groq.com](https://console.groq.com)",
        )
        if user_key:
            st.session_state.groq_api_key = user_key
            os.environ["GROQ_API_KEY"] = user_key
            st.success("Key saved for this session!")

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Paste a YouTube URL\n"
        "2. Click **Process**\n"
        "3. Read the AI summary\n"
        "4. Ask questions in the **Chat Q&A** tab"
    )

    st.markdown("---")
    st.markdown("### Stack")
    st.markdown(
        "- **LangGraph** — agent orchestration\n"
        "- **Groq** — ultra-fast LLM inference\n"
        "- **Llama 3.1 8B** — open-source LLM\n"
        "- **youtube-transcript-api** — captions\n"
        "- **Streamlit** — UI"
    )

    if st.session_state.transcript:
        st.markdown("---")
        if st.button("Clear & Reset", use_container_width=True):
            for _k in _DEFAULTS:
                st.session_state[_k] = _DEFAULTS[_k]
            st.rerun()


# ─────────────────────────────── Hero header ────────────────────────────── #

st.markdown(
    """
<div class="hero">
  <h1>YouTube Summarizer &amp; Q&amp;A</h1>
  <p>Multi-agent AI · LangGraph · Groq · Llama 3.1 8B Instant</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────── URL input ──────────────────────────────── #

col_url, col_btn = st.columns([5, 1])
with col_url:
    url_input = st.text_input(
        "url",
        placeholder="Paste a YouTube URL here…",
        label_visibility="collapsed",
    )
with col_btn:
    process_btn = st.button("Process", type="primary", use_container_width=True)

# ─────────────────────────────── Processing ─────────────────────────────── #

if process_btn and url_input:
    if not _load_api_key():
        st.error("Please enter your Groq API key in the sidebar first.")
    elif url_input == st.session_state.processed_url:
        st.info("This video is already loaded. Ask questions below!")
    else:
        st.session_state.conversation_history = []
        st.session_state.error = None

        initial_state = VideoState(
            youtube_url=url_input,
            mode="summarize",
            question=None,
            video_id="",
            transcript="",
            chunks=[],
            summary="",
            key_points=[],
            topics=[],
            answer=None,
            conversation_history=[],
            error=None,
        )

        accumulated: dict = {}
        error_occurred = False

        with st.status("Processing video...", expanded=True) as proc_status:
            proc_status.write("Fetching transcript from YouTube...")

            for chunk in graph.stream(initial_state, stream_mode="updates"):
                for node_name, updates in chunk.items():
                    accumulated.update(updates)

                    if updates.get("error"):
                        proc_status.update(
                            label=f"Error — {updates['error'][:60]}",
                            state="error",
                            expanded=True,
                        )
                        error_occurred = True
                        break

                    if node_name == "transcriber":
                        n = len(updates.get("chunks", []))
                        proc_status.write(
                            f"Transcript ready — {n} segments · "
                            f"{len(updates.get('transcript', '')):,} characters"
                        )
                        proc_status.write("Generating AI summary…")

                    elif node_name == "summarizer":
                        proc_status.write(
                            f"Summary generated — "
                            f"{len(updates.get('key_points', []))} key points"
                        )

                if error_occurred:
                    break

            if not error_occurred:
                proc_status.update(
                    label="Video processed!", state="complete", expanded=False
                )

        if accumulated.get("error"):
            st.session_state.error = accumulated["error"]
        elif not error_occurred:
            st.session_state.transcript    = accumulated.get("transcript", "")
            st.session_state.chunks        = accumulated.get("chunks", [])
            st.session_state.summary       = accumulated.get("summary", "")
            st.session_state.key_points    = accumulated.get("key_points", [])
            st.session_state.topics        = accumulated.get("topics", [])
            st.session_state.video_id      = accumulated.get("video_id", "")
            st.session_state.processed_url = url_input

        st.rerun()

# ─────────────────────────────── Error display ──────────────────────────── #

if st.session_state.error:
    st.error(st.session_state.error)

# ─────────────────────────────── Results ────────────────────────────────── #

if st.session_state.summary:
    vid_id = st.session_state.video_id

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown(
            f"""<div class="card" style="padding:.7rem;">
  <iframe width="100%" height="220"
    src="https://www.youtube.com/embed/{vid_id}"
    frameborder="0" allowfullscreen
    style="border-radius:8px; display:block;"></iframe>
</div>""",
            unsafe_allow_html=True,
        )

        if st.session_state.topics:
            tags_html = " ".join(
                f'<span class="tag">{t}</span>'
                for t in st.session_state.topics
            )
            st.markdown(
                f'<div class="card">{tags_html}</div>', unsafe_allow_html=True
            )

    with right_col:
        tab_summary, tab_qa, tab_transcript = st.tabs(
            ["Summary", "Chat Q&A", "Full Transcript"]
        )

        # ── Summary tab ──────────────────────────────────────────────────
        with tab_summary:
            st.markdown(
                f'<div class="card">{st.session_state.summary}</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.key_points:
                st.markdown("**Key Points**")
                for i, pt in enumerate(st.session_state.key_points, 1):
                    st.markdown(
                        f'<div class="kp"><b>{i}.</b> {pt}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Q&A tab ──────────────────────────────────────────────────────
        with tab_qa:
            st.caption("Ask anything about this video — I'll answer using the transcript.")

            for msg in st.session_state.conversation_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if question := st.chat_input("Ask about the video…"):
                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        qa_state = VideoState(
                            youtube_url=st.session_state.processed_url or "",
                            mode="qa",
                            question=question,
                            video_id=st.session_state.video_id or "",
                            transcript=st.session_state.transcript or "",
                            chunks=st.session_state.chunks or [],
                            summary=st.session_state.summary or "",
                            key_points=st.session_state.key_points or [],
                            topics=st.session_state.topics or [],
                            answer=None,
                            conversation_history=st.session_state.conversation_history[-10:],
                            error=None,
                        )
                        result = graph.invoke(qa_state)

                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        answer = result["answer"]
                        st.write(answer)
                        st.session_state.conversation_history.extend([
                            {"role": "user",      "content": question},
                            {"role": "assistant", "content": answer},
                        ])
                        st.rerun()

        # ── Transcript tab ───────────────────────────────────────────────
        with tab_transcript:
            dl_col, _ = st.columns([1, 3])
            with dl_col:
                st.download_button(
                    "Download .txt",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{vid_id}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            st.markdown(
                f'<div class="tx-box">{st.session_state.transcript}</div>',
                unsafe_allow_html=True,
            )

elif not st.session_state.error:
    st.markdown(
        """
<div class="card" style="text-align:center; padding:3rem; color:#555;">
  <h3 style="margin-bottom:.5rem;">Paste a YouTube URL above and click ▶ Process</h3>
  <p>Works best with videos that have auto-generated or manual captions.</p>
  <p style="margin-top:1.2rem; font-size:.85rem; opacity:.7;">
    Great for: talks · tutorials · lectures · interviews · podcasts
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
