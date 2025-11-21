"""
YouTube Summarizer & Q&A — Streamlit App
=========================================
Multi-agent AI system: LangGraph · Groq · Llama 3.1 8B · youtube-transcript-api
"""
import os

import streamlit as st

# ─────────────────────────────── Page config ────────────────────────────── #

st.set_page_config(
    page_title="YouTube Summarizer & Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── CSS ────────────────────────────────────── #

st.markdown(
    """
<style>
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    color: #fff;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero p  { margin: .4rem 0 0; opacity: .85; font-size: .95rem; }

.card {
    background: #fff;
    border-radius: 12px;
    padding: 1.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,.07);
    margin-bottom: .8rem;
}

.kp {
    background: #eef2ff;
    border-left: 4px solid #6366f1;
    padding: .45rem .9rem;
    border-radius: 0 8px 8px 0;
    margin: .3rem 0;
    font-size: .93rem;
}

.tag {
    display: inline-block;
    background: #6366f1;
    color: #fff;
    font-size: .78rem;
    padding: .2rem .65rem;
    border-radius: 20px;
    margin: .2rem .15rem;
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
    st.header("⚙️ Settings")

    api_key = _load_api_key()
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("✅ Groq API key ready")
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
    st.markdown("### 📖 How to use")
    st.markdown(
        "1. Paste a YouTube URL\n"
        "2. Click **▶ Process**\n"
        "3. Read the AI summary\n"
        "4. Ask questions about the video"
    )

    st.markdown("---")
    st.markdown("### 🛠 Stack")
    st.markdown(
        "- **LangGraph** — agent orchestration\n"
        "- **Groq** — ultra-fast LLM inference\n"
        "- **Llama 3.1 8B** — open-source LLM\n"
        "- **youtube-transcript-api** — captions\n"
        "- **Streamlit** — UI"
    )


# ─────────────────────────────── Hero header ────────────────────────────── #

st.markdown(
    """
<div class="hero">
  <h1>🎬 YouTube Summarizer &amp; Q&amp;A</h1>
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
    process_btn = st.button("▶ Process", type="primary", use_container_width=True)

if process_btn and url_input:
    st.info("Processing logic coming soon…")

# ─────────────────────────────── Welcome state ──────────────────────────── #

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
