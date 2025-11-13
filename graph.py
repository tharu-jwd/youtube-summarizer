"""
YouTube Summarizer & Q&A — LangGraph Multi-Agent Workflow
==========================================================
Three specialized agents orchestrated by LangGraph:

  1. Transcriber  — fetches and chunks the YouTube transcript
  2. Summarizer   — produces a structured JSON summary via Groq  [in progress]
  3. Q&A          — answers user questions grounded in context   [in progress]
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

MODEL = "llama-3.1-8b-instant"


# ─────────────────────────────── Shared State ───────────────────────────── #

class VideoState(TypedDict):
    """Single state object threaded through all agent nodes."""

    # Inputs
    youtube_url: str
    mode: str               # "summarize" | "qa"
    question: Optional[str]

    # Transcriber outputs
    video_id: str
    transcript: str
    chunks: List[str]

    # Summarizer outputs
    summary: str
    key_points: List[str]
    topics: List[str]

    # Q&A outputs
    answer: Optional[str]
    conversation_history: List[dict]

    # Error propagation
    error: Optional[str]


# ─────────────────────────────── Utilities ──────────────────────────────── #

def _groq() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=key)


def _extract_video_id(url: str) -> str:
    """Extract the 11-character video ID from any YouTube URL format."""
    match = re.search(
        r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})", url
    )
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract a video ID from: {url!r}")


def _fmt_ts(seconds: float) -> str:
    t = int(seconds)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _chunk(text: str, size: int = 3_000) -> List[str]:
    """Split text into word-bounded chunks of approximately `size` characters."""
    words = text.split()
    chunks, buf, n = [], [], 0
    for w in words:
        buf.append(w)
        n += len(w) + 1
        if n >= size:
            chunks.append(" ".join(buf))
            buf, n = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks or [""]


def _strip_fences(text: str) -> str:
    """Remove accidental markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()
    return text


# ─────────────────────────────── Agent Nodes ────────────────────────────── #

def transcriber_node(state: VideoState) -> dict:
    """
    Agent 1 — Transcriber
    Fetches the YouTube transcript, formats it with timestamps, and chunks it.
    """
    try:
        vid = _extract_video_id(state["youtube_url"])
    except ValueError as exc:
        return {"error": str(exc)}

    api = YouTubeTranscriptApi()
    try:
        raw_entries = list(api.fetch(vid))
    except TranscriptsDisabled:
        return {
            "error": (
                "Captions are disabled for this video. "
                "Please try a different video that has auto-generated or manual captions."
            )
        }
    except NoTranscriptFound:
        try:
            tl = api.list(vid)
            t = tl.find_generated_transcript(["en", "en-US", "en-GB"])
            raw_entries = list(t.fetch())
        except Exception as exc:
            return {"error": f"No English transcript found for this video: {exc}"}
    except Exception as exc:
        return {"error": f"Transcript fetch failed: {exc}"}

    lines = [
        f"[{_fmt_ts(e.start)}] {e.text.strip().replace(chr(10), ' ')}"
        for e in raw_entries
    ]
    transcript = "\n".join(lines)

    return {
        "video_id": vid,
        "transcript": transcript,
        "chunks": _chunk(transcript),
        "error": None,
    }
