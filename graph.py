"""
YouTube Summarizer & Q&A — LangGraph Multi-Agent Workflow
==========================================================
Three specialized agents orchestrated by LangGraph:

  1. Transcriber  — fetches and chunks the YouTube transcript
  2. Summarizer   — produces a structured JSON summary via Groq
  3. Q&A          — answers user questions grounded in the transcript + summary
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


def summarizer_node(state: VideoState) -> dict:
    """
    Agent 2 — Summarizer
    Calls Groq to produce a structured JSON summary from the transcript.
    """
    try:
        client = _groq()
    except ValueError as exc:
        return {"error": str(exc)}

    # Trim to stay comfortably within the model's token budget
    text = state["transcript"]
    if len(text) > 15_000:
        text = text[:15_000] + "\n... [transcript truncated for length]"

    prompt = f"""You are analyzing a YouTube video transcript. Produce a structured summary.

TRANSCRIPT:
{text}

Respond with ONLY valid JSON — no markdown fences, no commentary:
{{
  "summary": "<comprehensive 2-3 paragraph overview of the video>",
  "key_points": [
    "<specific insight — include timestamp if available>",
    "<specific insight — include timestamp if available>",
    "<specific insight — include timestamp if available>",
    "<specific insight — include timestamp if available>",
    "<specific insight — include timestamp if available>"
  ],
  "topics": ["<topic>", "<topic>", "<topic>"],
  "video_type": "<tutorial|lecture|interview|review|vlog|other>",
  "difficulty": "<beginner|intermediate|advanced|general>"
}}

Provide 5-7 key_points and 3-5 topics. Make key_points specific and actionable."""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2_048,
        )
        raw = _strip_fences(resp.choices[0].message.content)
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "summary": resp.choices[0].message.content,
            "key_points": [],
            "topics": [],
        }
    except Exception as exc:
        return {"error": f"Summarization failed: {exc}"}

    return {
        "summary": data.get("summary", ""),
        "key_points": data.get("key_points", []),
        "topics": data.get("topics", []),
        "error": None,
    }


def qa_node(state: VideoState) -> dict:
    """
    Agent 3 — Q&A
    Answers user questions grounded in the transcript and summary.
    """
    try:
        client = _groq()
    except ValueError as exc:
        return {"error": str(exc)}

    transcript_ctx = state.get("transcript", "")[:10_000]
    kp_block = "\n".join(f"• {p}" for p in state.get("key_points", []))

    system_prompt = f"""You are an expert assistant answering questions about a YouTube video.
You have access to the video's full transcript and a pre-generated summary.

SUMMARY:
{state.get("summary", "Not available")}

KEY POINTS:
{kp_block}

FULL TRANSCRIPT (with timestamps):
{transcript_ctx}

Instructions:
- Answer accurately using only information from the video content.
- Reference specific timestamps (e.g. "At 02:15, the speaker explains...") when helpful.
- If the answer isn't in the transcript, say so clearly rather than guessing.
- Be concise but thorough."""

    # Build messages: system + last 10 history messages + new question
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(state.get("conversation_history", [])[-10:])
    messages.append({"role": "user", "content": state["question"]})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1_024,
        )
        return {"answer": resp.choices[0].message.content, "error": None}
    except Exception as exc:
        return {"error": f"Q&A failed: {exc}"}


# ─────────────────────────────── Routing ────────────────────────────────── #

def _route_entry(state: VideoState) -> str:
    """Entry point: route to transcriber for summarization, qa for questions."""
    return "transcriber" if state.get("mode", "summarize") == "summarize" else "qa"


def _route_post_transcription(state: VideoState) -> str:
    """After transcription: proceed to summarizer or surface the error."""
    return "end" if state.get("error") else "summarizer"


# ─────────────────────────────── Graph Assembly ─────────────────────────── #

def _build_graph():
    wf = StateGraph(VideoState)

    wf.add_node("transcriber", transcriber_node)
    wf.add_node("summarizer",  summarizer_node)
    wf.add_node("qa",          qa_node)

    # Entry routing: summarize path vs. Q&A path
    wf.add_conditional_edges(START, _route_entry, {
        "transcriber": "transcriber",
        "qa":          "qa",
    })

    # After transcription: surface errors or continue to summarizer
    wf.add_conditional_edges("transcriber", _route_post_transcription, {
        "summarizer": "summarizer",
        "end":        END,
    })

    wf.add_edge("summarizer", END)
    wf.add_edge("qa",         END)

    return wf.compile()


# Module-level compiled graph — imported by app.py
graph = _build_graph()
