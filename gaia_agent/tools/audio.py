"""Audio transcription with local Whisper (optional dependency: pip install openai-whisper)."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from .base import make_tool

_MODEL = None


def _get_model():
    """Load Whisper lazily so importing this module stays fast."""
    global _MODEL
    if _MODEL is None:
        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = whisper.load_model(os.getenv("WHISPER_MODEL", "small"), device=device)
    return _MODEL


def transcribe_audio(path: str) -> str:
    """Transcribe a local audio file (mp3, wav, m4a, ...) to text."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    try:
        import torch

        result = _get_model().transcribe(str(p), fp16=torch.cuda.is_available())
    except Exception as e:
        return f"Transcription failed: {type(e).__name__}: {e}"
    return result["text"].strip() or "No speech detected."


def audio_tools() -> list[dict]:
    """Registered only when Whisper is installed, so the model never wastes steps on a missing tool."""
    if importlib.util.find_spec("whisper") is None:
        return []
    return [make_tool(
        "transcribe_audio",
        "Transcribe a local audio file (mp3, wav, m4a) to text with Whisper. "
        "Numbers and names may be slightly misheard: double-check anything that looks odd.",
        {"path": {"type": "string", "description": "Absolute local path of the audio file."}},
        ["path"],
        transcribe_audio,
    )]
