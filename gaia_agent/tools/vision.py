"""Optional image tool: enabled when the VISION_MODEL setting is present (a LiteLLM model ID)."""
from __future__ import annotations

import base64
import os
from pathlib import Path

from .base import make_tool

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp"}


def describe_image(path: str, question: str) -> str:
    """Ask a vision-capable model about a local image."""
    import litellm

    model = os.getenv("VISION_MODEL")
    if not model:
        return "No vision model configured."
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    mime = MIME.get(p.suffix.lower())
    if not mime:
        return f"Unsupported image type '{p.suffix}'."
    key = os.getenv("OPENROUTER_API_KEY_1") if model.startswith("openrouter/") else os.getenv("GROQ_API_KEY_1")
    data = base64.b64encode(p.read_bytes()).decode()
    try:
        resp = litellm.completion(
            model=model, api_key=key, temperature=0, max_tokens=1200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            ]}],
        )
    except Exception as e:
        return f"Vision call failed: {type(e).__name__}: {str(e)[:200]}"
    return (resp.choices[0].message.content or "").strip()


def vision_tools() -> list[dict]:
    if not os.getenv("VISION_MODEL"):
        return []
    return [make_tool(
        "describe_image",
        "Describe an image or answer a question about it (a vision model is used). Free vision models can misread "
        "fine details such as chess pieces or small text: be sceptical.",
        {
            "path": {"type": "string", "description": "Absolute local path of the image."},
            "question": {"type": "string", "description": "What to look for in the image."},
        },
        ["path", "question"],
        describe_image,
    )]
