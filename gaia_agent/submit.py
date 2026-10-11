"""Send the stored answers to the course scoring API."""
from __future__ import annotations

import json
import os

import requests

from .client import get_questions
from .config import BASE_URL, load_secrets
from .store import load_answers


def build_payload(username: str, code_url: str) -> tuple[dict, list[int]]:
    """Return (payload, missing question numbers). Missing answers are sent as UNKNOWN."""
    questions, stored = get_questions(), load_answers()
    answers, missing = [], []
    for idx, q in enumerate(questions, 1):
        answer = stored.get(q["task_id"])
        if answer is None:
            missing.append(idx)
            answer = "UNKNOWN"
        answers.append({"task_id": q["task_id"], "submitted_answer": answer})
    return {"username": username, "agent_code": code_url, "answers": answers}, missing


def submit(code_url: str, dry_run: bool = True, username: str | None = None):
    """Submit stored answers. code_url: your public Space, https://huggingface.co/spaces/<user>/<space>/tree/main"""
    load_secrets()
    username = (username or os.getenv("HF_USERNAME") or "").strip()
    if not username:
        raise ValueError("Set HF_USERNAME (your Hugging Face username).")
    if not (code_url.startswith("https://huggingface.co/spaces/") and code_url.rstrip("/").endswith("/tree/main")):
        print("Warning: the course expects the link of your public Space code (.../tree/main).")

    payload, missing = build_payload(username, code_url)
    total = len(payload["answers"])
    print(f"{total - len(missing)}/{total} questions have stored answers. Missing (sent as UNKNOWN): {missing}")
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return None
    resp = requests.post(f"{BASE_URL}/submit", json=payload, timeout=60)
    print(resp.status_code, resp.text)
    return resp
