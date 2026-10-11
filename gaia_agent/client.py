"""Questions and attachments (gated GAIA dataset first, course scoring API as fallback)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import requests

from .config import BASE_URL, GAIA_REPO, GAIA_SPLIT_DIR, data_dir


def get_questions(refresh: bool = False) -> list[dict]:
    """Return the benchmark questions, cached locally after the first download."""
    cache = data_dir() / "questions.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    resp = requests.get(f"{BASE_URL}/questions", timeout=30)
    resp.raise_for_status()
    questions = resp.json()
    cache.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    return questions


def download_file(task_id: str, file_name: str) -> Path:
    """Download the attachment of a task (cached). Tries the gated GAIA dataset first, then the scoring API."""
    target = data_dir() / "files" / file_name
    if target.exists():
        return target
    errors = []
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo_id=GAIA_REPO, repo_type="dataset",
                                filename=f"{GAIA_SPLIT_DIR}/{file_name}", token=os.getenv("HF_TOKEN"))
        shutil.copy(local, target)
        return target
    except Exception as e:
        errors.append(f"GAIA dataset: {type(e).__name__}: {str(e)[:150]}")
    try:
        resp = requests.get(f"{BASE_URL}/files/{task_id}", timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        return target
    except Exception as e:
        errors.append(f"scoring API: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError(" | ".join(errors))
