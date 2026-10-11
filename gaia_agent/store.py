"""answers.json handling: failed or UNKNOWN runs never overwrite a good answer."""
from __future__ import annotations

import json

from .config import data_dir

BAD_ANSWERS = {"", "UNKNOWN", "NOT FOUND"}


def answers_file():
    return data_dir() / "answers.json"


def load_answers() -> dict[str, str]:
    path = answers_file()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_answer(task_id: str, answer: str, status: str) -> None:
    """Merge one answer into answers.json. Errors are never stored, and UNKNOWN never replaces a stored answer."""
    if status != "ok":
        return
    answers = load_answers()
    if answer.strip().upper() in BAD_ANSWERS and task_id in answers:
        return
    answers[task_id] = answer
    answers_file().write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
