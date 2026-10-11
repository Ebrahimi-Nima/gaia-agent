"""Run the agent over the course questions."""
from __future__ import annotations

import time
from pathlib import Path

from .agent import Agent
from .client import download_file, get_questions
from .llm import QuotaExhausted
from .store import BAD_ANSWERS, load_answers, save_answer
from .tools import build_tools


def build_prompt(question: dict, file_path, file_missing: bool = False) -> str:
    if file_path is not None:
        return f'{question["question"]}\n\nAn attached file is available at this absolute local path: {Path(file_path).resolve()}'
    if file_missing:
        return (
            f'{question["question"]}\n\nNOTE: the attached file could not be downloaded and is NOT available. '
            "Do not search the file system for it. If the answer depends on the file, reply with FINAL ANSWER: UNKNOWN."
        )
    return question["question"]


def select_questions(questions: list[dict], only=None, limit: int = 0, skip_files: bool = False):
    """Return (1-based index, question) pairs after applying the filters. `only` is a list or '1,3,5'."""
    indexed = list(enumerate(questions, 1))
    if isinstance(only, str):
        only = [int(x) for x in only.split(",") if x.strip()]
    if only:
        wanted = set(only)
        indexed = [(i, q) for i, q in indexed if i in wanted]
    if skip_files:
        indexed = [(i, q) for i, q in indexed if not q.get("file_name")]
    if limit:
        indexed = indexed[:limit]
    return indexed


def iter_questions(only=None, limit: int = 0, skip_files: bool = False, missing: bool = False,
                   verbose: bool = True, preview: int = 200, refresh: bool = False, pause: float = 5.0):
    """Run the agent question by question, saving each answer; yields one result dict per question.

    missing: skip questions that already have a good stored answer.
    pause: seconds to wait between questions (eases per-minute token limits).
    """
    questions = get_questions(refresh=refresh)
    agent = Agent(tools=build_tools())
    done = {tid for tid, a in load_answers().items() if a.strip().upper() not in BAD_ANSWERS} if missing else set()

    for idx, q in select_questions(questions, only, limit, skip_files):
        if q["task_id"] in done:
            print(f"Question {idx}: skipped (already answered)")
            continue
        print(f"\n=== Question {idx}/{len(questions)} [{q['task_id'][:8]}] ===")
        print(q["question"][:300])

        file_path = None
        if q.get("file_name"):
            try:
                file_path = download_file(q["task_id"], q["file_name"])
                print(f"(attached file: {file_path})")
            except Exception as e:
                print(f"(failed to download attached file: {e})")
        file_missing = bool(q.get("file_name")) and file_path is None

        start = time.time()
        try:
            answer = agent.run(build_prompt(q, file_path, file_missing), verbose=verbose, preview=preview)
            status = "ok"
        except QuotaExhausted as e:
            print(f"\nSTOPPED: {e}")
            return
        except Exception as e:
            answer, status = f"ERROR: {type(e).__name__}: {str(e)[:300]}", "error"
        elapsed = time.time() - start

        print(f"ANSWER: {answer}  ({elapsed:.1f}s)")
        save_answer(q["task_id"], answer, status)
        yield {"index": idx, "task_id": q["task_id"], "file": bool(q.get("file_name")),
               "status": status, "seconds": round(elapsed, 1), "answer": answer}
        if pause:
            time.sleep(pause)


def run_questions(**kwargs) -> list[dict]:
    """Same arguments as iter_questions; prints a summary and returns all results."""
    results = list(iter_questions(**kwargs))
    print("\n===== SUMMARY =====")
    for r in results:
        print(f'{r["index"]:>2} {"F" if r["file"] else "-"} {r["status"]:<5} {r["seconds"]:>6}s  {r["answer"][:80]!r}')
    return results
