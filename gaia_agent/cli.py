"""Command line: python -m gaia_agent {check,ask,run,answers,submit}   (or: gaia-agent ...)"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_secrets


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="gaia-agent", description="Tool-using LLM agent for the GAIA course questions.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Test every API key with a tool-calling request")

    ask = sub.add_parser("ask", help="Ask the agent one question (no GAIA data or Hugging Face token needed)")
    ask.add_argument("question", help="The question, in quotes")
    ask.add_argument("--file", help="Optional local file (csv, xlsx, pdf, txt, py, audio...) the question refers to")
    ask.add_argument("--max-steps", type=int, default=12, help="Maximum number of tool-using steps")
    ask.add_argument("--quiet", action="store_true", help="Hide per-step tool logs")

    run = sub.add_parser("run", help="Run the agent on the course (GAIA) questions")
    run.add_argument("--only", default="", help="Comma-separated 1-based question numbers, e.g. 1,3,5")
    run.add_argument("--limit", type=int, default=0, help="Run at most N questions")
    run.add_argument("--skip-files", action="store_true", help="Skip questions with an attachment")
    run.add_argument("--missing", action="store_true", help="Skip questions that already have a good answer")
    run.add_argument("--refresh", action="store_true", help="Re-download the question list")
    run.add_argument("--quiet", action="store_true", help="Hide per-step tool logs")
    run.add_argument("--preview", type=int, default=200, help="Characters of each tool result to print")
    run.add_argument("--pause", type=float, default=5.0, help="Seconds to wait between questions")

    sub.add_parser("answers", help="Show the stored answers next to the questions")

    sp = sub.add_parser("submit", help="Submit stored answers to the course scoring API")
    sp.add_argument("--code-url", required=True, help="https://huggingface.co/spaces/<user>/<space>/tree/main")
    sp.add_argument("--dry-run", action="store_true", help="Print the payload instead of sending it")

    args = parser.parse_args(argv)
    print("Secrets found:", load_secrets())

    if args.command == "check":
        from .llm import healthcheck

        healthcheck()
    elif args.command == "ask":
        from .agent import Agent
        from .runner import build_prompt
        from .tools import build_tools

        file_path = Path(args.file) if args.file else None
        if file_path is not None and not file_path.exists():
            raise SystemExit(f"File not found: {file_path}")
        prompt = build_prompt({"question": args.question}, file_path)
        try:
            answer = Agent(tools=build_tools()).run(prompt, max_steps=args.max_steps, verbose=not args.quiet)
        except RuntimeError as e:  # no API keys, or the provider quota is exhausted
            raise SystemExit(f"Stopped: {e}")
        print(f"\nFINAL ANSWER: {answer}")
    elif args.command == "run":
        from .runner import run_questions

        run_questions(only=args.only, limit=args.limit, skip_files=args.skip_files, missing=args.missing,
                      refresh=args.refresh, verbose=not args.quiet, preview=args.preview, pause=args.pause)
    elif args.command == "answers":
        from .client import get_questions
        from .store import load_answers

        stored = load_answers()
        for i, q in enumerate(get_questions(), 1):
            flag = "F" if q.get("file_name") else "-"
            print(f"{i:>2} {flag} {stored.get(q['task_id'], '—')!r:<40} {q['question'][:70]!r}")
    elif args.command == "submit":
        from .submit import submit

        try:
            submit(args.code_url, dry_run=args.dry_run)
        except ValueError as e:
            raise SystemExit(str(e))
