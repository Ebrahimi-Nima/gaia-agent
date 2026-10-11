"""File tools: preview attachments and run Python (NOT a sandbox)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import make_tool

MAX_CHARS = 8000
PREVIEW_ROWS = 20
TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".xml", ".html", ".yaml", ".yml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def _clip(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text)} chars total]"


def read_file(path: str) -> str:
    """Preview a local file: spreadsheets (shape, columns, first rows), CSV, PDF and text/code files."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    suffix = p.suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            import pandas as pd

            parts = []
            for name, df in pd.read_excel(p, sheet_name=None).items():
                parts.append(
                    f"## Sheet '{name}': {df.shape[0]} rows x {df.shape[1]} columns\n"
                    f"Columns: {list(df.columns)}\n{df.head(PREVIEW_ROWS).to_csv(index=False)}"
                )
            return _clip("\n".join(parts))
        if suffix in {".csv", ".tsv"}:
            import pandas as pd

            df = pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",")
            return _clip(
                f"{df.shape[0]} rows x {df.shape[1]} columns\n"
                f"Columns: {list(df.columns)}\n{df.head(PREVIEW_ROWS).to_csv(index=False)}"
            )
        if suffix == ".pdf":
            import pymupdf
            import pymupdf4llm

            return _clip(pymupdf4llm.to_markdown(pymupdf.open(p)))
        if suffix in TEXT_SUFFIXES:
            return _clip(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return f"Failed to read {path}: {type(e).__name__}: {e}"
    if suffix in AUDIO_SUFFIXES:
        return "This is an audio file: use transcribe_audio."
    if suffix in IMAGE_SUFFIXES:
        return "This is an image: use describe_image if it is available; otherwise the answer cannot be determined."
    return f"Unsupported file type '{suffix}'."


def run_python(code: str = "", file_path: str = "", timeout: int = 60) -> str:
    """Run a Python snippet or an existing .py file in a subprocess and return stdout/stderr. NOT a sandbox."""
    if not code and not file_path:
        return "Provide either `code` or `file_path`."

    # Never pass API keys or tokens to generated code (this does not stop it reading files on disk).
    env = {
        k: v for k, v in os.environ.items()
        if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }

    with tempfile.TemporaryDirectory() as workdir:
        if file_path:
            script = Path(file_path).resolve()
            if not script.exists():
                return f"File not found: {file_path}"
        else:
            script = Path(workdir) / "snippet.py"
            script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=timeout, cwd=workdir, env=env,
            )
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout}s."

    result = f"exit code {proc.returncode}\n--- stdout ---\n{proc.stdout.strip()}"
    if proc.stderr.strip():
        result += f"\n--- stderr ---\n{proc.stderr.strip()[-2000:]}"
    return _clip(result)


FILE_TOOLS = [
    make_tool(
        "read_file",
        "Preview a local file by path: Excel/CSV (sheet names, shape, columns, first rows), PDF, text and code files. "
        "For spreadsheets this shows only the first rows: use run_python with pandas for totals and filtering.",
        {"path": {"type": "string", "description": "Absolute or relative local path."}},
        ["path"],
        read_file,
    ),
    make_tool(
        "run_python",
        "Execute Python (pandas and openpyxl are installed). Pass `code` for a snippet, or `file_path` to run an existing .py file. "
        "Use absolute file paths and print() everything you need to see.",
        {
            "code": {"type": "string", "description": "Python source code to run."},
            "file_path": {"type": "string", "description": "Path of a .py file to execute instead of `code`."},
        },
        [],
        run_python,
    ),
]
