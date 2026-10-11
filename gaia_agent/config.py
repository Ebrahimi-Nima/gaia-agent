"""Settings and secrets.

Secrets are read from (in this order of precedence): Kaggle Secrets, Colab Secrets, real environment variables,
and finally a `.env` file in the current directory (or a parent) when python-dotenv is installed. They are copied
into os.environ under the names the code expects, so the rest of the package only ever calls os.getenv().
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_URL = "https://agents-course-unit4-scoring.hf.space"
GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_SPLIT_DIR = "2023/validation"

SECRET_NAMES = (
    [f"GROQ_API_KEY_{i}" for i in range(1, 6)]
    + [f"OPENROUTER_API_KEY_{i}" for i in range(1, 6)]
    + ["HF_TOKEN", "HF_USERNAME", "EMAIL", "GROQ_MODEL", "GROQ_MODEL_2",
       "OPENROUTER_MODEL", "VISION_MODEL", "WHISPER_MODEL"]
)

# Name expected by the code -> name used in a secrets panel (Kaggle / Colab), when they differ.
# Set entries before calling load_secrets(), e.g.  SECRET_ALIASES["GROQ_API_KEY_1"] = "groq1"
SECRET_ALIASES: dict[str, str] = {}

_LOADED = False


def _load_dotenv() -> None:
    """Load a .env file from the current directory (or a parent). Never overrides real environment variables."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def _panel_getters() -> list:
    getters = []
    try:
        from kaggle_secrets import UserSecretsClient

        getters.append(UserSecretsClient().get_secret)
    except Exception:
        pass
    try:
        from google.colab import userdata

        getters.append(userdata.get)
    except Exception:
        pass
    return getters


def load_secrets(force: bool = False) -> list[str]:
    """Copy available secrets into os.environ (once). Returns the names that are available."""
    global _LOADED
    if force or not _LOADED:
        _load_dotenv()
        getters = _panel_getters()

        def lookup(name: str):
            for getter in getters:
                try:
                    value = getter(name)
                    if value:
                        return value
                except Exception:
                    pass  # missing secret / not attached to this notebook
            return os.getenv(name)

        for name in SECRET_NAMES:
            value = lookup(name)
            if not value and name in SECRET_ALIASES:
                value = lookup(SECRET_ALIASES[name])
            if value:
                os.environ[name] = value
        _LOADED = True
    return [n for n in SECRET_NAMES if os.getenv(n)]


def data_dir() -> Path:
    """Folder for questions, attachments and answers (git-ignored)."""
    custom = os.getenv("GAIA_DATA_DIR")
    if custom:
        path = Path(custom)
    elif Path("/kaggle/working").exists():
        path = Path("/kaggle/working/gaia_data")
    elif Path("/content").exists():
        path = Path("/content/gaia_data")
    else:
        path = Path("data")
    (path / "files").mkdir(parents=True, exist_ok=True)
    return path
