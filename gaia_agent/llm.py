"""LLM access: a LiteLLM Router with key pools, fallbacks and rate-limit handling."""
from __future__ import annotations

import logging
import os
import re
import time

from .config import load_secrets

_ROUTER = None
_ENTRY = None

MAX_AUTO_WAIT = 120  # seconds; longer waits are treated as an exhausted (daily) quota


class QuotaExhausted(RuntimeError):
    """The provider asked for a long wait before retrying (usually the daily token quota)."""


def _models() -> tuple[str, str, str]:
    return (
        os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-120b"),
        os.getenv("GROQ_MODEL_2", "groq/openai/gpt-oss-20b"),
        # Provider catalogs change often: check this ID if the health check fails.
        os.getenv("OPENROUTER_MODEL", "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"),
    )


def _deployments(group: str, model: str, prefix: str) -> list[dict]:
    out = []
    for i in range(1, 6):
        key = os.getenv(f"{prefix}_{i}")
        if key:
            out.append({"model_name": group, "litellm_params": {"model": model, "api_key": key}})
    return out


def build_router():
    """Return (router, entry_group). Groups and fallbacks are built from the keys that exist."""
    import litellm
    from litellm import Router

    litellm.suppress_debug_info = True
    for name in ("LiteLLM Router", "LiteLLM", "LiteLLM Proxy"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    load_secrets()
    groq, groq2, openrouter = _models()
    model_list = (
        _deployments("primary", groq, "GROQ_API_KEY")
        + _deployments("secondary", groq2, "GROQ_API_KEY")
        + _deployments("fallback", openrouter, "OPENROUTER_API_KEY")
    )
    if not model_list:
        raise RuntimeError("No API keys found. Set GROQ_API_KEY_1.. and/or OPENROUTER_API_KEY_1..")

    groups = [g for g in ("primary", "secondary", "fallback") if any(d["model_name"] == g for d in model_list)]
    fallbacks = [{g: groups[i + 1:]} for i, g in enumerate(groups[:-1])]
    router = Router(
        model_list=model_list,
        fallbacks=fallbacks,
        num_retries=4,
        timeout=60,
        routing_strategy="simple-shuffle",
        set_verbose=False,
    )
    return router, groups[0]


def get_router():
    global _ROUTER, _ENTRY
    if _ROUTER is None:
        _ROUTER, _ENTRY = build_router()
    return _ROUTER, _ENTRY


def _is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(s in text for s in ("ratelimit", "rate limit", "no deployments available", "429", "overloaded"))


def _retry_after(error) -> float | None:
    """Seconds suggested by the provider ('Please try again in 7m12.5s'), if present."""
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s)\s*)+)", str(error), re.I)
    if not m:
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1), re.I):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit.lower()]
    return total


def complete(**kwargs):
    """router.completion with temperature 0; waits for short rate limits, stops early on long ones."""
    router, entry = get_router()
    kwargs.setdefault("model", entry)
    kwargs.setdefault("temperature", 0)
    for attempt in range(1, 6):
        try:
            return router.completion(**kwargs)
        except Exception as e:
            if not _is_rate_limit(e) or attempt == 5:
                raise
            suggested = _retry_after(e)
            if suggested is not None and suggested > MAX_AUTO_WAIT:
                raise QuotaExhausted(
                    f"The provider asks for {suggested / 60:.0f} min before retrying "
                    f"(probably the daily token quota): {str(e)[:300]}"
                ) from e
            wait = min(max(suggested or 0, 15 * attempt) + 2, MAX_AUTO_WAIT)
            print(f"    [rate limit] {str(e)[:160]!r} -> waiting {wait:.0f}s (retry {attempt}/4)")
            time.sleep(wait)


def healthcheck() -> None:
    """Check every key with a tool-calling request, so problems show up before a long run."""
    import litellm

    load_secrets()
    groq, groq2, openrouter = _models()
    ping = [{"type": "function", "function": {
        "name": "ping", "description": "test",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    }}]
    for group, model, prefix in (
        ("primary", groq, "GROQ_API_KEY"),
        ("secondary", groq2, "GROQ_API_KEY"),
        ("fallback", openrouter, "OPENROUTER_API_KEY"),
    ):
        for i in range(1, 6):
            key = os.getenv(f"{prefix}_{i}")
            if not key:
                continue
            try:
                litellm.completion(model=model, api_key=key, max_tokens=50, tools=ping,
                                   messages=[{"role": "user", "content": "Say hi"}])
                print(f"{group} key {i} ({model}): OK")
            except Exception as e:
                print(f"{group} key {i} ({model}): FAIL {type(e).__name__}: {str(e)[:160]}")
