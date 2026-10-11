"""Web tools: search, page / PDF reader, Wikipedia reader (with historical revisions), YouTube transcripts."""
from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from markdownify import markdownify as to_markdown

from .base import make_tool

WIKI_API = "https://en.wikipedia.org/w/api.php"

DIRECT_CHARS = 6000       # pages shorter than this are returned as-is
MAX_OUTPUT_CHARS = 6000   # cap on what the agent receives from one page
SECTION_CHARS = 4000      # max size of one section

BLOCKED_URL_PARTS = (
    "crossword", "wordplays.com",
    "huggingface.co/datasets/gaia-benchmark", "huggingface.co/spaces", "gaia-benchmark",
)


def _headers() -> dict:
    return {"User-Agent": f"gaia-agent/0.2 (educational project; contact: {os.getenv('EMAIL', '')})"}


def _is_blocked(url: str) -> bool:
    return any(part in url.lower() for part in BLOCKED_URL_PARTS)


def _safe_url(url: str) -> bool:
    """Only public http(s) hosts (blocks localhost and cloud metadata servers). Redirects are not re-checked."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        for info in socket.getaddrinfo(parsed.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except socket.gaierror:
        return True  # let requests report the DNS failure itself
    return True


def _html_to_markdown(html: str, keep_links: bool = True) -> str:
    soup = BeautifulSoup(html, "lxml")
    for selector in [
        "script", "style", "noscript", "nav", "footer", "aside", "form",
        "sup.reference", "span.mw-editsection", "table.navbox", "div.navbox",
        "ol.references", "div.reflist", "div.printfooter",
    ]:
        for el in soup.select(selector):
            el.decompose()
    strip = ["img"] if keep_links else ["img", "a"]
    try:
        text = to_markdown(str(soup.body or soup), heading_style="ATX", strip=strip)
        text = re.sub(r"\\\(\s*\\(?:new|renew)command.*?\\\)", "", text)
    except RecursionError:
        text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pdf_to_markdown(data: bytes) -> str:
    import pymupdf
    import pymupdf4llm

    return pymupdf4llm.to_markdown(pymupdf.open(stream=data, filetype="pdf"))


def _chunk_markdown(text: str, size: int) -> list[str]:
    """Split at line boundaries; repeat the last heading at the top of each new chunk."""
    chunks, current, current_len, heading = [], [], 0, ""
    for line in text.split("\n"):
        if line.startswith("#"):
            heading = line
        if current and current_len + len(line) + 1 > size:
            chunks.append("\n".join(current))
            current = [heading] if heading and not line.startswith("#") else []
            current_len = sum(len(x) + 1 for x in current)
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "between", "what", "which",
    "how", "many", "are", "was", "were", "have", "has", "who", "did", "does", "into",
    "about", "list", "all", "any", "there", "their", "them", "its",
}


def _split_units(text: str, max_chars: int = SECTION_CHARS) -> list[dict]:
    """Split markdown into sections at headings; oversized sections are split into parts."""
    sections, title, buf = [], "(intro)", []
    for line in text.split("\n"):
        if re.match(r"#{1,6} ", line):
            if buf:
                sections.append((title, "\n".join(buf)))
            title, buf = line.lstrip("#").strip(), [line]
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf)))

    units = []
    for title, body in sections:
        if len(body.strip()) < 60:  # heading with no real content
            continue
        parts = _chunk_markdown(body, max_chars)
        for k, part in enumerate(parts, 1):
            name = title if len(parts) == 1 else f"{title} (part {k}/{len(parts)})"
            units.append({"title": name, "text": part})
    return units


def _keywords(question: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    words = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _score(unit: dict, keys: set[str]) -> int:
    title, body = unit["title"].lower(), unit["text"].lower()
    return sum(3 for k in keys if k in title) + sum(1 for k in keys if k in body)


def _extract_relevant(text: str, question: str) -> str:
    if len(text) <= DIRECT_CHARS:
        return text

    units = _split_units(text)
    keys = _keywords(question)
    scores = [_score(u, keys) for u in units]
    ranked = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)

    chosen, used = [], 0
    for i in ranked[:4]:
        if scores[i] <= 0:
            break
        size = len(units[i]["text"])
        if used + size <= MAX_OUTPUT_CHARS:
            chosen.append(i)
            used += size

    if not chosen:
        titles = " | ".join(u["title"] for u in units)[:1500]
        return (f"No section matched the question. Start of the page:\n{text[:2000]}\n\n"
                f"[Sections on this page: {titles}]")

    body = "\n\n---\n\n".join(units[i]["text"] for i in sorted(chosen))
    others = " | ".join(u["title"] for j, u in enumerate(units) if j not in chosen)[:800]
    return f"{body}\n\n[Other sections on this page: {others}]"


def _fetch_wikipedia_html(title: str, date: str = "") -> str:
    """Return article HTML, optionally the latest revision on or before `date` (YYYY-MM-DD)."""
    params = {"action": "parse", "prop": "text", "format": "json", "formatversion": 2}
    if date:
        rev = requests.get(
            WIKI_API, headers=_headers(), timeout=20,
            params={
                "action": "query", "prop": "revisions", "titles": title,
                "rvlimit": 1, "rvdir": "older", "rvstart": f"{date}T23:59:59Z",
                "rvprop": "ids", "redirects": 1, "format": "json", "formatversion": 2,
            },
        )
        rev.raise_for_status()
        page = rev.json()["query"]["pages"][0]
        if page.get("missing") or not page.get("revisions"):
            raise ValueError(f"No revision of '{title}' found on or before {date}.")
        params["oldid"] = page["revisions"][0]["revid"]
    else:
        params["page"] = title
        params["redirects"] = 1

    resp = requests.get(WIKI_API, headers=_headers(), timeout=30, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"Wikipedia error: {data['error'].get('info', data['error'])}")
    return data["parse"]["text"]


SEARCH_ATTEMPTS = [{}, {"backend": "bing"}, {"backend": "brave"}]


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs and snippets (retries with other backends)."""
    max_results = min(int(max_results), 6)
    last_error = "No results found."
    for extra in SEARCH_ATTEMPTS:
        try:
            results = DDGS().text(query, max_results=max_results + 4, **extra)
        except Exception as e:
            last_error = str(e) or type(e).__name__
            continue
        results = [
            r for r in results
            if "aclick" not in r.get("href", "") and not _is_blocked(r.get("href", ""))
        ][:max_results]
        if results:
            return "\n\n".join(
                f"[{i}] {r.get('title', '')}\n    URL: {r.get('href', '')}\n    {r.get('body', '')[:300]}"
                for i, r in enumerate(results, 1)
            )
    return f"Search failed: {last_error}. Try rephrasing with fewer quotes or different keywords."


def read_webpage(url: str, question: str) -> str:
    """Fetch a web page or PDF and return only the parts relevant to `question`."""
    if "youtube.com" in url or "youtu.be" in url:
        return "Use the youtube_transcript tool for YouTube links; read_webpage cannot read them. Do not retry this URL here."
    if _is_blocked(url):
        return "This source is blocked because it mirrors benchmark questions or answers. Use a primary source."
    if not _safe_url(url):
        return "This URL is not allowed (only public http/https hosts)."
    try:
        resp = requests.get(url, timeout=30, headers=_headers())
        resp.raise_for_status()
    except Exception as e:
        return f"Failed to fetch page: {e}"

    is_pdf = "pdf" in resp.headers.get("Content-Type", "").lower() or url.lower().split("?")[0].endswith(".pdf")
    try:
        text = _pdf_to_markdown(resp.content) if is_pdf else _html_to_markdown(resp.text, keep_links=True)
    except Exception as e:
        return f"Failed to parse content: {type(e).__name__}: {e}"
    return _extract_relevant(text, question)


def read_wikipedia(title: str, question: str, date: str = "") -> str:
    """Read an English Wikipedia article (optionally as of `date`) and return the parts relevant to `question`."""
    try:
        html = _fetch_wikipedia_html(title, date)
    except Exception as e:
        return f"Failed to fetch Wikipedia page: {e}"
    return _extract_relevant(_html_to_markdown(html, keep_links=False), question)


def youtube_transcript(url: str) -> str:
    """Return the spoken-word transcript of a YouTube video (no visual information)."""
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    if not match:
        return "Could not find a YouTube video id in that URL."
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        fetched = YouTubeTranscriptApi().fetch(match.group(1), languages=["en", "en-US", "en-GB"])
        text = " ".join(s.text for s in fetched)
    except Exception as e:
        return (f"No transcript available: {type(e).__name__}: {str(e)[:200]}. "
                "Do not retry; the answer may need visual information that no tool here provides.")
    return text if len(text) <= 10000 else text[:10000] + f"\n...[truncated, {len(text)} chars total]"


WEB_TOOLS = [
    make_tool(
        "web_search",
        "Search the web. Returns titles, URLs and short snippets. Use read_webpage afterwards to read a promising result.",
        {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "Number of results (default 5)."},
        },
        ["query"],
        web_search,
    ),
    make_tool(
        "read_webpage",
        "Read a web page or PDF by URL. Returns only the passages relevant to `question` (verbatim, tables included). "
        "If the result looks incomplete, call again with a differently worded question.",
        {
            "url": {"type": "string", "description": "Full URL starting with https://"},
            "question": {"type": "string", "description": "Specific description of the information you need from this page."},
        },
        ["url", "question"],
        read_webpage,
    ),
    make_tool(
        "read_wikipedia",
        "Read an English Wikipedia article by exact title. Returns only the passages relevant to `question`. "
        "Pass `date` (YYYY-MM-DD) to read the article as it was on that date, e.g. when the question mentions a specific year's version.",
        {
            "title": {"type": "string", "description": "Exact article title, e.g. 'Mercedes Sosa'."},
            "question": {"type": "string", "description": "Specific description of the information you need, e.g. 'studio albums with release years'."},
            "date": {"type": "string", "description": "Optional YYYY-MM-DD."},
        },
        ["title", "question"],
        read_wikipedia,
    ),
    make_tool(
        "youtube_transcript",
        "Get the spoken-word transcript of a YouTube video. It contains only speech: it cannot be used to count or "
        "identify things shown on screen.",
        {"url": {"type": "string", "description": "YouTube video URL."}},
        ["url"],
        youtube_transcript,
    ),
]
