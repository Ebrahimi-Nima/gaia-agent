from .audio import audio_tools
from .base import make_tool
from .files import FILE_TOOLS
from .vision import vision_tools
from .web import WEB_TOOLS


def build_tools() -> list[dict]:
    """All tools that are usable in the current environment."""
    return WEB_TOOLS + FILE_TOOLS + audio_tools() + vision_tools()


__all__ = ["build_tools", "make_tool"]
