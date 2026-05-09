import re
from html import unescape


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_text(content: str, max_len: int = 25000) -> str:
    if not content:
        return ""
    cleaned = unescape(content)
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned
