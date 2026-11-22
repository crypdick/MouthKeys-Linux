"""Render app-specific literal slash commands and mentions."""

import re

from .format_whitespace import horizontal, rstrip_horizontal

_SLASH_REJECTED = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "back",
    "backslash",
    "be",
    "been",
    "being",
    "bin",
    "by",
    "comma",
    "desktop",
    "documents",
    "dot",
    "downloads",
    "etc",
    "for",
    "forward",
    "from",
    "home",
    "in",
    "is",
    "library",
    "local",
    "mark",
    "of",
    "on",
    "or",
    "period",
    "private",
    "question",
    "quote",
    "quotes",
    "semicolon",
    "slash",
    "slashes",
    "source",
    "sources",
    "src",
    "the",
    "tmp",
    "to",
    "user",
    "users",
    "usr",
    "var",
    "volumes",
    "was",
    "were",
    "with",
    "without",
}
_SLASH_LEAD_INS = {
    "call",
    "choose",
    "do",
    "enter",
    "execute",
    "open",
    "pick",
    "press",
    "run",
    "say",
    "select",
    "send",
    "start",
    "try",
    "type",
    "use",
    "write",
}
_MENTION_REJECTED = {
    "a",
    "an",
    "airport",
    "breakfast",
    "brunch",
    "class",
    "dinner",
    "home",
    "hotel",
    "house",
    "lunch",
    "meeting",
    "night",
    "noon",
    "office",
    "place",
    "restaurant",
    "school",
    "shop",
    "store",
    "the",
    "today",
    "tomorrow",
    "work",
    "yesterday",
}
_MENTION_LEAD_INS = {
    "add",
    "ask",
    "cc",
    "dm",
    "hello",
    "hey",
    "hi",
    "invite",
    "message",
    "notify",
    "ping",
    "send",
    "tag",
    "tell",
}


def _lead_in(text: str, start: int, allowed: set[str]) -> bool:
    prefix = text[:start].strip()
    if not prefix or prefix[-1] in ".!?:;([{":
        return True
    words = re.findall(r"[a-zA-Z0-9_-]+", prefix)
    return not words or words[-1].casefold() in allowed


def apply_literal_formatting(
    text: str,
    *,
    app_name: str | None = None,
    app_id: str | None = None,
    window_title: str | None = None,
) -> str:
    """Upstream's opt-in slash commands and explicit/contextual mentions."""
    haystack = " ".join(value or "" for value in (app_name, app_id, window_title)).casefold()
    relaxed = any(app in haystack for app in ("slack", "discord", "teams"))
    command_name = r"([A-Za-z][A-Za-z0-9_-]{1,39})(?![A-Za-z0-9_-])"
    for pattern, spoken in (
        (r"(?<!\w)/\s+" + command_name, False),
        (r"(?<!\w)(?:forward\s+slash|slash)\s+" + command_name, True),
    ):

        def slash(match: re.Match[str], *, spoken: bool = spoken, source: str = text) -> str:
            command = match[1].lower()
            if command in _SLASH_REJECTED or not re.search(r"[a-z0-9_-]$", command):
                return match[0]
            if spoken and not _lead_in(source, match.start(), _SLASH_LEAD_INS):
                return match[0]
            return "/" + command

        text = re.sub(pattern, slash, text, flags=re.I)
    name = r"([A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,2})(?![A-Za-z0-9_.-])"
    for pattern, is_relaxed in (
        (r"(?<![\w@])(?i:at\s+(?:sign|the\s+rate)|tag|mention)\s+" + name, False),
        (r"(?<![\w@])[Aa]t\s+" + name, True),
    ):
        if is_relaxed and not relaxed:
            continue

        def mention(
            match: re.Match[str], *, is_relaxed: bool = is_relaxed, source: str = text
        ) -> str:
            words = match[1].split(" ")
            if any(word.lower() in _MENTION_REJECTED for word in words):
                return match[0]
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", word) for word in words):
                return match[0]
            if is_relaxed and (
                any(not word[:1].isupper() for word in words)
                or not _lead_in(source, match.start(), _MENTION_LEAD_INS)
            ):
                return match[0]
            if source[match.end() : match.end() + 1] in {"'", "’"}:
                return match[0]
            return "@" + match[1].strip()

        text = re.sub(pattern, mention, text)
    return text


def literal_autocomplete_spacing(
    text: str,
    *,
    app_name: str | None = None,
    app_id: str | None = None,
    window_title: str | None = None,
) -> str:
    haystack = " ".join(value or "" for value in (app_name, app_id, window_title)).casefold()
    if not text or not horizontal(text[-1]):
        return text
    stripped = rstrip_horizontal(text)
    if any(app in haystack for app in ("codex", "chatgpt", "claude", "cursor", "windsurf")) and (
        re.fullmatch(r"/[A-Za-z][A-Za-z0-9_-]{1,39}", stripped)
    ):
        return stripped
    if any(app in haystack for app in ("slack", "discord", "teams")) and re.search(
        r"(?<![\w@])@[A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,2}$", stripped
    ):
        return stripped
    return text
