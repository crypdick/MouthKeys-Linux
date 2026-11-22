"""Optional AI cleanup of an already transcribed, locally formatted recording."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from .config import Config
from .postprocess import process


@dataclass(frozen=True)
class CleanupResult:
    text: str
    applied: bool = False
    warning: str = ""


def cleanup_transcript(
    text: str, config: Config, cancel_event: Event, publish: Callable[[str], None]
) -> CleanupResult:
    if config.ai_enabled != "on" or not text or cancel_event.is_set():
        return CleanupResult(text)

    pieces: list[str] = []

    def delta(piece: str) -> None:
        if not cancel_event.is_set():
            pieces.append(piece)
            publish("".join(pieces))

    try:
        cleaned = process(
            text,
            config,
            on_delta=delta if config.ai_streaming == "on" else None,
            cancel_event=cancel_event,
        )
    # Keep local text when optional cleanup fails; do not write provider diagnostics to logs.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        if cancel_event.is_set():
            return CleanupResult(text)
        return CleanupResult(
            text, warning=f"AI cleanup failed ({exc}); using original transcription"
        )
    return CleanupResult(cleaned, applied=True)
