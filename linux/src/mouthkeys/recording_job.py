"""Recording identity, per-job state, and IPC input validation."""

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .profiles import PromptProfile
from .routing import PromptRoute

MAX_SELECTION_LENGTH = 100000
QUICK_TOGGLE_SECONDS = 0.2


@dataclass
class DictationJob:
    item_id: str
    config: Config
    predecessors: tuple[threading.Event, ...]
    audio_path: Path
    mode: str = "dictation"
    selected_text: str = ""
    label: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    held_activation: bool = False
    cancelled: bool = False
    quiet_discard: bool = False
    duration: float = 0
    phase: str = "recording"
    send_pending: bool = False
    send_dispatched: bool = False
    context: str | None = None
    suppress_send: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_override: str | None = None
    profiles: list[PromptProfile] = field(default_factory=list)
    routes: list[PromptRoute] = field(default_factory=list)
    start_target: object = None
    assistant_session_id: str | None = None
    assistant_memory_session: dict[str, Any] | None = None
    edit_consumed: bool = False


def wait_for_predecessors(job: DictationJob, interrupted: Callable[[], bool]) -> bool:
    for previous in job.predecessors:
        while not previous.wait(0.1):
            if interrupted():
                return False
    return not interrupted()


def recording_input(data: Mapping[str, object]) -> tuple[str, str, str]:
    mode = data.get("mode", "dictation")
    if not isinstance(mode, str) or mode not in (
        "dictation",
        "command",
        "edit",
        "training",
        "pronunciation",
    ):
        raise ValueError("Unknown recording mode")
    selected = data.get("selected_text", "")
    if not isinstance(selected, str) or len(selected) > MAX_SELECTION_LENGTH:
        raise ValueError("Selected text exceeds 100000 characters")
    label = data.get("label", "")
    if mode != "pronunciation":
        return mode, selected, ""
    if (
        not isinstance(label, str)
        or not label.strip()
        or len(label) > 128
        or len(label.split()) > 5
        or not label.isprintable()
    ):
        raise ValueError("Pronunciation label must contain 1–5 words, up to 128 characters")
    return mode, selected, label.strip()
