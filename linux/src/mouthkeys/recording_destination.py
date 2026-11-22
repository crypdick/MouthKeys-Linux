"""Read optional desktop evidence without changing focus."""

import threading
import time
from dataclasses import dataclass, field

from . import accessibility
from .config import Config

FOCUS_TIMEOUT = 4


@dataclass
class FocusGate:
    ready: threading.Event = field(default_factory=threading.Event)
    valid: bool = True
    deadline: float | None = None

    @classmethod
    def automatic(cls) -> "FocusGate":
        return cls(deadline=time.monotonic() + FOCUS_TIMEOUT)

    def wait(self) -> None:
        # A stop acknowledgment must not precede its focus snapshot.
        if not self.ready.wait(timeout=FOCUS_TIMEOUT):
            self.valid = False

    def resolved(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.valid = False
        self.ready.set()


@dataclass(frozen=True)
class StopDestination:
    target: object
    automatic: bool
    context: str | None
    metadata_target: object
    collect_metadata: bool

    def metadata(self, delivery: object) -> dict[str, object] | None:
        if not self.collect_metadata:
            return None
        lookup = getattr(delivery, "target_metadata", None)
        if not callable(lookup):
            return None
        try:
            values = lookup(self.metadata_target)
        # Destination metadata is optional; an unavailable desktop must not block transcription.
        except Exception:  # allow: exception-handling  # noqa: BLE001
            return None
        return values if isinstance(values, dict) else None


def _focused_target(delivery: object, config: Config, *, window_fallback: bool = True) -> object:
    try:
        snapshot = getattr(delivery, "capture_target", None)
        if callable(snapshot):
            return snapshot(config)
        if not window_fallback:
            return None
        focus = getattr(delivery, "focus", None)
        return focus() if callable(focus) else None
    # Unknown focus disables automatic delivery without blocking capture.
    except Exception:  # allow: exception-handling  # noqa: BLE001
        return None


def snapshot_for_start(delivery: object, config: Config, mode: str) -> object:
    if config.target_policy == "start" or config.return_to_start == "on" or mode == "edit":
        return _focused_target(delivery, config, window_fallback=False)
    return None


def _formatting_context(config: Config) -> str | None:
    if config.smart_capitalization != "on" and config.space_between_dictations != "on":
        return None
    try:
        return accessibility.read_context()
    # Formatting context is optional and may disappear with its window.
    except Exception:  # allow: exception-handling  # noqa: BLE001
        return None


def snapshot_for_stop(
    delivery: object,
    config: Config,
    target: object,
    *,
    start_target: object,
    mode: str,
) -> StopDestination:
    automatic = target == "capture-focus"
    if automatic:
        if config.target_policy == "start" or config.return_to_start == "on":
            target = start_target
        else:
            target = _focused_target(delivery, config)
    needs_metadata = (
        config.save_history == "on" or config.ai_enabled == "on" or mode in ("edit", "command")
    ) and callable(getattr(delivery, "target_metadata", None))
    metadata_target = (
        _focused_target(delivery, config) if needs_metadata and target is None else target
    )
    return StopDestination(
        target, automatic, _formatting_context(config), metadata_target, needs_metadata
    )
