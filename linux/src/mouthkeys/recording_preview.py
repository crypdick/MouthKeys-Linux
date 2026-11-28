"""Meter live capture and schedule provisional decoding without owning recording state."""

import threading
import time
import wave
from _thread import LockType, RLock
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio import is_subsecond_silence, live_meter
from .config import Config
from .formatting import enabled, parse_send

VOICE_LEVEL = 0.12
RESUMED_SPEECH_SECONDS = 0.35
SEND_PAUSE_SECONDS = 0.5


@dataclass(frozen=True)
class PreviewWork:
    audio_path: Path
    config: Config
    transcriber: object
    predecessors: tuple[threading.Event, ...]
    is_recording: Callable[[], bool]
    cancelled: Callable[[], bool]
    publish: Callable[[str], None]


def reset_owned_stream(transcriber: object, path: Path) -> None:
    """Call under the model lock; do not clear a different recording's native cache."""
    key = getattr(transcriber, "stream_key", None)
    reset = getattr(transcriber, "reset_stream", None)
    if key and key[0] == path and callable(reset):
        reset()


class RecordingPreview:
    """State changes run under the controller lock; decoding only holds the model lock."""

    def __init__(self, lock: RLock, model_lock: LockType) -> None:
        self.lock = lock
        self.model_lock = model_lock
        self.worker: threading.Thread | None = None
        self.reset()

    def reset(self) -> None:
        self._generation = object()
        self.preview_at = time.monotonic()
        self.armed_at: float | None = None
        self.disarmed_text: str | None = None
        self.reset_meter()

    def reset_meter(self) -> None:
        self.meter: dict[str, int | float] = {}
        self.last_voice_at = time.monotonic()

    def suspend(self, text: str) -> None:
        # The same job may be recording again by the time an old decoder returns.
        self._generation = object()
        self.armed_at = None
        self.disarmed_text = text

    def sample(self, path: Path, text: str) -> None:
        try:
            meter = live_meter(path)
        except (OSError, ValueError, EOFError, wave.Error):
            self.meter = {}
            return
        if meter["frame"] == self.meter.get("frame"):
            return
        now = time.monotonic()
        self.meter = {**meter, "at": now}
        if meter["level"] >= VOICE_LEVEL:
            self.last_voice_at = now
            if self.armed_at is not None and now - self.armed_at >= RESUMED_SPEECH_SECONDS:
                self.disarmed_text = text
                self.armed_at = None

    def due(self, config: Config) -> bool:
        return (
            (config.live_preview == "on" or config.send_after_pause == "on")
            and time.monotonic() - self.preview_at >= config.preview_interval_ms / 1000
            and (self.worker is None or not self.worker.is_alive())
        )

    def schedule(self, work: PreviewWork) -> None:
        decode = getattr(work.transcriber, "preview", None)
        if callable(decode):
            self.worker = threading.Thread(
                target=self._decode, args=(work, self._generation, decode), daemon=True
            )
            self.worker.start()

    def should_stop(self, config: Config, *, held: bool) -> bool:
        now = time.monotonic()
        return (
            self.armed_at is not None
            and enabled(config, "send_after_pause")
            and not held
            and enabled(config, "spoken_send")
            and now - self.armed_at >= SEND_PAUSE_SECONDS
            and now - self.last_voice_at >= SEND_PAUSE_SECONDS
        )

    def _accept(self, generation: object, work: PreviewWork) -> bool:
        return generation is self._generation and work.is_recording() and not work.cancelled()

    def _decode(
        self, work: PreviewWork, generation: object, decode: Callable[[Path, Config], str]
    ) -> None:
        try:
            with self.model_lock:
                with self.lock:
                    if not self._accept(generation, work):
                        return
                    if any(not done.is_set() for done in work.predecessors):
                        return  # Preserve earlier native stream caches until ordered finalization.
                if is_subsecond_silence(work.audio_path):
                    return
                text = decode(work.audio_path, work.config)
                if work.cancelled():
                    reset_owned_stream(work.transcriber, work.audio_path)
            with self.lock:
                if self._accept(generation, work):
                    work.publish(text)
                    self._arm(text, work.config)
        # Preview is provisional; final inference reports actionable errors without transcript logs.
        except Exception:  # allow: exception-handling  # noqa: BLE001, S110
            pass
        finally:
            with self.lock:
                if generation is self._generation:
                    self.preview_at = time.monotonic()

    def _arm(self, text: str, config: Config) -> None:
        plan = parse_send(
            text,
            config.spoken_send_phrase,
            active=enabled(config, "spoken_send"),
            literal_prefix=config.literal_prefix,
        )
        if not plan.should_send or text == self.disarmed_text:
            self.armed_at = None
        elif self.armed_at is None:
            self.armed_at = time.monotonic()
