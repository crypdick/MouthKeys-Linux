"""Session microphone preferences and bounded capture fallback.

The service owns capture lifetime and commits transitions under its state lock.
This module retries adapters outside that lock and shares the lock for source
preferences and the failed-device quarantine.
"""

import subprocess
import time
from _thread import RLock
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from .audio import CaptureTeardownError
from .config import Config
from .desktop import choose_source, source_candidates

RETRY_SECONDS = 8
MAX_ATTEMPTS = 16
MAX_QUARANTINE = 64
QUARANTINE_SECONDS = 60


@runtime_checkable
class Recording(Protocol):
    def stop(self) -> None: ...
    def finished(self) -> bool: ...


@runtime_checkable
class SwitchSource(Protocol):
    def __call__(self, source: str, *, timeout: float) -> object: ...


@dataclass(frozen=True)
class SwitchOutcome:
    source: str | None
    error: Exception | None
    failed_previous: tuple[str, str] | None = None


def selection(data: Mapping[str, object]) -> tuple[str, bool]:
    source, persist = data.get("source"), data.get("persist", False)
    if (
        not isinstance(source, str)
        or not source.strip()
        or len(source) > 500
        or not source.isprintable()
        or type(persist) is not bool
    ):
        raise ValueError("Microphone needs a printable source name and boolean persist")
    return source, persist


def _timeout(deadline: float) -> float:
    return min(5, max(0.01, deadline - time.monotonic()))


def _next_source(config: Config, excluded: Iterable[str]) -> str | None:
    try:
        sources = source_candidates(config, excluded=excluded)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return sources[0] if sources else None


def _previous_failure(transition: object) -> tuple[str, str] | None:
    if isinstance(transition, dict):
        source, error = transition.get("failed_source"), transition.get("error")
        if isinstance(source, str) and source and isinstance(error, str):
            return source, error
    return None


class MicrophoneSelection:
    def __init__(self, lock: RLock) -> None:
        self.lock = lock
        self.session_source: str | None = None
        self.quarantine: dict[str, float] = {}

    def startup_config(self, config: Config) -> Config:
        with self.lock:
            if self.session_source is not None:
                config = replace(config, audio_source=self.session_source)
            now = time.monotonic()
            self.quarantine = {
                source: until for source, until in self.quarantine.items() if until > now
            }
            selected = choose_source(config)
            if selected in self.quarantine:
                candidates = source_candidates(config, excluded=self.quarantine)
                if not candidates:
                    raise ValueError("No available microphone outside failed/Never lists")
                selected = candidates[0]
            return replace(config, audio_source=selected)

    def remember(self, source: str, config: Config, *, persist: bool) -> None:
        with self.lock:
            if persist:
                replace(config, audio_source=source).save()
                self.session_source = None
            else:
                self.session_source = source

    def mark_ready(self, source: str) -> None:
        with self.lock:
            self.quarantine.pop(source, None)

    def mark_failed(self, source: str) -> None:
        with self.lock:
            self.quarantine[source] = time.monotonic() + QUARANTINE_SECONDS
            if len(self.quarantine) > MAX_QUARANTINE:
                self.quarantine = dict(list(self.quarantine.items())[-MAX_QUARANTINE:])

    def open(
        self,
        config: Config,
        start: Callable[[Config, float], Recording],
        cancelled: Callable[[], bool],
    ) -> tuple[Recording, Config]:
        deadline = time.monotonic() + RETRY_SECONDS
        with self.lock:
            tried = set(self.quarantine)
        source = config.audio_source
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            if time.monotonic() >= deadline:
                break
            if cancelled():
                raise RuntimeError("Microphone start cancelled")
            tried.add(source)
            chosen = replace(config, audio_source=source)
            try:
                capture = start(chosen, _timeout(deadline))
            # The service reports the failure; don't log device payloads here.
            except Exception as exc:  # allow: exception-handling
                last_error = exc
                with self.lock:
                    if cancelled():
                        raise
                    self.mark_failed(source)
                alternate = _next_source(config, tried)
                if alternate is None:
                    break
                source = alternate
            else:
                self.mark_ready(source)
                return capture, chosen
        raise RuntimeError(str(last_error or "Microphone startup budget exhausted")) from last_error

    def switch(
        self,
        requested: str,
        fallback: Config,
        switch_source: SwitchSource,
        current: Callable[[], bool],
    ) -> SwitchOutcome | None:
        deadline = time.monotonic() + RETRY_SECONDS
        attempted: set[str] = set()
        source = requested
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            if time.monotonic() >= deadline:
                break
            if not current():
                return None
            attempted.add(source)
            try:
                transition = switch_source(source, timeout=_timeout(deadline))
            # Adapter failures can be device- or subprocess-specific; preserve audio on failure.
            except Exception as exc:  # allow: exception-handling  # noqa: BLE001
                last_error = exc
                with self.lock:
                    if not current():
                        return None
                    failed = exc.source if isinstance(exc, CaptureTeardownError) else source
                    self.mark_failed(failed)
                    excluded = attempted | self.quarantine.keys()
                if isinstance(exc, CaptureTeardownError):
                    break  # Never open another device before the previous one is stopped.
                alternate = _next_source(fallback, excluded)
                if alternate is None:
                    break
                source = alternate
            else:
                return SwitchOutcome(source, last_error, _previous_failure(transition))
        return SwitchOutcome(None, last_error)
