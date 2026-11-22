"""Validate captured audio and serialize final decoding without publishing job state."""

import copy
from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .audio import check_audio, is_short_silence
from .config import Config
from .recording_job import DictationJob
from .recording_preview import reset_owned_stream


@runtime_checkable
class _SpeechModel(Protocol):
    def transcribe(self, path: Path, config: Config) -> str: ...


@dataclass(frozen=True)
class Recognition:
    text: str = ""
    alignment: object = None
    skipped_silence: bool = False


def _alignment_snapshot(model: object) -> object:
    try:
        return copy.deepcopy(getattr(model, "last_alignment", None))
    # Word timings are optional; an adapter's unusable metadata must not lose recognized text.
    except Exception:  # allow: exception-handling  # noqa: BLE001
        return {"warning": "Word timing metadata unavailable"}


def recognize_audio(
    job: DictationJob,
    model: _SpeechModel,
    lock: LockType,
    stopped: Callable[[], bool],
) -> Recognition:
    """Return a result for the controller to commit only if the job is still live."""
    config, path = job.config, job.audio_path
    check_audio(path)
    if stopped():
        return Recognition()
    if config.skip_silent_recordings == "on" and is_short_silence(path):
        return Recognition(skipped_silence=True)
    with lock:
        if stopped():
            return Recognition()
        text = model.transcribe(path, config)
        alignment = _alignment_snapshot(model) if config.pronunciation_matching == "on" else None
    return Recognition(text, alignment)


def reset_cancelled_stream(job: DictationJob, model: object, lock: LockType) -> None:
    key = getattr(model, "stream_key", None)
    if job.cancelled and key and key[0] == job.audio_path:
        with lock:
            reset_owned_stream(model, job.audio_path)
