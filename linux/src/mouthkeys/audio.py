"""Record and read live PCM without invalidating committed preview frames."""

import math
import os
import tempfile
import threading
import wave
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .capture_recorder import Recorder
from .config import Config, private_dir, runtime_dir


@runtime_checkable
class Recording(Protocol):
    def stop(self) -> None: ...

    def finished(self) -> bool: ...


class CaptureTeardownError(RuntimeError):
    """A source transition must not open another capture after unsafe teardown."""

    def __init__(self, source: str, error: Exception) -> None:
        self.source = source
        super().__init__(f"Cannot finish microphone {source}: {error}")


class CaptureSession:
    """Append capture segments to one stable WAV across explicit source switches.

    PCM prefix never changes: concurrent preview and native streaming cursors remain
    valid. Switching has a real microphone restart gap; no silence is invented.
    One private runtime segment is bounded by the recording limit and removed on
    stop or the next daemon launch. Canonical audio is temporary too.
    """

    def __init__(
        self,
        config: Config,
        path: str | Path,
        *,
        recorder: Callable[..., Recording] = Recorder,
        cancel_event: threading.Event | None = None,
        timeout: float = 5,
    ) -> None:
        self.path = Path(path)
        self.factory = recorder
        self.cancel_event = cancel_event
        self.stopping = threading.Event()
        self.lock = threading.RLock()
        self.capture: Recording | None = None
        self.segment: Path | None = None
        self.frames = 0
        self.failure: Exception | None = None
        self._files = ExitStack()
        self.closed = False
        self.config = config
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            self.output = self._files.enter_context(os.fdopen(descriptor, "wb"))
            os.fchmod(descriptor, 0o600)
            self.writer = self._files.enter_context(wave.open(self.output, "wb"))
            self.writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            self.writer.writeframes(b"")
            self.output.flush()
            self._open(config, timeout)
            self.thread = threading.Thread(target=self._relay, daemon=True)
            self.thread.start()
        except BaseException:
            self.stopping.set()
            self._close()
            raise

    def _cancelled(self) -> bool:
        return self.stopping.is_set() or bool(self.cancel_event and self.cancel_event.is_set())

    def _open(self, config: Config, timeout: float) -> None:
        descriptor, name = tempfile.mkstemp(
            prefix=f"{self.path.stem}-capture-",
            suffix=".wav",
            dir=private_dir(runtime_dir() / "private-recordings"),
        )
        os.close(descriptor)
        segment = Path(name)
        self.segment, self.frames = segment, 0
        try:
            self.capture = self.factory(config, segment, timeout=timeout, cancelled=self._cancelled)
            self.config = config
            self._drain()
        except BaseException:
            capture, self.capture = self.capture, None
            try:
                if capture:
                    capture.stop()
            finally:
                segment.unlink(missing_ok=True)
                self.segment = None
            raise

    def _drain(self) -> None:
        if self.segment is None:
            return
        with _capture_wav(self.segment) as (wav, available):
            wav.setpos(self.frames)
            while self.frames < available:
                raw = wav.readframes(min(16000, available - self.frames))
                if not raw:
                    break
                self.writer.writeframes(raw)
                self.frames += len(raw) // 2
            self.output.flush()

    def _end_segment(self, *, tolerate_ended_failure: bool = False) -> Exception | None:
        capture = self.capture
        stop_error = None
        if capture:
            try:
                capture.stop()
            # Carry device failure back to the capture owner for teardown and recovery.
            except Exception as exc:  # allow: exception-handling
                try:
                    ended = capture.finished()
                # If capture state is unreadable, assume the device has not stopped.
                except Exception:  # allow: exception-handling  # noqa: BLE001
                    ended = False
                if not ended:
                    # Retain ownership for a subsequent stop; never overlap a
                    # replacement capture with one whose shutdown is unproven.
                    raise CaptureTeardownError(self.config.audio_source, exc) from exc
                stop_error = exc
        self.capture = None
        try:
            self._drain()
        finally:
            if self.segment:
                self.segment.unlink(missing_ok=True)
                self.segment = None
        if stop_error and not tolerate_ended_failure:
            raise stop_error
        return stop_error

    def _relay(self) -> None:
        while not self.stopping.wait(0.02):
            try:
                with self.lock:
                    self._drain()
            # Carry device failure back to the capture owner for teardown and recovery.
            except Exception as exc:  # allow: exception-handling  # noqa: BLE001
                self.failure = exc
                self.stopping.set()

    def switch_source(self, source: str, *, timeout: float = 5) -> dict[str, str] | None:
        with self.lock:
            if self._cancelled():
                raise RuntimeError("Recording is stopping")
            try:
                stop_error = self._end_segment(tolerate_ended_failure=True)
            except CaptureTeardownError:
                raise
            except Exception as exc:
                raise CaptureTeardownError(self.config.audio_source, exc) from exc
            if self._cancelled():
                raise RuntimeError("Recording is stopping")
            previous_source = self.config.audio_source
            self._open(replace(self.config, audio_source=source), timeout)
            if stop_error:
                return {"failed_source": previous_source, "error": str(stop_error)}
            return None

    def finished(self) -> bool:
        return self.failure is not None or (self.capture is not None and self.capture.finished())

    def _close(self) -> None:
        if self.closed:
            return
        try:
            self._end_segment()
        finally:
            if self.capture is None:
                try:
                    self._files.close()
                finally:
                    self.closed = True

    def stop(self) -> None:
        self.stopping.set()
        try:
            with self.lock:
                self._close()
        finally:
            self.thread.join(timeout=1)
            if self.thread.is_alive():
                raise RuntimeError("Capture relay did not stop")
        if self.failure:
            raise RuntimeError(f"Capture relay failed: {self.failure}") from self.failure


def check_audio(path: Path) -> None:
    with wave.open(str(path), "rb") as wav:
        if wav.getnframes() < 1600:
            raise ValueError("Recording is too short; speak for at least a moment")


def is_short_silence(path: Path) -> bool:
    """Only skip up to four seconds of near-zero PCM, never ordinary quiet speech."""
    with wave.open(str(path), "rb") as source:
        if source.getnframes() > source.getframerate() * 4 or source.getsampwidth() != 2:
            return False
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return bool(len(samples)) and int(np.max(np.abs(samples.astype(np.int32)))) <= 1


@contextmanager
def _capture_wav(path: Path) -> Iterator[tuple[wave.Wave_read, int]]:
    with path.open("rb") as source, wave.open(source, "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected mono 16 kHz PCM")
        # Live headers can overstate data length; finalized files can have trailing chunks.
        available = max(0, (os.fstat(source.fileno()).st_size - source.tell()) // 2)
        yield wav, min(wav.getnframes(), available)


def _normalized_pcm(raw: bytes) -> NDArray[np.float32]:
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def _read_live_audio(path: Path, seconds: float) -> tuple[NDArray[np.float32], int]:
    """Read bounded samples and their frame cursor before the WAV is finalized."""
    count = int(seconds * 16000)
    with _capture_wav(path) as (wav, frames):
        wav.setpos(max(0, frames - count))
        return _normalized_pcm(wav.readframes(count)), frames


def read_live_audio(path: Path, seconds: float = 20) -> NDArray[np.float32]:
    return _read_live_audio(path, seconds)[0]


def read_audio_from(
    path: Path, start_frame: int = 0, seconds: float | None = None
) -> NDArray[np.float32]:
    """Read forward from a committed PCM frame, including an unfinished WAV."""
    with _capture_wav(path) as (wav, frames):
        if start_frame > frames:
            raise ValueError("Recording shrank after transcription started")
        wav.setpos(start_frame)
        count = frames - start_frame
        if seconds is not None:
            count = min(count, int(seconds * 16000))
        return _normalized_pcm(wav.readframes(count))


def live_meter(path: Path) -> dict[str, int | float]:
    samples, frames = _read_live_audio(path, 0.02)
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    level = max(0.0, min(1.0, (20 * math.log10(max(rms, 1e-9)) + 60) / 55))
    return {"frame": frames, "level": level}
