"""Start a supervised FFmpeg microphone process and own its lifeline."""

import os
import signal
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path

from .capture_readiness import bluetooth_capture_ready
from .config import Config
from .processes import python_command


def _has_samples(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            return len(wav.readframes(320)) >= 640
    except (OSError, EOFError, wave.Error):
        return False  # Capture may still be opening the source or forming a WAV header.


class Recorder:
    def __init__(
        self,
        config: Config,
        path: Path,
        *,
        timeout: float = 5,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.last_error = ""
        with ExitStack() as startup:
            self.errors = startup.enter_context(tempfile.TemporaryFile())
            self.process = subprocess.Popen(  # noqa: S603
                [
                    *python_command("mouthkeys.capture"),
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-f",
                    "pulse",
                    "-sample_rate",
                    "16000",
                    "-channels",
                    "1",
                    "-fragment_size",
                    "640",  # 20 ms of mono 16 kHz PCM, before the input opens.
                    "-probesize",
                    "32",  # Pulse already declares PCM; avoid buffering seconds to probe it.
                    "-i",
                    config.audio_source,
                    "-t",
                    str(config.max_seconds),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-flush_packets",
                    "1",
                    str(path),
                ],
                # EOF is a lifeline: even SIGKILL of the daemon closes this pipe.
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.errors,
                start_new_session=True,
            )
            try:
                self._wait_until_ready(path, config.audio_source, timeout, cancelled)
            except BaseException:
                self._wait_for_exit(3, 1)
                raise
            self._files = startup.pop_all()

    def _wait_until_ready(
        self, path: Path, source: str, timeout: float, cancelled: Callable[[], bool] | None
    ) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                raise RuntimeError("Microphone start cancelled")
            if self.finished():
                raise RuntimeError(self.error() or "Microphone capture failed")
            if _has_samples(path) and bluetooth_capture_ready(source):
                return  # Silence is valid only after the physical input is ready.
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Microphone did not produce audio or become ready within {timeout:g} seconds"
                )
            time.sleep(0.02)

    def error(self) -> str:
        if not self.errors.closed:
            self.errors.seek(0)
            self.last_error = self.errors.read(4096).decode(errors="replace").strip()
        return self.last_error

    def finished(self) -> bool:
        return self.process.poll() is not None

    def _wait_for_exit(self, grace: float, reap_timeout: float) -> tuple[int, bool]:
        pipe = self.process.stdin
        if pipe is not None:
            pipe.close()
        try:
            return self.process.wait(timeout=grace), False
        except subprocess.TimeoutExpired:
            forced = False
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
                forced = True
            return self.process.wait(timeout=reap_timeout), forced

    def stop(self) -> None:
        try:
            code, forced = self._wait_for_exit(4, 3)
        finally:
            try:
                error = self.error()
            finally:
                self._files.close()
        if forced:
            raise RuntimeError("Microphone did not stop; temporary audio will be discarded")
        # FFmpeg returns 255 after a clean SIGINT, and 0 at the duration limit.
        if code not in (0, 255):
            raise RuntimeError(error or f"Microphone capture exited with status {code}")
