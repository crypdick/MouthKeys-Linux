import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path

from .config import Config, private_dir, runtime_dir
from .processes import python_command


def bluetooth_capture_ready(source):
    """PipeWire's Bluetooth loopback can run before its real microphone exists."""
    if source == "default":
        try:
            source = subprocess.run(
                ["pactl", "get-default-source"],
                capture_output=True,
                text=True,
                check=True,
                timeout=0.5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return True  # No source discovery: retain normal PulseAudio readiness.
    if not source.startswith("bluez_input."):
        return True
    try:
        result = subprocess.run(
            ["pactl", "-f", "json", "list", "sources"],
            capture_output=True,
            text=True,
            check=True,
            timeout=0.5,
        )
        sources = json.loads(result.stdout)
        selected_source = next((s for s in sources if s.get("name") == source), None)
        if selected_source is None:
            return False
        if selected_source.get("properties", {}).get("bluez5.loopback") not in (True, "true"):
            return True  # Native PulseAudio capture does not need a PipeWire graph.
        result = subprocess.run(
            ["pw-dump", "Node"],
            capture_output=True,
            text=True,
            check=True,
            timeout=0.5,
        )
        nodes = [node.get("info", {}) for node in json.loads(result.stdout)]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("Cannot verify Bluetooth microphone readiness with pw-dump") from exc
    selected = next((n for n in nodes if n.get("props", {}).get("node.name") == source), None)
    if selected is None:
        return False
    props = selected.get("props", {})
    if props.get("bluez5.loopback") not in (True, "true"):
        return True
    device = props.get("device.id")
    return device is not None and any(
        str(n.get("props", {}).get("device.id")) == str(device)
        and n.get("props", {}).get("bluez5.loopback-target") in (True, "true")
        and n.get("state") == "running"
        for n in nodes
    )


class Recorder:
    def __init__(self, config: Config, path: Path, *, timeout=5, cancelled=None):
        self.errors = tempfile.TemporaryFile()
        self.last_error = ""
        self.process = None
        try:
            self.process = subprocess.Popen(
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
            deadline = time.monotonic() + timeout
            while True:
                if cancelled and cancelled():
                    raise RuntimeError("Microphone start cancelled")
                if self.process.poll() is not None:
                    raise RuntimeError(self.error() or "Microphone capture failed")
                try:
                    with wave.open(str(path), "rb") as wav:
                        samples_ready = len(wav.readframes(320)) >= 640
                    if samples_ready and bluetooth_capture_ready(config.audio_source):
                        break  # Silence is valid only after the physical input is ready.
                except (OSError, EOFError, wave.Error):
                    pass  # FFmpeg may still be opening the source or writing the WAV header.
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Microphone did not produce audio or become ready "
                        f"within {timeout:g} seconds"
                    )
                time.sleep(0.02)
        except BaseException:
            if self.process is not None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=1)
            self.errors.close()
            raise

    def error(self):
        if self.errors.closed:
            return self.last_error
        self.errors.seek(0)
        self.last_error = self.errors.read(4096).decode(errors="replace").strip()
        return self.last_error

    def finished(self):
        return self.process.poll() is not None

    def stop(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=3)
            raise RuntimeError("Microphone did not stop; saved audio may need recovery") from None
        finally:
            error = self.error()
            self.errors.close()
        # FFmpeg returns 255 after a clean SIGINT, and 0 at the duration limit.
        if code not in (0, 255):
            raise RuntimeError(error or f"Microphone capture exited with status {code}")


class CaptureTeardownError(RuntimeError):
    """A source transition must not open another capture after unsafe teardown."""

    def __init__(self, source, error):
        self.source = source
        super().__init__(f"Cannot finish microphone {source}: {error}")


class CaptureSession:
    """Append capture segments to one stable WAV across explicit source switches.

    PCM prefix never changes: concurrent preview and native streaming cursors remain
    valid. Switching has a real microphone restart gap; no silence is invented.
    One private runtime segment is bounded by the recording limit and removed on
    stop or the next daemon launch; only canonical audio is retained for recovery.
    """

    def __init__(self, config, path, *, recorder=Recorder, cancel_event=None, timeout=5):
        self.path = Path(path)
        self.factory = recorder
        self.cancel_event = cancel_event
        self.stopping = threading.Event()
        self.lock = threading.RLock()
        self.capture = None
        self.segment = None
        self.frames = 0
        self.failure = None
        self.writer = None
        self.output = None
        self.closed = False
        self.config = config
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(descriptor, 0o600)
            self.output = os.fdopen(descriptor, "wb")
            self.writer = wave.open(self.output, "wb")
            self.writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            self.writer.writeframes(b"")
            self.output.flush()
            self._open(config, timeout)
        except BaseException:
            self._close()
            raise
        self.thread = threading.Thread(target=self._relay, daemon=True)
        self.thread.start()

    def _cancelled(self):
        return self.stopping.is_set() or bool(self.cancel_event and self.cancel_event.is_set())

    def _open(self, config, timeout):
        descriptor, name = tempfile.mkstemp(
            prefix=f"{self.path.stem}-capture-",
            suffix=".wav",
            dir=private_dir(runtime_dir() / "private-recordings"),
        )
        os.close(descriptor)
        self.segment, self.frames = Path(name), 0
        try:
            self.capture = self.factory(
                config, self.segment, timeout=timeout, cancelled=self._cancelled
            )
            self.config = config
            self._drain()
        except BaseException:
            capture, self.capture = self.capture, None
            try:
                if capture:
                    capture.stop()
            finally:
                self.segment.unlink(missing_ok=True)
                self.segment = None
            raise

    def _drain(self):
        if self.segment is None:
            return
        with self.segment.open("rb") as stream, wave.open(stream, "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected mono 16 kHz PCM capture")
            # FFmpeg's live WAV header has an unfinalized data length.
            available = max(0, (self.segment.stat().st_size - stream.tell()) // 2)
            wav.setpos(self.frames)
            while self.frames < available:
                raw = wav.readframes(min(16000, available - self.frames))
                if not raw:
                    break
                self.writer.writeframes(raw)
                self.frames += len(raw) // 2
            self.output.flush()

    def _end_segment(self, *, tolerate_ended_failure=False):
        capture = self.capture
        stop_error = None
        if capture:
            try:
                capture.stop()
            except Exception as exc:
                try:
                    ended = capture.finished()
                except Exception:
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

    def _relay(self):
        while not self.stopping.wait(0.02):
            try:
                with self.lock:
                    self._drain()
            except Exception as exc:
                self.failure = exc
                self.stopping.set()

    def switch_source(self, source, *, timeout=5):
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

    def finished(self):
        return self.failure is not None or (self.capture is not None and self.capture.finished())

    def _close(self):
        if self.closed:
            return
        try:
            self._end_segment()
        finally:
            if self.capture is None:
                try:
                    if self.writer:
                        self.writer.close()
                finally:
                    self.writer = None
                    if self.output:
                        self.output.close()
                        self.output = None
                    self.closed = True

    def stop(self):
        self.stopping.set()
        try:
            with self.lock:
                self._close()
        finally:
            if hasattr(self, "thread"):
                self.thread.join(timeout=1)
                if self.thread.is_alive():
                    raise RuntimeError("Capture relay did not stop")
        if self.failure:
            raise RuntimeError(f"Capture relay failed: {self.failure}") from self.failure


def check_audio(path: Path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnframes() < 1600:
            raise ValueError("Recording is too short; speak for at least a moment")


def _read_live_audio(path: Path, seconds):
    """Read bounded samples and their frame cursor before the WAV is finalized."""
    import numpy as np

    with path.open("rb") as source:
        with wave.open(source, "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected mono 16 kHz PCM")
            frames = max(0, (path.stat().st_size - source.tell()) // 2)
            wav.setpos(max(0, frames - int(seconds * 16000)))
            raw = wav.readframes(int(seconds * 16000))
    samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    return samples, frames


def read_live_audio(path: Path, seconds=20):
    return _read_live_audio(path, seconds)[0]


def read_audio_from(path: Path, start_frame=0, seconds=None):
    """Read forward from a committed PCM frame, including an unfinished WAV."""
    import numpy as np

    with path.open("rb") as source:
        with wave.open(source, "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected mono 16 kHz PCM")
            frames = min(wav.getnframes(), max(0, (path.stat().st_size - source.tell()) // 2))
            if start_frame > frames:
                raise ValueError("Recording shrank after transcription started")
            wav.setpos(start_frame)
            count = frames - start_frame
            if seconds is not None:
                count = min(count, int(seconds * 16000))
            raw = wav.readframes(count)
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def live_meter(path: Path):
    import math

    import numpy as np

    samples, frames = _read_live_audio(path, 0.02)
    rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
    level = max(0.0, min(1.0, (20 * math.log10(max(rms, 1e-9)) + 60) / 55))
    return {"frame": frames, "level": level}
