import os
import signal
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from .config import Config
from .processes import python_command


class Recorder:
    def __init__(self, config: Config, path: Path):
        self.errors = tempfile.TemporaryFile()
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
            deadline = time.monotonic() + 5
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(self.error() or "Microphone capture failed")
                try:
                    with wave.open(str(path), "rb") as wav:
                        if len(wav.readframes(320)) >= 640:
                            break  # A saved 20 ms buffer, including silence, proves readiness.
                except (OSError, EOFError, wave.Error):
                    pass  # FFmpeg may still be opening the source or writing the WAV header.
                if time.monotonic() >= deadline:
                    raise RuntimeError("Microphone did not produce audio within 5 seconds")
                time.sleep(0.02)
        except BaseException:
            if self.process is not None:
                self.process.stdin.close()
                self.process.wait(timeout=3)
            self.errors.close()
            raise

    def error(self):
        self.errors.seek(0)
        return self.errors.read(4096).decode(errors="replace").strip()

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
