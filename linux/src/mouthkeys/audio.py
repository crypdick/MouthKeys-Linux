import json
import os
import signal
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from .config import Config
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
                        samples_ready = len(wav.readframes(320)) >= 640
                    if samples_ready and bluetooth_capture_ready(config.audio_source):
                        break  # Silence is valid only after the physical input is ready.
                except (OSError, EOFError, wave.Error):
                    pass  # FFmpeg may still be opening the source or writing the WAV header.
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Microphone did not produce audio or become ready within 5 seconds"
                    )
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
