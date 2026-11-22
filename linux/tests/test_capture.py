import io
import os
import shutil
import signal
import subprocess
import sys
import time
import wave
from types import SimpleNamespace

import pytest

from mouthkeys import capture
from mouthkeys.audio import Recorder
from mouthkeys.config import Config


@pytest.fixture(autouse=True)
def fake_bluetooth_readiness(monkeypatch):
    monkeypatch.setattr("mouthkeys.capture_recorder.bluetooth_capture_ready", lambda _: True)


def test_supervisor_passes_through_completed_child_status_without_signalling(monkeypatch):
    def wait(*, timeout):
        assert timeout == 2
        return 7

    process = SimpleNamespace(poll=lambda: 7, wait=wait)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    assert capture.supervise(["fake-capture"]) == 7


def test_supervisor_waits_for_eof_then_escalates_unresponsive_child(monkeypatch):
    pipe = io.BytesIO(b"!")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=pipe))
    invocations = []

    def select(readers, _writers, _errors, timeout):
        assert readers == [pipe]
        assert timeout == 0.1
        invocations.append("select")
        return readers, [], []

    def wait(*, timeout):
        invocations.append(("wait", timeout))
        if timeout == 2:
            raise subprocess.TimeoutExpired("fake-capture", timeout)
        return -signal.SIGKILL

    process = SimpleNamespace(
        poll=lambda: None,
        send_signal=lambda signum: invocations.append(("signal", signum)),
        kill=lambda: invocations.append("kill"),
        wait=wait,
    )
    monkeypatch.setattr(capture.select, "select", select)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    assert capture.supervise(["fake-capture"]) == 1
    assert invocations == [
        "select",
        "select",
        ("signal", signal.SIGINT),
        ("wait", 2),
        "kill",
        ("wait", 1),
    ]


@pytest.mark.parametrize("crash", [False, True])
def test_capture_lifeline_stops_child_on_stop_or_daemon_death(tmp_path, crash):
    ready, stopped = tmp_path / "ready", tmp_path / "stopped"
    capture_script = tmp_path / "fake_capture.py"
    capture_script.write_text(
        "import signal, time\nfrom pathlib import Path\n"
        f"def stop(*_):\n    Path({str(stopped)!r}).touch()\n    raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(0.1)\n"
    )
    # Stand-in daemon owns only the lifeline; no capture device is opened.
    parent_script = tmp_path / "fake_daemon.py"
    parent_script.write_text(
        "import subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, '-m', 'mouthkeys.capture', "
        f"sys.executable, {str(capture_script)!r}], stdin=subprocess.PIPE)\n"
        "sys.stdin.buffer.read(1)\n"
        "p.stdin.close()\np.wait(timeout=4)\n"
    )
    parent = subprocess.Popen([sys.executable, str(parent_script)], stdin=subprocess.PIPE)  # noqa: S603
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Fake capture did not start"
        if crash:
            parent.kill()
        else:
            parent.stdin.close()
        parent.wait(timeout=4)
        deadline = time.monotonic() + 3
        while not stopped.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert stopped.exists(), "Capture survived loss of daemon lifeline"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if not parent.stdin.closed:
            parent.stdin.close()


def test_recorder_uses_supervised_ffmpeg_and_finishes_wav(monkeypatch, tmp_path):
    class FakeProcess:
        def __init__(self, args, **kwargs):
            assert args[1:5] == ["-P", "-m", "mouthkeys.capture", "ffmpeg"]
            assert args[args.index("-f") + 1] == "pulse"
            before_input = args[: args.index("-i")]
            assert before_input[before_input.index("-sample_rate") + 1] == "16000"
            assert before_input[before_input.index("-channels") + 1] == "1"
            assert before_input[before_input.index("-fragment_size") + 1] == "640"
            assert before_input[before_input.index("-probesize") + 1] == "32"
            assert kwargs["stdin"] == subprocess.PIPE
            assert kwargs["start_new_session"]
            self.stdin = io.BytesIO()
            self.pid = 999999
            with wave.open(str(tmp_path / "recording.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(bytes(640))

        def poll(self):
            return None

        def wait(self, timeout):
            assert self.stdin.closed
            return 255

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    recorder = Recorder(Config(), tmp_path / "recording.wav")
    recorder.stop()
    assert recorder.errors.closed


@pytest.mark.parametrize("outcome", ["ready", "exit", "stall"])
def test_recorder_requires_samples_before_reporting_ready(monkeypatch, tmp_path, outcome):
    path = tmp_path / "recording.wav"
    elapsed = 0
    processes = []

    class FakeProcess:
        def __init__(self, args, **kwargs):
            self.stdin = io.BytesIO()
            self.pid = 999999
            self.errors = kwargs["stderr"]
            processes.append(self)

        def poll(self):
            return 1 if outcome == "exit" and elapsed >= 0.5 else None

        def wait(self, timeout):
            assert self.stdin.closed
            return 0

    def advance(seconds):
        nonlocal elapsed
        elapsed += seconds
        if outcome == "ready":
            if elapsed < 0.2:
                path.write_bytes(b"RIFF")  # A partially written header is not ready.
            else:
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    # Silence counts as readiness; do not wait for speech.
                    wav.writeframes(bytes(640) if elapsed >= 0.5 else b"")
        elif outcome == "exit" and elapsed >= 0.5:
            processes[0].errors.write(b"Source unplugged")

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(time, "sleep", advance)
    if outcome == "ready":
        recorder = Recorder(Config(), path)
        with wave.open(str(path), "rb") as wav:
            assert len(wav.readframes(320)) == 640
        recorder.stop()
    else:
        message = "Source unplugged" if outcome == "exit" else "Microphone did not produce audio"
        with pytest.raises(RuntimeError, match=message):
            Recorder(Config(), path)
        assert elapsed <= 5.1
    assert processes[0].stdin.closed
    assert processes[0].errors.closed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_ffmpeg_startup_streams_audio_and_preserves_first_buffer(monkeypatch, tmp_path):
    raw = tmp_path / "source.pcm"
    first_buffer = b"\x34\x12" * 320
    raw.write_bytes(first_buffer + bytes(4 * 16000 * 2))
    path = tmp_path / "recording.wav"
    popen = subprocess.Popen

    def synthetic_capture(args, **kwargs):
        # Keep the real supervisor, FFmpeg probing, and WAV output; replace only
        # the microphone with PCM read at real time. No audio device is opened.
        args = list(args)
        index = args.index("-f")
        args[index + 1] = "s16le"
        args[args.index("-sample_rate")] = "-ar"
        args[args.index("-channels")] = "-ac"
        index = args.index("-fragment_size")
        del args[index : index + 2]
        args.insert(args.index("-i"), "-re")
        args[args.index("-i") + 1] = str(raw)
        return popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", synthetic_capture)
    started = time.monotonic()
    recorder = Recorder(Config(), path)
    try:
        assert time.monotonic() - started < 2, "FFmpeg buffered audio during startup"
        with wave.open(str(path), "rb") as wav:
            assert wav.readframes(320) == first_buffer
    finally:
        recorder.stop()
    with wave.open(str(path), "rb") as wav:
        assert wav.readframes(320) == first_buffer


def test_recorder_reaps_process_that_exits_while_escalating_stop(monkeypatch, tmp_path):
    path = tmp_path / "race.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(640))
    waits = []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("capture", timeout)
        return 0

    def exited(_pid, _signal):
        raise ProcessLookupError("Capture exited")

    process = SimpleNamespace(stdin=io.BytesIO(), pid=999999, poll=lambda: None, wait=wait)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_opts: process)
    monkeypatch.setattr(os, "killpg", exited)
    recorder = Recorder(Config(), path)
    recorder.stop()
    assert waits == [4, 3]
    assert process.stdin.closed
    assert recorder.errors.closed


def test_cancelled_start_still_reaps_and_closes_the_error_stream(monkeypatch, tmp_path):
    pipe = io.BytesIO()
    waits = []
    captured = []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("capture", timeout)
        return -signal.SIGKILL

    def spawn(_args, **options):
        captured.append(options["stderr"])
        return SimpleNamespace(stdin=pipe, pid=999999, poll=lambda: None, wait=wait)

    killed = []
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(os, "killpg", lambda pid, stop: killed.append((pid, stop)))
    with pytest.raises(RuntimeError, match="Microphone start cancelled"):
        Recorder(Config(), tmp_path / "cancelled.wav", cancelled=lambda: True)
    assert killed == [(999999, signal.SIGKILL)]
    assert waits == [3, 1]
    assert captured[0].closed
    assert pipe.closed


@pytest.mark.parametrize("stderr", [b"", b"Input failed"])
def test_recorder_stop_reports_nonzero_exit(monkeypatch, tmp_path, stderr):
    path = tmp_path / "recording.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(640))

    def spawn(_args, **options):
        options["stderr"].write(stderr)
        return SimpleNamespace(stdin=None, poll=lambda: None, wait=lambda timeout: 7)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    recorder = Recorder(Config(), path)
    message = "Input failed" if stderr else "exited with status 7"
    with pytest.raises(RuntimeError, match=message):
        recorder.stop()
    assert recorder.errors.closed


def test_forced_stop_discards_temporary_audio_after_reaping(monkeypatch, tmp_path):
    path = tmp_path / "recording.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(640))
    killed = []

    def wait(timeout):
        if not killed:
            raise subprocess.TimeoutExpired("capture", timeout)
        assert timeout == 3
        return -signal.SIGKILL

    process = SimpleNamespace(stdin=io.BytesIO(), pid=999999, poll=lambda: None, wait=wait)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_opts: process)
    monkeypatch.setattr(os, "killpg", lambda pid, stop: killed.append((pid, stop)))
    recorder = Recorder(Config(), path)
    with pytest.raises(RuntimeError, match="temporary audio will be discarded"):
        recorder.stop()
    assert killed == [(999999, signal.SIGKILL)]
    assert recorder.errors.closed
