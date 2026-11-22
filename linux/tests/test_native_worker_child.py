"""Exercise the native worker entry point with isolated fake speech engines."""

import ctypes.util
import io
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mouthkeys import native_protocol, native_worker
from mouthkeys.config import Config


def test_kernel_network_isolation_preserves_local_ipc_in_child():
    if not ctypes.util.find_library("seccomp"):
        pytest.skip("Optional native worker requires libseccomp")
    script = """
import socket
import sys
from mouthkeys.native_worker import isolate_network
assert "numpy" not in sys.modules
isolate_network()
try:
    socket.getaddrinfo("localhost", 0)
except OSError as exc:
    assert "Network disabled" in str(exc)
else:
    raise AssertionError("Resolver was enabled")
for family in (socket.AF_INET, socket.AF_INET6):
    try:
        socket.socket(family)
    except PermissionError:
        pass
    else:
        raise AssertionError('Internet socket allowed')
left, right = socket.socketpair()
left.send(b'local')
assert right.recv(5) == b'local'
left.close()
right.close()
print('isolated')
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-P", "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"


@pytest.mark.parametrize("engine", ["nemo", "transformers"])
def test_worker_runs_real_framing_for_backend_selection_and_redacts_inference_errors(engine):
    if not ctypes.util.find_library("seccomp"):
        pytest.skip("Optional native worker requires libseccomp")
    script = """
from types import SimpleNamespace

from mouthkeys import native_worker, nemo_asr, transformers_asr


class Offline:
    def __init__(self, config):
        assert config.engine == "transformers"
        self.words = []

    def segments(self, samples, config, *, presegmented):
        assert len(samples) == 2 and presegmented
        return [SimpleNamespace(start=0.0, end=0.125, text="decoded file")]


class Streaming:
    supports_native_streaming = True

    def __init__(self, config):
        assert config.engine == "nemo"
        self.words = []

    def stream(self, samples, config, session_key, *, final):
        if session_key == "invalid":
            raise ValueError("private inference payload")
        assert session_key == ["test", "take"] and final and len(samples) == 3
        self.words = [{"word": "decoded", "start": 0, "end": 0.125}]
        return "decoded"

    def segments(self, samples, config, *, presegmented):
        return Offline.segments(self, samples, config, presegmented=presegmented)

    def reset_stream(self):
        self.words = []


nemo_asr.NemoBackend = Streaming
transformers_asr.TransformersBackend = Offline
native_worker.worker_main()
"""
    config = native_worker.asr_config(replace(Config(), engine=engine))
    payload = io.BytesIO()
    native_protocol.write_message(payload, {"config": config})
    if engine == "nemo":
        native_protocol.write_message(
            payload,
            {"method": "stream", "config": config, "session_key": ["test", "take"], "final": True},
            [0, 1, 2],
        )
        native_protocol.write_message(
            payload, {"method": "stream", "config": config, "session_key": "invalid"}, [0]
        )
    else:
        native_protocol.write_message(payload, {"method": "unknown", "config": config}, [0])
        native_protocol.write_message(payload, {"method": "stream", "config": config}, [0])
    native_protocol.write_message(payload, {"method": "segments", "config": config})
    native_protocol.write_message(payload, {"method": "reset_stream"})
    native_protocol.write_message(
        payload, {"method": "segments", "config": config, "presegmented": True}, [0, 1]
    )

    result = subprocess.run(  # noqa: S603 - local worker using fake models
        [sys.executable, "-P", "-c", script],
        input=payload.getvalue(),
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert b"private inference payload" not in result.stdout + result.stderr
    output = io.BytesIO(result.stdout)
    ready, _ = native_protocol.read_message(output)
    assert ready == {"result": "ready", "native_streaming": engine == "nemo"}
    if engine == "nemo":
        response, _ = native_protocol.read_message(output)
        assert response == {
            "result": "decoded",
            "words": [{"word": "decoded", "start": 0, "end": 0.125}],
        }
    for _ in range(2 if engine == "nemo" else 3):
        failure, _ = native_protocol.read_message(output)
        assert "failed (ValueError)" in failure["error"]
    reset, _ = native_protocol.read_message(output)
    assert reset == {"result": None, "words": []}
    segments, _ = native_protocol.read_message(output)
    assert segments == {
        "result": [{"start": 0, "end": 0.125, "text": "decoded file", "words": []}],
        "words": [],
    }
    assert not output.read()


def test_worker_loader_failure_is_returned_before_any_inference():
    if not ctypes.util.find_library("seccomp"):
        pytest.skip("Optional native worker requires libseccomp")
    payload = io.BytesIO()
    native_protocol.write_message(
        payload, {"config": native_worker.asr_config(Config(engine="nemo"))}
    )
    script = """
from mouthkeys import native_worker, nemo_asr

def missing(config):
    raise RuntimeError("Requested local archive unavailable")

nemo_asr.NemoBackend = missing
native_worker.worker_main()
"""
    result = subprocess.run(  # noqa: S603 - local worker using fake models
        [sys.executable, "-P", "-c", script],
        input=payload.getvalue(),
        capture_output=True,
        timeout=10,
        check=True,
    )
    response, _ = native_protocol.read_message(io.BytesIO(result.stdout))
    assert response == {"error": "Requested local archive unavailable"}


@pytest.mark.parametrize(
    ("failure", "release_count", "error"),
    [
        ("library", 0, "requires libseccomp"),
        ("init", 0, "Cannot initialize"),
        ("resolve", 1, "unsupported on this platform"),
        ("rule", 1, "Cannot configure"),
        ("load", 1, "Cannot enable"),
    ],
)
def test_isolation_failures_stop_worker_before_using_model(
    monkeypatch, failure, release_count, error
):
    releases = []

    def release(context):
        releases.append(context)

    library = SimpleNamespace(
        seccomp_init=lambda *_: 0 if failure == "init" else 1,
        seccomp_syscall_resolve_name=lambda *_: -1 if failure == "resolve" else 1,
        seccomp_rule_add_array=lambda *_: -1 if failure == "rule" else 0,
        seccomp_load=lambda *_: -1 if failure == "load" else 0,
        seccomp_release=release,
    )
    monkeypatch.setattr(
        native_worker.ctypes.util, "find_library", lambda _: "" if failure == "library" else "fake"
    )
    monkeypatch.setattr(native_worker.ctypes, "CDLL", lambda _: library)
    with pytest.raises(RuntimeError, match=error):
        native_worker.isolate_network()
    assert len(releases) == release_count


def test_worker_module_exits_cleanly_on_startup_eof():
    if not ctypes.util.find_library("seccomp"):
        pytest.skip("Optional native worker requires libseccomp")
    result = subprocess.run(
        [sys.executable, "-P", "-m", "mouthkeys.native_worker"],
        input=b"",
        capture_output=True,
        check=True,
        timeout=10,
    )
    response, _ = native_protocol.read_message(io.BytesIO(result.stdout))
    assert response == {"error": "Native speech pipe closed"}
