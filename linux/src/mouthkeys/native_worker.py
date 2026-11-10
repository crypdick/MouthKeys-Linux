"""Persistent offline native model isolation with data-only pipe communication."""

import atexit
import ctypes
import ctypes.util
import errno
import importlib.util
import json
import os
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .config import Config, runtime_dir
from .processes import python_command

_WORKERS = weakref.WeakSet()
MAX_HEADER = 8 * 1024 * 1024
FILE_WINDOW_SAMPLES = 300 * 16000
ASR_FIELDS = (
    "model",
    "engine",
    "language",
    "device",
    "compute_type",
    "model_revision",
    "asr_max_tokens",
    "native_streaming",
    "streaming_latency",
    "native_timeout_seconds",
    "vocabulary_boosting",
    "pronunciation_matching",
)


def asr_config(config):
    values = asdict(config)
    return {key: values[key] for key in ASR_FIELDS}


def _write_message(stream, value, samples=None, *, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    if samples is not None:
        import numpy as np

        samples = np.asarray(samples, dtype="<f4").reshape(-1)
        value = {**value, "audio_samples": len(samples)}
    header = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(header) > MAX_HEADER:
        raise ValueError("Native speech message exceeds 8 MiB")
    chunks = [struct.pack("!I", len(header)), header]
    if samples is not None:
        chunks.append(memoryview(samples).cast("B"))
    for chunk in chunks:
        view = memoryview(chunk)
        while view:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [stream], [], remaining)[1]:
                    raise TimeoutError("Native speech request timed out")
                # PIPE_BUF-sized writes after readiness cannot block waiting for
                # the worker to consume a large audio message.
                written = stream.write(view[:4096])
            else:
                written = stream.write(view)
            if not written:
                raise BrokenPipeError("Native speech pipe closed")
            view = view[written:]
    stream.flush()


def _read_exact(stream, length, deadline=None):
    value = bytearray()
    while len(value) < length:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                raise TimeoutError("Native speech response timed out")
        chunk = stream.read(min(65536, length - len(value)))
        if not chunk:
            raise EOFError("Native speech pipe closed")
        value.extend(chunk)
    return value


def _read_message(stream, timeout=None, *, allow_audio=False):
    deadline = None if timeout is None else time.monotonic() + timeout
    length = struct.unpack("!I", _read_exact(stream, 4, deadline))[0]
    if not 0 < length <= MAX_HEADER:
        raise ValueError("Invalid native speech message size")
    value = json.loads(_read_exact(stream, length, deadline))
    if not isinstance(value, dict):
        raise ValueError("Invalid native speech message")
    samples = None
    if "audio_samples" in value:
        if not allow_audio:
            raise ValueError("Unexpected audio in native speech response")
        count = value["audio_samples"]
        if type(count) is not int or not 0 <= count <= 2**31:
            raise ValueError("Invalid native speech audio size")
        import numpy as np

        samples = np.frombuffer(_read_exact(stream, count * 4, deadline), dtype="<f4")
    return value, samples


def isolate_network():
    """Deny IPv4/IPv6 socket creation; preserve local PyTorch Unix-socket IPC."""
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("Native speech isolation requires libseccomp (libseccomp2 on Debian)")
    seccomp = ctypes.CDLL(library)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int

    class Comparison(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_int),
            ("datum_a", ctypes.c_uint64),
            ("datum_b", ctypes.c_uint64),
        ]

    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Comparison),
    ]
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("Cannot initialize native speech network isolation")
    try:
        for name in (b"socket", b"socketpair"):
            syscall = seccomp.seccomp_syscall_resolve_name(name)
            if syscall < 0:
                raise RuntimeError("Native speech network isolation unsupported on this platform")
            for family in (socket.AF_INET, socket.AF_INET6):
                comparison = Comparison(0, 4, family, 0)  # SCMP_CMP_EQ
                if (
                    seccomp.seccomp_rule_add_array(
                        context,
                        0x00050000 | errno.EPERM,
                        syscall,
                        1,
                        ctypes.byref(comparison),
                    )
                    < 0
                ):
                    raise RuntimeError("Cannot configure native speech network isolation")
        if seccomp.seccomp_load(context) < 0:
            raise RuntimeError("Cannot enable native speech network isolation")
    finally:
        seccomp.seccomp_release(context)

    def no_dns(*args, **kwargs):
        raise OSError("Network disabled in native speech worker")

    socket.getaddrinfo = no_dns


def worker_main():
    source = sys.stdin.buffer
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    # Native/Python diagnostics must not corrupt framing or enter daemon logs.
    sys.stdout = sys.stderr
    os.dup2(sys.stderr.fileno(), 1)
    try:
        isolate_network()
        request, _ = _read_message(source)
        config = Config(**request["config"]).validate()
        from .native_asr import NemoBackend, TransformersBackend

        backend = (NemoBackend if config.engine == "nemo" else TransformersBackend)(config)
        _write_message(output, {"result": "ready"})
    except Exception as exc:
        _write_message(output, {"error": str(exc)[:1000]})
        return
    while True:
        try:
            request, samples = _read_message(source, allow_audio=True)
        except EOFError:
            return
        try:
            method = request["method"]
            if method == "reset_stream":
                if hasattr(backend, "reset_stream"):
                    backend.reset_stream()
                result = None
            else:
                config = Config(**request["config"]).validate()
                if method == "stream":
                    result = backend.stream(
                        samples, config, request["session_key"], final=request.get("final", False)
                    )
                elif method == "segments":
                    result = [
                        {
                            "start": float(s.start),
                            "end": float(s.end),
                            "text": s.text,
                            "words": getattr(s, "words", []),
                        }
                        for s in backend.segments(
                            samples, config, presegmented=request.get("presegmented", False)
                        )
                    ]
                else:
                    raise ValueError("Unknown native operation")
            response = {"result": result, "words": getattr(backend, "words", [])}
        except Exception as exc:
            # Inference exceptions may echo model output; diagnostics stay structural.
            response = {
                "error": f"Native speech operation failed ({type(exc).__name__}); "
                "verify engine settings or reprocess retained audio"
            }
        _write_message(output, response)


class NativeWorkerBackend:
    def __init__(self, config):
        for module in ("torch", "transformers" if config.engine == "transformers" else "nemo"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"Install the {config.engine} optional dependency extra")
        self.timeout = getattr(config, "native_timeout_seconds", 300)
        self.lock = threading.Lock()
        self.directory = tempfile.TemporaryDirectory(prefix="native-", dir=runtime_dir())
        environment = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "WANDB_MODE": "disabled",
            "DO_NOT_TRACK": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
            "MOUTHKEYS_NATIVE_STATE": self.directory.name,
        }
        self.process = subprocess.Popen(
            python_command("mouthkeys.native_worker"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            close_fds=True,
            bufsize=0,
        )
        _WORKERS.add(self)
        try:
            _write_message(self.process.stdin, {"method": "load", "config": asr_config(config)})
            self._response()
        except BaseException:
            self.close()
            raise

    def alive(self):
        return self.process.poll() is None

    def _response(self):
        try:
            response, _ = _read_message(self.process.stdout, self.timeout)
        except TimeoutError:
            self.close()
            raise RuntimeError(
                f"Native speech exceeded {self.timeout}s; worker stopped. "
                "Increase native timeout or choose a faster model, then retry."
            ) from None
        except (EOFError, OSError, ValueError):
            self.close()
            raise RuntimeError("Native speech worker exited; audio can be reprocessed") from None
        if "error" in response:
            raise RuntimeError(response["error"])
        self.words = response.get("words", [])
        return response["result"]

    def _call(self, request, samples=None):
        with self.lock:
            if not self.alive():
                raise RuntimeError("Native speech worker stopped; reload model before retrying")
            try:
                _write_message(self.process.stdin, request, samples, timeout=self.timeout)
            except TimeoutError:
                self.close()
                raise RuntimeError(
                    "Native speech input timed out; worker stopped. Audio can be reprocessed."
                ) from None
            except OSError:
                self.close()
                raise RuntimeError(
                    "Native speech worker exited; audio can be reprocessed"
                ) from None
            return self._response()

    def segments(self, samples, config):
        self.timeout = getattr(config, "native_timeout_seconds", self.timeout)
        request = {"method": "segments", "config": asr_config(config)}
        if len(samples) <= FILE_WINDOW_SAMPLES:
            result = self._call(request, samples)
            return [self._segment(segment) for segment in result]
        # Do not serialize an entire multi-hour file into a worker pipe. Reuse
        # offline VAD's <=25-second speech spans; preserve exact source offsets
        # and do not split a word at arbitrary transport window boundaries.
        from .native_asr import speech_spans

        output = []
        for span in speech_spans(samples):
            start, end = span["start"], span["end"]
            result = self._call({**request, "presegmented": True}, samples[start:end])
            output.extend(
                self._segment(
                    {
                        **item,
                        "start": item["start"] + start / 16000,
                        "end": item["end"] + start / 16000,
                        "words": [
                            {
                                **word,
                                "start": word["start"] + start / 16000,
                                "end": word["end"] + start / 16000,
                            }
                            for word in item.get("words", [])
                        ],
                    }
                )
                for item in result
            )
        self.words = [vars(word) for segment in output for word in segment.words]
        return output

    @staticmethod
    def _segment(item):
        return SimpleNamespace(
            **{**item, "words": [SimpleNamespace(**word) for word in item.get("words", [])]}
        )

    def stream(self, samples, config, session_key, *, final=False):
        self.timeout = getattr(config, "native_timeout_seconds", self.timeout)
        key = (
            [str(value) if isinstance(value, Path) else value for value in session_key]
            if isinstance(session_key, tuple | list)
            else session_key
        )
        return self._call(
            {"method": "stream", "config": asr_config(config), "session_key": key, "final": final},
            samples,
        )

    def reset_stream(self):
        if self.alive():
            self._call({"method": "reset_stream"})

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait(timeout=3)
        process.stdin.close()
        process.stdout.close()
        directory = getattr(self, "directory", None)
        if directory is not None:
            directory.cleanup()
        _WORKERS.discard(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@atexit.register
def close_workers():
    for worker in list(_WORKERS):
        worker.close()


if __name__ == "__main__":
    worker_main()
