"""Persistent offline native model isolation with data-only pipe communication."""

import atexit
import ctypes
import ctypes.util
import errno
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import threading
import weakref
from contextlib import ExitStack, suppress
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

from .config import Config, runtime_dir
from .native_protocol import AudioInput, ReadablePipe, pcm_samples, read_message, write_message
from .processes import python_command

_WORKERS: "weakref.WeakSet[NativeWorkerBackend]" = weakref.WeakSet()
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


def asr_config(config: Config) -> dict[str, object]:
    values = asdict(config)
    return {key: values[key] for key in ASR_FIELDS}


def isolate_network() -> None:
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

    def no_dns(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError("Network disabled in native speech worker")

    socket.getaddrinfo = no_dns


def worker_main() -> None:
    source = sys.stdin.buffer
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    # Native/Python diagnostics must not corrupt framing or enter daemon logs.
    sys.stdout = sys.stderr
    os.dup2(sys.stderr.fileno(), 1)
    try:
        isolate_network()
        request, _ = read_message(source)
        config = Config.from_mapping(request["config"])
        from . import nemo_asr, transformers_asr

        backend = (
            nemo_asr.NemoBackend
            if config.engine == "nemo"
            else transformers_asr.TransformersBackend
        )(config)
        write_message(
            output,
            {
                "result": "ready",
                "native_streaming": getattr(backend, "supports_native_streaming", False),
            },
        )
    # Report through the private worker protocol; never log input data.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        write_message(output, {"error": str(exc)[:1000]})
        return
    while True:
        try:
            request, samples = read_message(source, allow_audio=True)
        except EOFError:
            return
        try:
            method = request["method"]
            result: object
            if method == "reset_stream":
                if hasattr(backend, "reset_stream"):
                    backend.reset_stream()
                result = None
            else:
                config = Config.from_mapping(request["config"])
                if samples is None:
                    raise ValueError("Missing native speech audio")
                if method == "stream":
                    stream = getattr(backend, "stream", None)
                    if not callable(stream):
                        raise ValueError("Native speech model does not support streaming")
                    result = stream(
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
        # Report through the private worker protocol; never log input data.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            # Inference exceptions may echo model output; diagnostics stay structural.
            response = {
                "error": f"Native speech operation failed ({type(exc).__name__}); "
                "verify engine settings and record again"
            }
        write_message(output, response)


class NativeWorkerBackend:
    def __init__(self, config: Config) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.directory: tempfile.TemporaryDirectory[str] | None = None
        self.lock = threading.Lock()
        self._close_lock = threading.Lock()
        for module in ("torch", "transformers" if config.engine == "transformers" else "nemo"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"Install the {config.engine} optional dependency extra")
        self.timeout = config.native_timeout_seconds
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
        try:
            self.process = subprocess.Popen(  # noqa: S603
                python_command("mouthkeys.native_worker"),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                close_fds=True,
                bufsize=0,
            )
            _WORKERS.add(self)
            if self._call({"method": "load", "config": asr_config(config)}) != "ready":
                raise RuntimeError("Native speech worker did not finish loading")
        except BaseException:
            self.close()
            raise

    def alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def _broken_connection(self) -> NoReturn:
        self.close()
        raise RuntimeError("Native speech worker exited; record again") from None

    def _response(self, stream: ReadablePipe) -> object:
        try:
            response, _ = read_message(stream, self.timeout)
            if "error" in response:
                error = response["error"]
                if not isinstance(error, str):
                    raise ValueError("Invalid native speech error")
                raise RuntimeError(error)
            words = response.get("words", [])
            if "result" not in response or not isinstance(words, list):
                raise ValueError("Incomplete native speech response")
        except TimeoutError:
            self.close()
            raise RuntimeError(
                f"Native speech exceeded {self.timeout}s; worker stopped. "
                "Increase native timeout or choose a faster model, then retry."
            ) from None
        except (EOFError, OSError, ValueError, RecursionError):
            self._broken_connection()
        self.words = words
        if "native_streaming" in response:
            self.native_streaming = response["native_streaming"] is True
        return response["result"]

    def _call(self, request: dict[str, object], samples: AudioInput | None = None) -> object:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                raise RuntimeError("Native speech worker stopped; reload model before retrying")
            writer, reader = process.stdin, process.stdout
            if writer is None or reader is None:
                self._broken_connection()
            try:
                write_message(writer, request, samples, timeout=self.timeout)
            except TimeoutError:
                self.close()
                raise RuntimeError(
                    "Native speech input timed out; worker stopped. Record again."
                ) from None
            except OSError:
                self._broken_connection()
            return self._response(reader)

    def segments(self, samples: AudioInput, config: Config) -> list[SimpleNamespace]:
        self.timeout = config.native_timeout_seconds
        request: dict[str, object] = {"method": "segments", "config": asr_config(config)}
        audio = pcm_samples(samples)
        if len(audio) <= FILE_WINDOW_SAMPLES:
            return self._decode_segments(request, audio)

        # Reuse bounded VAD spans instead of splitting words or sending a multi-hour file.
        from .native_asr import speech_spans

        output = []
        for span in speech_spans(audio):
            start, end = span["start"], span["end"]
            output.extend(
                self._decode_segments(
                    {**request, "presegmented": True},
                    audio[start:end],
                    offset=start / 16000,
                )
            )
        self.words = [vars(word) for segment in output for word in segment.words]
        return output

    def _decode_segments(
        self, request: dict[str, object], audio: AudioInput, *, offset: float = 0
    ) -> list[SimpleNamespace]:
        result = self._call(request, audio)
        if not isinstance(result, list):
            self._broken_connection()
        try:
            return [self._segment(item, offset) for item in result]
        except (KeyError, TypeError):
            self._broken_connection()

    @staticmethod
    def _segment(item: object, offset: float) -> SimpleNamespace:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise TypeError("Invalid native speech segment")

        def shift(entry: dict[str, Any]) -> dict[str, Any]:
            return {**entry, "start": entry["start"] + offset, "end": entry["end"] + offset}

        return SimpleNamespace(
            **{
                **shift(item),
                "words": [SimpleNamespace(**shift(word)) for word in item.get("words", [])],
            }
        )

    def stream(
        self, samples: AudioInput, config: Config, session_key: object, *, final: bool = False
    ) -> str:
        self.timeout = config.native_timeout_seconds
        key = (
            [str(value) if isinstance(value, Path) else value for value in session_key]
            if isinstance(session_key, tuple | list)
            else session_key
        )
        result = self._call(
            {"method": "stream", "config": asr_config(config), "session_key": key, "final": final},
            samples,
        )
        if not isinstance(result, str):
            self._broken_connection()
        return result

    def reset_stream(self) -> None:
        if self.alive():
            self._call({"method": "reset_stream"})

    def close(self) -> None:
        with self._close_lock:
            process = self.process
            if process is not None:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                process.wait(timeout=3)
                self.process = None
            with ExitStack() as cleanup:
                cleanup.callback(_WORKERS.discard, self)
                if self.directory is not None:
                    directory, self.directory = self.directory, None
                    cleanup.callback(directory.cleanup)
                if process is not None:
                    for pipe in (process.stdout, process.stdin):
                        if pipe is not None:
                            cleanup.callback(pipe.close)

    def __del__(self) -> None:
        # Teardown must not raise from a finalizer, including after partial construction.
        with suppress(Exception):
            self.close()


@atexit.register
def close_workers() -> None:
    for worker in list(_WORKERS):
        with suppress(OSError, subprocess.TimeoutExpired):
            worker.close()


if __name__ == "__main__":
    worker_main()
