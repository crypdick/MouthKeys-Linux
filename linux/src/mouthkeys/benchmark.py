"""Explicit, offline, fixture-scoped speech measurements; never record or deliver text."""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

from .benchmark_context import decode_settings, hardware_context, model_identity, software_context
from .benchmark_reports import save_measurement
from .config import Config, private_dir

if TYPE_CHECKING:
    WorkerProcess = subprocess.Popen[bytes]
else:
    WorkerProcess = object

MAX_REFERENCE = 16000
MAX_FIXTURE_BYTES = 128 * 1024 * 1024
MAX_ARTIFACT_FILES = 10000
MAX_METRIC_CELLS = 4_000_000
MEMORY_SCOPE = (
    "Sampled sum of benchmark worker and descendant resident sets, including model loading. "
    "20 ms sampling can miss peaks; shared pages may be counted twice. GPU VRAM excluded. "
    "At most 128 processes observed per sample."
)
SCOPE = (
    "One offline WAV fixture, fresh process/model load then one file decode. "
    "OS/device caches not flushed; not general model accuracy."
)


def _distance(left: Sequence[str], right: Sequence[str]) -> int | None:
    if len(left) * len(right) > MAX_METRIC_CELLS:
        return None
    if len(right) > len(left):
        left, right = right, left
    row = list(range(len(right) + 1))
    for index, item in enumerate(left, 1):
        following = [index]
        for column, other in enumerate(right, 1):
            following.append(
                min(following[-1] + 1, row[column] + 1, row[column - 1] + (item != other))
            )
        row = following
    return row[-1]


def accuracy(reference: str, hypothesis: str) -> dict[str, str | int | float | None]:
    """Exact Levenshtein rates, with a bounded cell budget per metric."""

    def words(value: str) -> list[str]:
        return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold())

    expected, actual = words(reference), words(hypothesis)
    if not expected:
        raise ValueError("Reference must contain at least one letter or digit")
    result: dict[str, str | int | float | None] = {
        "algorithm": "levenshtein-nfkc-casefold-alphanumeric-v1",
        "normalization": "Unicode alphanumeric words; CER joins words without spaces/punctuation",
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "max_cells_per_metric": MAX_METRIC_CELLS,
    }
    for name, left, right in (
        ("wer", expected, actual),
        ("cer", "".join(expected), "".join(actual)),
    ):
        errors = _distance(left, right)
        result[name] = None if errors is None else errors / len(left)
        result[name + "_errors"] = errors
        result[name + "_reference_units"] = len(left)
        if errors is None:
            result[name + "_unavailable"] = "Comparison exceeds bounded metric work budget"
    return result


def _fixture(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > MAX_FIXTURE_BYTES:
        raise ValueError("Benchmark needs a WAV file no larger than 128 MiB")
    with wave.open(str(path)) as audio:
        duration = audio.getnframes() / audio.getframerate()
        if not 0 < duration <= 600:
            raise ValueError("Benchmark WAV must contain at most ten minutes of audio")
    return path


def _tree_rss(pid: int, proc: Path = Path("/proc")) -> int | None:
    pending = [pid]
    seen: set[int] = set()
    total, observed = 0, False
    while pending and len(seen) < 128:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = (proc / str(current) / "status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+) kB$", status, re.M)
            if match:
                total += int(match[1]) * 1024
                observed = True
            # The native worker can be spawned by a non-main inference thread.
            for task in (proc / str(current) / "task").iterdir():
                pending.extend(int(value) for value in (task / "children").read_text().split())
        except (OSError, ValueError):
            continue
    return total if observed else None


def _stop(process: WorkerProcess) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=3)


def _check_cancellation(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Benchmark cancelled")


def _sample_worker(
    process: WorkerProcess, deadline: float, cancel_event: Event | None
) -> tuple[int | None, int]:
    peak: int | None = None
    samples = 0
    while process.poll() is None:
        _check_cancellation(cancel_event)
        if time.monotonic() >= deadline:
            raise RuntimeError("Benchmark timed out; worker stopped")
        rss = _tree_rss(process.pid)
        if rss is not None:
            peak = max(peak or 0, rss)
            samples += 1
        time.sleep(0.02)
    return peak, samples


def run_benchmark(  # noqa: PLR0913 - independent fixture and benchmark options
    config: Config,
    fixture: str | Path,
    *,
    reference: object = None,
    timeout: object = 600,
    cancel_event: Event | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run a separate socket-denied worker. Reports retain neither text nor fixture paths."""
    config.validate()
    _check_cancellation(cancel_event)
    fixture = _fixture(fixture)
    if not isinstance(timeout, int | float) or not 1 <= timeout <= 3600:
        raise ValueError("Benchmark timeout must be between 1 and 3600 seconds")
    if reference is not None:
        if not isinstance(reference, str) or not 0 < len(reference) <= MAX_REFERENCE:
            raise ValueError("Reference must contain 1–16000 characters")
        accuracy(reference, "")  # Reject unusable references before model loading.
    request = {"config": asdict(config), "fixture": str(fixture), "reference": reference}
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as output:
        source.write(json.dumps(request).encode())
        source.seek(0)
        process = subprocess.Popen(
            [sys.executable, "-I", "-m", "mouthkeys.benchmark", "--worker"],
            stdin=source,
            stdout=output,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            peak, samples = _sample_worker(process, deadline, cancel_event)
            output.seek(0)
            raw = output.read(256 * 1024 + 1)
            if len(raw) > 256 * 1024 or process.returncode:
                raise RuntimeError(
                    "Benchmark worker failed; verify installed model and dependencies"
                )
            response = json.loads(raw)
            if "error" in response:
                raise RuntimeError(response["error"])
            report: dict[str, Any] = response["result"]
        finally:
            _stop(process)
    _check_cancellation(cancel_event)
    report.update(
        sampled_peak_tree_rss_bytes=peak,
        memory_samples=samples,
        memory_scope=MEMORY_SCOPE,
    )
    if save:
        save_measurement(report)
    return report


def _measurement_context(report: dict[str, Any]) -> tuple[object, object]:
    return report["fixture_sha256"], (report.get("accuracy") or {}).get("reference_sha256")


def matching_measurement(
    config: Config,
    reports: Sequence[dict[str, Any]],
    context: Sequence[object] | None = None,
) -> dict[str, Any] | None:
    """Never apply another device/language/library/artifact/fixture result to this row."""
    if not reports:
        return None

    from .native_asr import effective_revision  # noqa: PLC0415 - load benchmark dependencies only for this operation

    revision = (
        effective_revision(config.model, config.model_revision, config.engine)
        if config.engine in ("nemo", "transformers")
        else config.model_revision
    )
    hardware, software = hardware_context(), software_context()
    if config.device == "cpu":
        hardware = {key: hardware[key] for key in ("architecture", "cpu", "logical_cpus")}
        software.pop("nvidia_driver", None)
    reports = [
        report
        for report in reports
        if all(report["hardware"].get(key) == value for key, value in hardware.items())
        and {
            key: value
            for key, value in report["software"].items()
            if config.device != "cpu" or key != "nvidia_driver"
        }
        == software
    ]
    model = model_identity(config.model)
    settings = decode_settings(config)
    candidates = [
        report
        for report in reports
        if report["engine"] == config.engine
        and report["model"] == model
        and report["revision"] == revision
        and report["device"] == config.device
        and report["compute_type"] == config.compute_type
        and report.get("decode_settings") == settings
    ]
    if not candidates:
        return None
    scope = _measurement_context(reports[0]) if context is None else tuple(context)
    candidates = [report for report in candidates if _measurement_context(report) == scope]
    if not candidates:
        return None
    try:
        _, path = _local_config(config)
        artifact = _artifact_identity(path)
    except (OSError, RuntimeError, ValueError):
        return None
    return next((report for report in candidates if report.get("artifact") == artifact), None)


def _local_config(config: Config) -> tuple[Config, Path]:
    from .engines import READY, managed_path, onnx_path, resolve_whisper  # noqa: PLC0415 - load benchmark dependencies only for this operation
    from .native_asr import local_artifact  # noqa: PLC0415 - load benchmark dependencies only for this operation

    if config.engine == "whisper":
        path = Path(resolve_whisper(config))
    elif config.engine in ("nemo", "transformers"):
        path = local_artifact(config)
    else:
        path = Path(config.model).expanduser()
        if not path.is_dir():
            path = onnx_path(config.model, config.model_revision)
            if not (path / READY).is_file():
                raise RuntimeError("Benchmark requires an already installed ONNX model")
        if not (managed_path("silero") / READY).is_file():
            raise RuntimeError("Benchmark requires an already installed Silero VAD")
    path = path.resolve()
    return replace(config, model=str(path)), path


def _artifact_identity(path: Path) -> dict[str, str | None]:
    # A metadata fingerprint is quick even for multi-GB weights; never call it a weight hash.
    directory = path.is_dir()
    files = []
    for item in path.rglob("*") if directory else [path]:
        if not item.is_file():
            continue
        files.append(item)
        if len(files) > MAX_ARTIFACT_FILES:
            raise ValueError("Benchmark model contains too many files")
    manifest = []
    for item in sorted(files):
        stat = item.stat()
        name = str(item.relative_to(path)) if directory else item.name
        manifest.append([name, stat.st_size, stat.st_mtime_ns])
    return {
        "metadata_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(),
        "scope": "Relative file names, byte sizes and mtimes; not a weight-content hash",
        "snapshot_commit": path.name if path.parent.name == "snapshots" else None,
    }


def _copy_fixture(source: Path, destination: Path) -> str:
    digest, size = hashlib.sha256(), 0
    with source.open("rb") as incoming, destination.open("xb") as output:
        destination.chmod(0o600)
        while chunk := incoming.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FIXTURE_BYTES:
                raise ValueError("Benchmark WAV exceeds 128 MiB")
            digest.update(chunk)
            output.write(chunk)
    _fixture(destination)
    return digest.hexdigest()


def _measure(request: dict[str, Any]) -> dict[str, Any]:
    from .file_transcription import transcribe_file  # noqa: PLC0415 - load benchmark dependencies only for this operation
    from .native_asr import effective_revision  # noqa: PLC0415 - load benchmark dependencies only for this operation
    from .transcribe import Transcriber  # noqa: PLC0415 - load benchmark dependencies only for this operation

    original = Config.from_mapping(request["config"])
    config, artifact = _local_config(original)
    identity = _artifact_identity(artifact)
    fixture = _fixture(request["fixture"])
    # Do not consume private vocabulary, prompts, pronunciation data or save history.
    config = replace(
        config,
        vocabulary_boosting="off",
        pronunciation_matching="off",
        file_speaker_labels="off",
        ai_enabled="off",
    )
    software = software_context()
    with tempfile.TemporaryDirectory(prefix="mouthkeys-benchmark-") as temporary:
        snapshot = Path(temporary) / "fixture.wav"
        fixture_hash = _copy_fixture(fixture, snapshot)
        for kind in ("CONFIG", "STATE", "DATA", "RUNTIME"):
            path = private_dir(Path(temporary) / kind.lower())
            os.environ[f"XDG_{kind}_HOME" if kind != "RUNTIME" else "XDG_RUNTIME_DIR"] = str(path)
        transcriber = Transcriber()
        try:
            started = time.monotonic()
            transcriber.load(config)
            loaded = time.monotonic()
            result = transcribe_file(snapshot, config, transcriber=transcriber, max_seconds=600)
            finished = time.monotonic()
        finally:
            close = getattr(transcriber.model, "close", None)
            if callable(close):
                close()
    return {
        "version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "engine": original.engine,
        "model": model_identity(original.model),
        "revision": effective_revision(original.model, original.model_revision, original.engine)
        if original.engine in ("nemo", "transformers")
        else original.model_revision,
        "device": original.device,
        "compute_type": original.compute_type,
        "settings_scope": "Requested device/precision; auto/default are backend-selected",
        "language": config.language,
        "decode_settings": decode_settings(config),
        "artifact": identity,
        "fixture_sha256": fixture_hash,
        "audio_seconds": result["duration"],
        "load_seconds": loaded - started,
        "decode_seconds": finished - loaded,
        "real_time_factor": (finished - loaded) / result["duration"],
        "decode_scope": "File decoding, VAD and inference; excludes cold model loading",
        "accuracy": accuracy(request["reference"], result["text"])
        if request.get("reference") is not None
        else None,
        "hardware": hardware_context(),
        "software": software,
        "network": "AF_INET/AF_INET6 denied by seccomp; offline libraries; existing weights only",
        "scope": SCOPE,
    }


def worker_main() -> None:
    # Keep library diagnostics separate from the bounded JSON result.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    sys.stdout = sys.stderr
    os.dup2(sys.stderr.fileno(), 1)
    try:
        from .native_worker import isolate_network  # noqa: PLC0415 - load benchmark dependencies only for this operation

        isolate_network()  # Fail closed before any model/library loading.
        raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Benchmark request too large")
        response: dict[str, object] = {"result": _measure(json.loads(raw))}
    # The worker returns failure through its private response channel.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        response = {"error": str(exc)[:1000]}
    output.write(json.dumps(response, allow_nan=False).encode())
    output.close()


if __name__ == "__main__":
    worker_main()
