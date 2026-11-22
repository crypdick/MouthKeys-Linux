from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .audio import read_audio_from
from .config import Config, cache_dir
from .model_progress import DownloadProgress, progress_tqdm
from .native_asr import download_native, effective_revision, uses_native_streaming
from .transcript_window import Segment, TimedWord, TranscriptWindow, decode_transcript, timed_words

StreamKey = tuple[Path | str | int, ...]


def download(
    model: str,
    engine: str = "whisper",
    compute_type: str = "default",
    *,
    revision: str = "main",
    progress: Callable[[DownloadProgress], None] | None = None,
) -> str:
    if progress:
        progress({"phase": "preparing", "completed": 0, "total": None, "unit": "file"})
    if engine == "onnx":
        from .engines import download_onnx

        path: str = download_onnx(model, compute_type, progress=progress, revision=revision)
        return path
    if engine in ("transformers", "nemo"):
        return download_native(model, engine, revision, progress)
    if engine != "whisper":
        raise ValueError("Unknown engine")
    from faster_whisper.utils import download_model

    if progress:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import snapshot_download

        repository = model if "/" in model else _MODELS.get(model)
        if not repository:
            raise ValueError("Unknown Whisper model")
        return snapshot_download(
            repository,
            cache_dir=str(cache_dir() / "models"),
            revision=revision,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
            ],
            tqdm_class=progress_tqdm(progress),
        )
    downloaded: str = download_model(
        model, cache_dir=str(cache_dir() / "models"), revision=revision
    )
    return downloaded


def _model_for_config(config: Config) -> object:
    # Keep heavyweight engines unloaded until the first request.
    if config.engine == "onnx":
        from .engines import load_onnx

        return load_onnx(config)
    if config.engine in ("transformers", "nemo"):
        from .native_worker import NativeWorkerBackend

        return NativeWorkerBackend(config)
    if config.engine != "whisper":
        raise ValueError("Unknown speech engine")
    from faster_whisper import WhisperModel

    from .engines import resolve_whisper

    return WhisperModel(
        resolve_whisper(config),
        device=config.device,
        compute_type=config.compute_type,
        download_root=str(cache_dir() / "models"),
        local_files_only=True,
        revision=config.model_revision,
    )


def _model_alive(model: object) -> bool:
    try:
        probe = getattr(model, "alive", None)
        return not callable(probe) or bool(probe())
    except (OSError, RuntimeError):
        return False


class Transcriber:
    def __init__(self) -> None:
        self.key: tuple[str, ...] | None = None
        self.model: Any = None
        self._loaded_model: dict[str, str | bool] | None = None
        self.stream_key: StreamKey | None = None
        self._stream_epoch = 0
        self._preview: TranscriptWindow | None = None
        self.words: list[TimedWord] = []
        self.last_alignment: dict[str, object] | None = None
        self.aligner: Any = None
        self.alignment_warning = ""

    def load(self, config: Config) -> None:
        if self.aligner is not None and (
            config.pronunciation_matching != "on" or not config.pronunciation_aligner_model
        ):
            self.aligner.clear()
        key = (
            config.model,
            config.device,
            config.compute_type,
            config.engine,
            config.model_revision,
            config.native_streaming,
            config.streaming_latency,
        )
        if key == self.key and _model_alive(self.model):
            return
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
        self.model = None
        self.key = None
        self._loaded_model = None
        self.model = _model_for_config(config)
        self.key = key
        revision = config.model_revision
        if config.engine in ("nemo", "transformers"):
            revision = effective_revision(config.model, revision, config.engine)
        self._loaded_model = {
            "engine": config.engine,
            "model": config.model,
            "revision": revision,
            "device": config.device,
            "compute_type": config.compute_type,
            "native_streaming": bool(
                uses_native_streaming(config) and getattr(self.model, "native_streaming", False)
            ),
        }

    @property
    def loaded_model(self) -> dict[str, str | bool] | None:
        """Nonblocking resident identity; a dead isolated worker has no loaded model."""
        model, details = self.model, self._loaded_model
        if model is None or details is None or not _model_alive(model):
            return None
        if model is not self.model or details is not self._loaded_model:
            return None
        return dict(details)

    def _forget_preview(self) -> None:
        self.stream_key = None
        self._preview = None

    @property
    def committed_segments(self) -> int:
        return self._preview.committed_segments if self._preview is not None else 0

    def _stream_key(self, path: Path | str, config: Config) -> StreamKey:
        return (
            path,
            config.model,
            config.device,
            config.compute_type,
            config.language,
            config.engine,
            config.model_revision,
            config.vocabulary_boosting,
            config.asr_max_tokens,
            config.native_streaming,
            config.streaming_latency,
            config.pronunciation_matching,
            config.pronunciation_aligner_model,
            self._stream_epoch,
        )

    def _align_segments(
        self, audio: NDArray[np.generic], config: Config, segments: list[Segment]
    ) -> list[Segment]:
        if config.pronunciation_matching != "on" or not config.pronunciation_aligner_model:
            return segments
        if self.aligner is None:
            from .forced_alignment import WhisperForcedAligner

            self.aligner = WhisperForcedAligner()
        aligned, warning = self.aligner.align(audio, segments, config)
        if warning and warning not in self.alignment_warning:
            self.alignment_warning = "; ".join(filter(None, [self.alignment_warning, warning]))
        result: list[Segment] = aligned
        return result

    def _record_alignment(self, path: Path, text: str, words: list[TimedWord]) -> None:
        self.words = words
        self.last_alignment = {"text": text, "words": words, "source": str(path)}
        if self.alignment_warning:
            self.last_alignment["warning"] = self.alignment_warning

    def _uses_native_streaming(self, config: Config) -> bool:
        if not uses_native_streaming(config):
            return False
        self.load(config)
        return bool(getattr(self.model, "native_streaming", True))

    def preview(self, path: Path, config: Config) -> str:
        """Keep completed segments; revise only the uncommitted audio prefix."""
        key = self._stream_key(path, config)
        if key != self.stream_key:
            self._forget_preview()
            self.alignment_warning = ""
        self.stream_key = key
        if self._uses_native_streaming(config):
            self._preview = None
            text: str = self.model.stream(read_audio_from(path), config, key)
            return text
        if self._preview is None:
            self.words = []
            self.last_alignment = None
            self._preview = TranscriptWindow()
        return self._preview.preview(path, lambda audio: self.segments(audio, config))

    def transcribe(self, path: str | Path, config: Config) -> str:
        """Finalize a live session, or decode a standalone file from its beginning."""
        path = Path(path)
        key = self._stream_key(path, config)
        self.words = []
        self.last_alignment = None
        if key != self.stream_key:
            self.alignment_warning = ""
        try:
            if self._uses_native_streaming(config):
                return self._finish_native(path, config, key)
            if self._preview is None or key != self.stream_key:
                result = decode_transcript(self.segments(str(path), config))
            else:
                result = self._preview.finish(path, lambda audio: self.segments(audio, config))
            if config.pronunciation_matching == "on":
                self._record_alignment(path, result.text, result.words)
            return result.text
        finally:
            self._forget_preview()

    def _finish_native(self, path: Path, config: Config, key: StreamKey) -> str:
        audio = read_audio_from(path)
        text: str = self.model.stream(audio, config, key, final=True)
        if config.pronunciation_matching == "on":
            words = getattr(self.model, "words", [])
            if not words:
                segments = self._align_segments(
                    audio, config, [SimpleNamespace(start=0, end=len(audio) / 16000, text=text)]
                )
                words = timed_words(segments)
            self._record_alignment(path, text, words)
        return text

    def reset_stream(self) -> None:
        """Release recording caches on cancel under the controller's model lock."""
        self._forget_preview()
        self._stream_epoch += 1
        self.words = []
        self.last_alignment = None
        self.alignment_warning = ""
        reset = getattr(self.model, "reset_stream", None)
        if callable(reset):
            reset()

    def segments(self, audio: str | NDArray[np.generic], config: Config) -> list[Segment]:
        self.load(config)
        if config.engine == "whisper":
            from .vocabulary import whisper_options

            segments, _ = self.model.transcribe(
                audio,
                language=None if config.language == "auto" else config.language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=config.pronunciation_matching == "on",
                **whisper_options(config),
            )
            return list(segments)

        samples: NDArray[np.generic]
        if isinstance(audio, str):
            from faster_whisper.audio import decode_audio

            samples = decode_audio(audio, sampling_rate=16000)
        else:
            samples = audio
        if config.engine == "onnx":
            segments = self.model.recognize(
                samples,
                sample_rate=16000,
                language=None if config.language == "auto" else config.language,
            )
        else:
            segments = self.model.segments(samples, config)
        return self._align_segments(samples, config, list(segments))
