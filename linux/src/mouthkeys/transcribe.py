import math
from pathlib import Path
from types import SimpleNamespace

from .config import Config, cache_dir
from .native_asr import uses_native_streaming


def download(
    model: str, engine="whisper", compute_type="default", *, revision="main", progress=None
):
    if progress:
        progress({"phase": "preparing", "completed": 0, "total": None, "unit": "file"})
    if engine == "onnx":
        from .engines import download_onnx

        return download_onnx(model, compute_type, progress=progress, revision=revision)
    if engine in ("transformers", "nemo"):
        from .native_asr import download_native

        return download_native(model, engine, revision, progress)
    if engine != "whisper":
        raise ValueError("Unknown engine")
    from faster_whisper.utils import download_model

    if progress:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import snapshot_download

        from .model_progress import progress_tqdm

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
    return download_model(model, cache_dir=str(cache_dir() / "models"), revision=revision)


class Transcriber:
    def __init__(self):
        self.key = None
        self.model = None
        self._loaded_model = None
        self.stream_key = None
        self.cursor = 0
        self.committed = []
        self.pending = ""
        self.covered = 0
        self.words = []
        self.last_alignment = None
        self.committed_words = []
        self.pending_words = []
        self.decoded_words = []
        self.aligner = None
        self.alignment_warning = ""

    def load(self, config: Config):
        if self.aligner is not None and (
            config.pronunciation_matching != "on" or not config.pronunciation_aligner_model
        ):
            self.aligner.clear()
        key = (
            config.model,
            config.device,
            config.compute_type,
            getattr(config, "engine", "whisper"),
            getattr(config, "model_revision", "main"),
            getattr(config, "native_streaming", "on"),
            getattr(config, "streaming_latency", "model"),
        )
        alive = not hasattr(self.model, "alive") or self.model.alive()
        if key != self.key or not alive:
            if hasattr(self.model, "close"):
                self.model.close()
            self.model = None
            self.key = None
            self._loaded_model = None
            if getattr(config, "engine", "whisper") == "onnx":
                from .engines import load_onnx

                self.model = load_onnx(config)
                self.key = key
                self._loaded(config)
                return
            if getattr(config, "engine", "whisper") in ("transformers", "nemo"):
                from .native_worker import NativeWorkerBackend

                self.model = NativeWorkerBackend(config)
                self.key = key
                self._loaded(config)
                return
            if getattr(config, "engine", "whisper") != "whisper":
                raise ValueError("Unknown speech engine")
            from faster_whisper import WhisperModel

            from .engines import resolve_whisper

            self.model = WhisperModel(
                resolve_whisper(config),
                device=config.device,
                compute_type=config.compute_type,
                download_root=str(cache_dir() / "models"),
                local_files_only=True,
                revision=getattr(config, "model_revision", "main"),
            )
            self.key = key
            self._loaded(config)

    def _loaded(self, config):
        revision = getattr(config, "model_revision", "main")
        if config.engine in ("nemo", "transformers"):
            from .native_asr import effective_revision

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
    def loaded_model(self):
        """Nonblocking resident identity; a dead isolated worker has no loaded model."""
        model = self.model
        if model is None:
            return None
        try:
            if hasattr(model, "alive") and not model.alive():
                return None
        except (OSError, RuntimeError):
            return None
        return dict(self._loaded_model) if self._loaded_model is not None else None

    def _session(self, path, config):
        key = self._stream_key(path, config)
        if key != self.stream_key:
            self.stream_key = key
            self.cursor = 0
            self.committed = []
            self.pending = ""
            self.covered = 0
            self.words = []
            self.last_alignment = None
            self.committed_words = []
            self.pending_words = []
            self.alignment_warning = ""

    @staticmethod
    def _stream_key(path, config):
        return (
            path,
            config.model,
            config.device,
            config.compute_type,
            config.language,
            getattr(config, "engine", "whisper"),
            getattr(config, "model_revision", "main"),
            getattr(config, "pronunciation_matching", "off"),
            getattr(config, "pronunciation_aligner_model", ""),
        )

    def _align_segments(self, audio, config, segments):
        if config.pronunciation_matching != "on" or not config.pronunciation_aligner_model:
            if self.aligner is not None:
                self.aligner.clear()
            return segments
        if self.aligner is None:
            from .forced_alignment import WhisperForcedAligner

            self.aligner = WhisperForcedAligner()
        aligned, warning = self.aligner.align(audio, segments, config)
        if warning and warning not in self.alignment_warning:
            self.alignment_warning = "; ".join(filter(None, [self.alignment_warning, warning]))
        return aligned

    def _record_alignment(self, path, text, words):
        self.words = words
        self.last_alignment = {"text": text, "words": words, "source": str(path)}
        if self.alignment_warning:
            self.last_alignment["warning"] = self.alignment_warning

    def _uses_native_streaming(self, config):
        if not uses_native_streaming(config):
            return False
        self.load(config)
        return getattr(self.model, "native_streaming", True)

    def preview(self, path: Path, config: Config):
        """Keep completed segments; revise only the uncommitted audio prefix."""
        from .audio import read_audio_from

        if self._uses_native_streaming(config):
            if self.stream_key != self._stream_key(path, config):
                self.alignment_warning = ""
            self.stream_key = self._stream_key(path, config)
            return self.model.stream(read_audio_from(path), config, self.stream_key)
        self._session(path, config)
        # Grow an unresolved window rather than freezing at its original end.
        unresolved = (self.covered - self.cursor) / 16000
        seconds = unresolved + 30 if unresolved >= 30 else 30
        audio = read_audio_from(path, self.cursor, seconds=seconds)
        end = self.cursor + len(audio)
        if end == self.covered or len(audio) < 8000:
            return " ".join([*self.committed, self.pending]).strip()
        segments = self._segments(audio, config)
        # NOTE: segment boundaries with 2 s lookahead trade accuracy for latency.
        # Word agreement could refine boundaries if real dictation exposes clipping.
        cutoff = len(audio) / 16000 - 2
        count = 0
        cursor = self.cursor
        for segment in segments:
            boundary = round(segment.end * 16000)
            if not 0 < boundary <= round(cutoff * 16000):
                break
            self.committed.append(segment.text.strip())
            self.committed_words.extend(self._word_metadata([segment], self.cursor / 16000))
            cursor = self.cursor + boundary
            count += 1
        self.pending_words = self._word_metadata(segments[count:], self.cursor / 16000)
        self.cursor = cursor
        self.pending = " ".join(s.text.strip() for s in segments[count:]).strip()
        self.covered = end
        return " ".join([*self.committed, self.pending]).strip()

    def transcribe(self, path: Path, config: Config):
        """Finalize a live session, or decode a standalone file from its beginning."""
        from .audio import read_audio_from

        key = self._stream_key(path, config)
        self.words = []
        self.last_alignment = None
        if key != self.stream_key:
            self.alignment_warning = ""
        try:
            if self._uses_native_streaming(config):
                audio = read_audio_from(path)
                text = self.model.stream(audio, config, key, final=True)
                if config.pronunciation_matching == "on":
                    words = getattr(self.model, "words", [])
                    if not words:
                        segments = self._align_segments(
                            audio,
                            config,
                            [SimpleNamespace(start=0, end=len(audio) / 16000, text=text)],
                        )
                        words = self._word_metadata(segments)
                    self._record_alignment(path, text, words)
                return text
            if key != self.stream_key:
                text = self._decode(str(path), config)
                words = self.decoded_words
            else:
                audio = read_audio_from(path, self.cursor)
                if self.cursor + len(audio) == self.covered:
                    tail, tail_words = self.pending, self.pending_words
                elif len(audio):
                    tail = self._decode(audio, config, offset=self.cursor / 16000)
                    tail_words = self.decoded_words
                else:
                    tail, tail_words = "", []
                text = " ".join([*self.committed, tail]).strip()
                words = [*self.committed_words, *tail_words]
            if config.pronunciation_matching == "on":
                self._record_alignment(path, text, words)
            return text
        finally:
            self.stream_key = None
            self.committed = []
            self.pending = ""
            self.committed_words = []
            self.pending_words = []

    def reset_stream(self):
        """Release recording caches on cancel under the controller's model lock."""
        if hasattr(self.model, "reset_stream"):
            self.model.reset_stream()
        self.stream_key = None
        self.committed = []
        self.pending = ""
        self.words = []
        self.last_alignment = None
        self.alignment_warning = ""
        self.committed_words = []
        self.pending_words = []

    def _segments(self, audio, config):
        self.load(config)
        if getattr(config, "engine", "whisper") in ("transformers", "nemo"):
            if isinstance(audio, str):
                from faster_whisper.audio import decode_audio

                audio = decode_audio(audio, sampling_rate=16000)
            return self._align_segments(audio, config, list(self.model.segments(audio, config)))
        if getattr(config, "engine", "whisper") == "onnx":
            if isinstance(audio, str):
                from faster_whisper.audio import decode_audio

                audio = decode_audio(audio, sampling_rate=16000)
            segments = list(
                self.model.recognize(
                    audio,
                    sample_rate=16000,
                    language=None if config.language == "auto" else config.language,
                )
            )
            return self._align_segments(audio, config, segments)
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

    @staticmethod
    def _word_metadata(segments, offset=0):
        words = []
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                start, end = float(word.start), float(word.end)
                if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end:
                    continue
                entry = {"word": word.word, "start": start + offset, "end": end + offset}
                probability = getattr(word, "probability", None)
                if probability is not None and math.isfinite(float(probability)):
                    entry["probability"] = float(probability)
                if getattr(word, "alignment_source", None):
                    entry["alignment_source"] = word.alignment_source
                words.append(entry)
        return words

    def _decode(self, audio, config, *, offset=0):
        self.decoded_words = []
        segments = self._segments(audio, config)
        self.decoded_words = self._word_metadata(segments, offset)
        return " ".join(segment.text.strip() for segment in segments).strip()
