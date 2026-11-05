from pathlib import Path

from .config import Config, cache_dir


def download(model: str):
    from faster_whisper.utils import download_model

    return download_model(model, cache_dir=str(cache_dir() / "models"))


class Transcriber:
    def __init__(self):
        self.key = None
        self.model = None
        self.stream_key = None
        self.cursor = 0
        self.committed = []
        self.pending = ""
        self.covered = 0

    def load(self, config: Config):
        key = (config.model, config.device, config.compute_type)
        if key != self.key:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                config.model,
                device=config.device,
                compute_type=config.compute_type,
                download_root=str(cache_dir() / "models"),
                local_files_only=True,
            )
            self.key = key

    def _session(self, path, config):
        key = (path, config.model, config.device, config.compute_type, config.language)
        if key != self.stream_key:
            self.stream_key = key
            self.cursor = 0
            self.committed = []
            self.pending = ""
            self.covered = 0

    def preview(self, path: Path, config: Config):
        """Keep completed segments; revise only the uncommitted audio prefix."""
        from .audio import read_audio_from

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
            cursor = self.cursor + boundary
            count += 1
        self.cursor = cursor
        self.pending = " ".join(s.text.strip() for s in segments[count:]).strip()
        self.covered = end
        return " ".join([*self.committed, self.pending]).strip()

    def transcribe(self, path: Path, config: Config):
        """Finalize a live session, or decode a standalone file from its beginning."""
        from .audio import read_audio_from

        key = (path, config.model, config.device, config.compute_type, config.language)
        try:
            if key != self.stream_key:
                return self._decode(str(path), config)
            audio = read_audio_from(path, self.cursor)
            if self.cursor + len(audio) == self.covered:
                tail = self.pending
            else:
                tail = self._decode(audio, config) if len(audio) else ""
            return " ".join([*self.committed, tail]).strip()
        finally:
            self.stream_key = None
            self.committed = []
            self.pending = ""

    def _segments(self, audio, config):
        self.load(config)
        segments, _ = self.model.transcribe(
            audio,
            language=None if config.language == "auto" else config.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return list(segments)

    def _decode(self, audio, config):
        return " ".join(segment.text.strip() for segment in self._segments(audio, config)).strip()
