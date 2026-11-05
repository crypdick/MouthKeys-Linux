import wave
from types import SimpleNamespace

import pytest

from mouthkeys.config import Config
from mouthkeys.transcribe import Transcriber


def audio(path, seconds):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(int(seconds * 16000) * 2))


def segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


@pytest.fixture
def engine(monkeypatch):
    import faster_whisper

    class Model:
        def __init__(self):
            self.results = []
            self.lengths = []

        def transcribe(self, samples, **kwargs):
            if isinstance(samples, str):
                with wave.open(samples) as wav:
                    duration = wav.getnframes() / wav.getframerate()
            else:
                duration = len(samples) / 16000
            self.lengths.append(duration)
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return iter(result), None

    model = Model()
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **kw: model)
    return Transcriber(), model


def test_stop_reuses_complete_preview(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 4)
    model.results = [[segment(0, 3.5, " Final words.")]]
    assert transcriber.preview(path, Config()) == "Final words."
    assert transcriber.transcribe(path, Config()) == "Final words."
    assert model.lengths == [4]


def test_completed_prefix_survives_long_recording_and_stop_decodes_only_tail(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 20)
    model.results = [[segment(0, 12, " First sentence."), segment(12, 19, " Uncertain")]]
    assert transcriber.preview(path, Config()) == "First sentence. Uncertain"
    audio(path, 26)
    model.results = [[segment(0, 8, " Corrected sentence."), segment(8, 14, " Last words.")]]
    assert (
        transcriber.transcribe(path, Config()) == "First sentence. Corrected sentence. Last words."
    )
    assert model.lengths == [20, 14]


def test_preview_never_skips_backlog(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 65)
    model.results = [[segment(0, 25, " Beginning."), segment(25, 30, " Middle.")]]
    assert transcriber.preview(path, Config()) == "Beginning. Middle."
    model.results = [[segment(0, 25, " Middle."), segment(25, 30, " Later.")]]
    assert transcriber.preview(path, Config()) == "Beginning. Middle. Later."
    model.results = [[segment(0, 15, " Ending.")]]
    assert transcriber.transcribe(path, Config()) == "Beginning. Middle. Ending."
    assert model.lengths == [30, 30, 15]


def test_preview_failure_keeps_prefix_and_retries_tail(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 8)
    model.results = [[segment(0, 4, " Kept."), segment(4, 8, " Provisional")]]
    transcriber.preview(path, Config())
    audio(path, 10)
    model.results = [RuntimeError("preview failed")]
    with pytest.raises(RuntimeError, match="preview failed"):
        transcriber.preview(path, Config())
    model.results = [[segment(0, 6, " Recovered.")]]
    assert transcriber.transcribe(path, Config()) == "Kept. Recovered."
    assert model.lengths == [8, 6, 6]


@pytest.mark.parametrize("change", ["path", "language"])
def test_new_recording_or_language_does_not_reuse_old_preview(engine, tmp_path, change):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 4)
    model.results = [[segment(0, 1, " Old.")]]
    transcriber.preview(path, Config())
    config = Config(language="fr") if change == "language" else Config()
    if change == "path":
        path = tmp_path / "retry.wav"
        audio(path, 4)
    model.results = [[segment(0, 3, " New.")]]
    assert transcriber.transcribe(path, config) == "New."
    assert model.lengths == [4, 4]


def test_final_without_preview_and_silence(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 5)
    model.results = [[]]
    assert transcriber.transcribe(path, Config()) == ""
    model.results = [[segment(0, 4, " Fresh.")]]
    assert transcriber.transcribe(path, Config()) == "Fresh."
    assert model.lengths == [5, 5]


@pytest.mark.parametrize("first", [[], [segment(0, 30, " Long sentence.")]])
def test_unresolved_window_grows_instead_of_freezing_preview(engine, tmp_path, first):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 30)
    model.results = [first]
    transcriber.preview(path, Config())
    audio(path, 40)
    model.results = [[segment(0, 35, " Speech after silence."), segment(35, 39, " Tail.")]]
    assert transcriber.preview(path, Config()) == "Speech after silence. Tail."
    assert transcriber.transcribe(path, Config()) == "Speech after silence. Tail."
    assert model.lengths == [30, 40]


def test_standalone_stereo_file_keeps_decoder_format_support(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(bytes(44100 * 4))
    model.results = [[segment(0, 1, " Supported.")]]
    assert transcriber.transcribe(path, Config()) == "Supported."
    assert model.lengths == [1]


def test_unfinished_wav_reads_from_committed_frame(tmp_path):
    import struct

    import numpy as np

    from mouthkeys.audio import read_audio_from

    path = tmp_path / "live.wav"
    audio(path, 3)
    with path.open("r+b") as file:
        file.seek(4)
        file.write(struct.pack("<I", 0xFFFFFFFF))
        file.seek(40)
        file.write(struct.pack("<I", 0xFFFFFFFF))
    samples = read_audio_from(path, 16000, seconds=1)
    assert len(samples) == 16000
    np.testing.assert_array_equal(samples, np.zeros(16000))
