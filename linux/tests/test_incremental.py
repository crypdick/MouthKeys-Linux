import wave
from dataclasses import replace
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
    monkeypatch.setattr("mouthkeys.engines.resolve_whisper", lambda config: config.model)
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


def test_preview_waits_for_audio_and_does_not_redo_unchanged_pcm(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 0.25)
    assert transcriber.preview(path, Config()) == ""
    assert transcriber.committed_segments == 0
    audio(path, 5)
    model.results = [[segment(0, 2, "Stable."), segment(2, 5, "Pending.")]]
    assert transcriber.preview(path, Config()) == "Stable. Pending."
    assert transcriber.committed_segments == 1
    assert transcriber.preview(path, Config()) == "Stable. Pending."
    assert model.lengths == [5]


def test_trimmed_recording_does_not_reuse_removed_preview_tail(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 5)
    model.results = [[segment(0, 2, "Stable."), segment(2, 5, "Pending.")]]
    assert transcriber.preview(path, Config()) == "Stable. Pending."
    audio(path, 2)
    assert transcriber.transcribe(path, Config()) == "Stable."
    assert model.lengths == [5]


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


@pytest.mark.parametrize(
    "change",
    [
        {"vocabulary_boosting": "on"},
        {"asr_max_tokens": 256},
        {"native_streaming": "off"},
        {"streaming_latency": "80"},
    ],
)
def test_changed_decode_settings_do_not_reuse_old_preview(engine, tmp_path, change):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 5)
    model.results = [[segment(0, 2, "Old preview.")]]
    transcriber.preview(path, Config())
    model.results = [[segment(0, 4, "Fresh transcript.")]]
    assert transcriber.transcribe(path, replace(Config(), **change)) == "Fresh transcript."
    assert model.lengths == [5, 5]


def test_failed_preview_update_does_not_partially_commit_segments(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 8)
    model.results = [[segment(0, 2, "First."), segment(2, float("nan"), "Invalid boundary.")]]
    with pytest.raises(ValueError, match="NaN"):
        transcriber.preview(path, Config())
    model.results = [[segment(0, 2, "First."), segment(2, 7, "Then second.")]]
    assert transcriber.preview(path, Config()) == "First. Then second."
    assert transcriber.transcribe(path, Config()) == "First. Then second."
    assert model.lengths == [8, 8]


def test_failed_backend_reset_still_discards_preview(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "recording.wav"
    audio(path, 5)
    model.results = [[segment(0, 2, "Cancelled.")]]
    transcriber.preview(path, Config())

    def failed_reset():
        raise RuntimeError("Worker lost")

    model.reset_stream = failed_reset
    with pytest.raises(RuntimeError, match="Worker lost"):
        transcriber.reset_stream()
    assert transcriber.stream_key is None
    model.results = [[segment(0, 4, "Fresh transcript.")]]
    assert transcriber.transcribe(path, Config()) == "Fresh transcript."


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


def timed_segment(start, end, text):
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=[SimpleNamespace(word=text, start=start, end=end, probability=0.8)],
    )


def test_final_word_alignment_retains_committed_and_offsets_fresh_tail(engine, tmp_path):
    transcriber, model = engine
    config = replace(Config(), pronunciation_matching="on")
    path = tmp_path / "words.wav"
    audio(path, 10)
    model.results = [[timed_segment(0, 5, " First."), timed_segment(5, 9, " Wrong.")]]
    transcriber.preview(path, config)
    audio(path, 12)
    model.results = [[timed_segment(0, 6, " Correct.")]]
    assert transcriber.transcribe(path, config) == "First. Correct."
    assert model.lengths == [10, 7]
    assert [(w["word"], w["start"], w["end"]) for w in transcriber.words] == [
        (" First.", 0, 5),
        (" Correct.", 5, 11),
    ]
    assert transcriber.last_alignment["source"] == str(path)
    assert transcriber.last_alignment["text"] == "First. Correct."


def test_alignment_discards_invalid_word_timing_and_nonfinite_confidence(engine, tmp_path):
    transcriber, model = engine
    path = tmp_path / "words.wav"
    audio(path, 3)
    words = [
        SimpleNamespace(word="First", start=0, end=1, probability=float("nan")),
        SimpleNamespace(word="invalid", start=2, end=1),
        SimpleNamespace(word="invalid", start=float("inf"), end=2),
        SimpleNamespace(word="invalid", start=1, end=float("inf")),
        SimpleNamespace(word="second", start=1, end=2),
    ]
    model.results = [[SimpleNamespace(start=0, end=2, text="First second", words=words)]]
    assert transcriber.transcribe(path, Config(pronunciation_matching="on")) == "First second"
    assert transcriber.words == [
        {"word": "First", "start": 0, "end": 1},
        {"word": "second", "start": 1, "end": 2},
    ]


@pytest.mark.parametrize("native_words", [False, True])
def test_native_final_keeps_words_or_reports_optional_alignment_failure(
    monkeypatch, tmp_path, native_words
):
    transcriber = Transcriber()
    path = tmp_path / "recording.wav"
    audio(path, 1)
    timing = {"word": "Native", "start": 0.0, "end": 1.0}
    model = SimpleNamespace(stream=lambda *_args, **_options: "Native")
    if native_words:
        model.words = [timing]
    monkeypatch.setattr("mouthkeys.native_worker.NativeWorkerBackend", lambda _: model)
    config = Config(
        engine="nemo", pronunciation_matching="on", pronunciation_aligner_model=str(tmp_path)
    )
    assert transcriber.transcribe(path, config) == "Native"
    assert transcriber.last_alignment["text"] == "Native"
    if native_words:
        assert transcriber.words == [timing]
        assert "warning" not in transcriber.last_alignment
    else:
        assert transcriber.words == []
        assert "ASR text retained" in transcriber.last_alignment["warning"]


def test_final_word_alignment_reuses_pending_and_clears_on_reset(engine, tmp_path):
    transcriber, model = engine
    config = replace(Config(), pronunciation_matching="on")
    path = tmp_path / "words.wav"
    audio(path, 3)
    model.results = [[timed_segment(0, 2.5, " Word.")]]
    transcriber.preview(path, config)
    assert transcriber.transcribe(path, config) == "Word."
    assert len(transcriber.words) == 1
    assert model.lengths == [3]
    transcriber.reset_stream()
    assert transcriber.words == []
    assert transcriber.last_alignment is None
