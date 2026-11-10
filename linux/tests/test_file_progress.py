import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, QProcess, Signal
from PySide6.QtWidgets import QApplication, QPushButton

from mouthkeys import cli, file_transcription, gui
from mouthkeys.config import Config
from mouthkeys.file_history import FileHistory


def result():
    return {
        "text": "Final words.",
        "segments": [{"start": 0, "end": 1, "text": "Final words."}],
        "speaker_turns": [],
        "duration": 1,
        "processing_seconds": 0.5,
    }


def test_file_progress_counts_real_turns_and_uncovered_chunks(monkeypatch):
    from faster_whisper import audio

    events = []

    def decode_audio(*args, **kwargs):
        assert events[-1] == {"stage": "decode"}
        return np.zeros(5 * 16000)

    def diarize(*args):
        assert events[-1] == {"stage": "diarization"}
        return [{"start": 1, "end": 2, "speaker": 0}, {"start": 3, "end": 4, "speaker": 1}]

    sizes = []

    def segments(samples, config):
        index = len(sizes)
        assert events[-1] == {
            "stage": "transcribe",
            "completed": index,
            "total": 5,
            "kind": "turn" if index in (1, 3) else "uncovered",
        }
        sizes.append(len(samples))
        return [SimpleNamespace(start=0, end=1, text=f"Part {index}.")]

    monkeypatch.setattr(audio, "decode_audio", decode_audio)
    monkeypatch.setattr(file_transcription, "diarize", diarize)
    output = file_transcription.transcribe_file(
        "example.wav",
        replace(Config(), file_speaker_labels="on"),
        transcriber=SimpleNamespace(_segments=segments),
        on_progress=events.append,
    )
    assert sizes == [16000] * 5
    assert [part.get("speaker") for part in output["segments"]] == [None, 0, None, 1, None]
    assert events[-2:] == [
        {"stage": "transcribe", "completed": 5, "total": 5},
        {"stage": "finalize"},
    ]


def test_whole_file_progress_stays_indeterminate_while_backend_blocks(monkeypatch):
    from faster_whisper import audio

    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: np.zeros(16000))
    events = []

    def segments(*args):
        assert events == [{"stage": "decode"}, {"stage": "transcribe"}]
        return []

    file_transcription.transcribe_file(
        "example.wav",
        Config(),
        transcriber=SimpleNamespace(_segments=segments),
        on_progress=events.append,
    )
    assert events[-1] == {"stage": "finalize"}
    assert not any("total" in event for event in events)


def test_cli_progress_is_opt_in_and_never_changes_stdout(monkeypatch, tmp_path, capsys):
    calls = []

    def transcribe(path, config, **options):
        calls.append(options)
        if "on_progress" in options:
            options["on_progress"]({"stage": "decode"})
            options["on_progress"]({"stage": "transcribe", "completed": 1, "total": 2})
        return result()

    monkeypatch.setattr(file_transcription, "transcribe_file", transcribe)
    assert cli.main(["transcribe", "example.wav", "--json"]) == 0
    plain = capsys.readouterr()
    assert json.loads(plain.out) == result()
    assert plain.err == ""
    assert calls == [{}]
    assert cli.main(["transcribe", "example.wav", "--json", "--progress"]) == 0
    streamed = capsys.readouterr()
    assert streamed.out == plain.out
    events = [
        json.loads(line.removeprefix("transcription_progress "))
        for line in streamed.err.splitlines()
    ]
    assert events == [
        {"stage": "decode"},
        {"stage": "transcribe", "completed": 1, "total": 2},
        {"stage": "complete"},
    ]
    output = tmp_path / "transcript.txt"
    assert (
        cli.main(
            ["transcribe", "example.wav", "--progress", "--save-history", "--output", str(output)]
        )
        == 0
    )
    exported = capsys.readouterr()
    assert exported.out == ""
    assert output.read_text() == "Final words.\n"
    stages = [json.loads(line.split(" ", 1)[1])["stage"] for line in exported.err.splitlines()]
    assert stages == ["decode", "transcribe", "save", "export", "complete"]
    assert FileHistory().rows()[0]["result"] == result()


class FakeFileProcess(QObject):
    finished = Signal(int, object)
    errorOccurred = Signal(object)
    readyReadStandardError = Signal()
    ProcessError = QProcess.ProcessError

    def __init__(self, parent):
        super().__init__(parent)
        self.stderr = b""
        self.killed = False

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = arguments

    def start(self):
        pass

    def readAllStandardError(self):
        data, self.stderr = self.stderr, b""
        return data

    def readAllStandardOutput(self):
        return json.dumps(result()).encode()

    def kill(self):
        self.killed = True


def window(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    monkeypatch.setattr(gui.Window, "save", lambda self: True)
    monkeypatch.setattr(gui, "QProcess", FakeFileProcess)
    widget = gui.Window()
    widget.timer.stop()
    return widget


def test_gui_reads_live_fragmented_progress_and_copies_only_on_explicit_click(
    monkeypatch, tmp_path
):
    widget = window(monkeypatch)
    copied = []
    monkeypatch.setattr(QApplication, "clipboard", lambda: SimpleNamespace(setText=copied.append))
    source = tmp_path / "source.wav"
    source.touch()
    widget.file_path.setText(str(source))
    widget.transcribe_file()
    process = widget.file_process
    assert process.arguments[-2:] == ["--json", "--progress"]
    assert widget.file_progress.maximum() == 0
    process.stderr = b'transcription_progress {"stage":"transcribe","completed":1,'
    process.readyReadStandardError.emit()
    assert widget.file_progress.maximum() == 0
    process.stderr = b'"total":4,"kind":"uncovered"}\n'
    process.readyReadStandardError.emit()
    assert widget.file_progress.maximum() == 4
    assert widget.file_progress.value() == 1
    assert "uncovered audio" in widget.file_progress_label.text()
    assert copied == []
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert widget.file_result == result()
    assert widget.file_text.toPlainText() == "Final words."
    assert widget.file_progress.format() == "Complete"
    assert FileHistory().rows()[0]["result"] == result()
    assert copied == []
    button = next(
        button
        for button in widget.file_path.parentWidget().findChildren(QPushButton)
        if button.text() == "Copy result"
    )
    button.click()
    assert copied == ["Final words."]
    assert widget.file_process is None


def test_gui_progress_bounds_stderr_and_rejects_invalid_counters(monkeypatch, tmp_path):
    widget = window(monkeypatch)
    source = tmp_path / "source.wav"
    source.touch()
    widget.file_path.setText(str(source))
    widget.transcribe_file()
    process = widget.file_process
    for event in (
        {"stage": "transcribe", "completed": True, "total": 2},
        {"stage": "transcribe", "completed": 3, "total": 2},
        {"stage": "transcribe", "completed": 0, "total": 2**40},
        {"stage": "unexpected"},
    ):
        process.stderr = ("transcription_progress " + json.dumps(event) + "\n").encode()
        process.readyReadStandardError.emit()
    assert widget.file_progress.maximum() == 0
    process.stderr = b"x" * 50000
    process.readyReadStandardError.emit()
    assert len(widget.file_progress_buffer) <= 8192
    process.stderr = b"\n" + b"y\n" * 20000 + b"recognizer failed\n"
    process.readyReadStandardError.emit()
    assert len(widget.file_errors) <= 16384
    widget.cancel_file()
    assert process.killed
    process.finished.emit(1, QProcess.ExitStatus.CrashExit)
    assert "recognizer failed" in widget.file_text.toPlainText()
    assert "transcription_progress" not in widget.file_text.toPlainText()
    assert widget.file_result is None
    assert widget.file_progress.format() == "Stopped"
