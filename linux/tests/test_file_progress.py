import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QMimeData, QPointF, QProcess, Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from mouthkeys import cli, file_transcription, file_transcription_ui, gui
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
        transcriber=SimpleNamespace(segments=segments),
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
        transcriber=SimpleNamespace(segments=segments),
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


class FakeFileProcess(QProcess):
    def __init__(self, parent):
        super().__init__(parent)
        self.stderr = b""
        self.stdout = json.dumps(result()).encode()
        self.killed = False

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = arguments

    def start(self):
        pass

    def readAllStandardError(self):
        data, self.stderr = self.stderr, b""
        return QByteArray(data)

    def readAllStandardOutput(self):
        return QByteArray(self.stdout)

    def errorString(self):  # noqa: N802 - Qt override
        return "Fake child could not start"

    def kill(self):
        self.killed = True


def settings_window(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    monkeypatch.setattr(gui.Window, "save", lambda self: True)
    monkeypatch.setattr(file_transcription_ui, "QProcess", FakeFileProcess)
    widget = gui.Window()
    widget.timer.stop()
    return widget


def test_gui_reads_live_fragmented_progress_and_copies_only_on_explicit_click(
    monkeypatch, tmp_path
):
    widget = settings_window(monkeypatch).file_page
    copied = []
    monkeypatch.setattr(QApplication, "clipboard", lambda: SimpleNamespace(setText=copied.append))
    source = tmp_path / "source.wav"
    source.touch()
    widget.file_path.setText(str(source))
    widget.transcribe_file()
    process = widget.run.process
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
    process.stderr = b'transcription_progress {"stage":"transcribe","completed":2,"total":4}\n'
    process.readyReadStandardError.emit()
    assert widget.file_progress_label.text() == "Transcribing audio… 2 / 4 chunks"
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
    assert widget.run is None


def test_gui_progress_bounds_stderr_and_rejects_invalid_counters(monkeypatch, tmp_path):
    widget = settings_window(monkeypatch).file_page
    source = tmp_path / "source.wav"
    source.touch()
    widget.file_path.setText(str(source))
    widget.transcribe_file()
    process = widget.run.process
    for event in (
        [],
        {"stage": []},
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
    assert len(widget.run.progress_buffer) <= 8192
    process.stderr = b"\n" + b"y\n" * 20000 + b"recognizer failed\n"
    process.readyReadStandardError.emit()
    assert len(widget.run.errors) <= 16384
    widget.cancel_file()
    assert process.killed
    process.finished.emit(1, QProcess.ExitStatus.CrashExit)
    assert "recognizer failed" in widget.file_text.toPlainText()
    assert "transcription_progress" not in widget.file_text.toPlainText()
    assert widget.file_result is None
    assert widget.file_progress.format() == "Stopped"


def test_file_workflow_guards_close_and_discards_success_after_cancel(monkeypatch, tmp_path):
    window = settings_window(monkeypatch)
    page = window.file_page
    monkeypatch.setattr(QApplication, "clipboard", lambda: pytest.fail("No result to copy"))
    page.copy_file_result()
    page.export_file_result("txt")
    page.cancel_file()
    page.transcribe_file()
    assert window.status.text() == "Choose an existing audio file"
    assert not window.busy
    source = tmp_path / "chosen.wav"
    source.touch()
    page.file_path.setText(str(source))
    page.transcribe_file()
    process = page.run.process
    page.transcribe_file()
    assert page.run.process is process
    assert window.busy
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()

    page.cancel_file()
    assert process.killed
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert page.file_result is None
    assert not FileHistory().rows()
    assert page.file_progress.format() == "Stopped"
    assert not window.busy
    window.closeEvent(event)
    assert event.isAccepted()


def test_start_failure_releases_window_and_stale_signals_cannot_finish_retry(monkeypatch, tmp_path):
    window = settings_window(monkeypatch)
    page = window.file_page
    source = tmp_path / "input.wav"
    source.touch()
    page.file_path.setText(str(source))
    page.transcribe_file()
    failed = page.run.process
    failed.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    assert not window.busy
    assert "could not start" in page.file_text.toPlainText()
    assert page.file_progress.format() == "Stopped"

    page.transcribe_file()
    retry = page.run.process
    failed.readyReadStandardError.emit()
    failed.finished.emit(0, QProcess.ExitStatus.NormalExit)
    failed.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    assert page.run.process is retry
    assert page.file_result is None
    retry.errorOccurred.emit(QProcess.ProcessError.Crashed)
    assert window.busy  # QProcess emits finished after a crash.
    retry.finished.emit(0, QProcess.ExitStatus.CrashExit)
    assert not window.busy
    assert not FileHistory().rows()
    assert page.file_progress.format() == "Stopped"


@pytest.mark.parametrize("setting", ["save_history", "save_file_history"])
def test_file_result_respects_history_snapshot_but_can_be_exported(monkeypatch, tmp_path, setting):
    replace(Config(), **{setting: "off"}).save()
    page = settings_window(monkeypatch).file_page
    source = tmp_path / "audio.wav"
    source.touch()
    path = tmp_path / "transcript.json"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    page.choose_file()
    assert not page.file_path.text()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    page.choose_file()
    assert page.file_path.text() == str(source)
    page.transcribe_file()
    Config().save()  # In-flight jobs use the configuration saved when they started.
    page.run.process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert page.file_result == result()
    assert not FileHistory().rows()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    page.export_file_result("json")
    assert not path.exists()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    page.export_file_result("json")
    assert json.loads(path.read_text()) == result()
    assert path.stat().st_mode & 0o777 == 0o600
    assert page.file_progress.format() == "Complete"


def test_file_history_selection_and_confirmations_are_wired_to_the_page(monkeypatch):
    history = FileHistory()
    item = history.add("one.wav", result() | {"confidence": 0.8})
    history.add("two.wav", result() | {"text": "Other transcript"})
    window = settings_window(monkeypatch)
    page = window.file_page
    page.file_history_search.setText("ONE.WAV")
    assert page.file_history_list.count() == 1
    page.file_history_list.setCurrentRow(0)
    assert page.file_text.toPlainText() == "Final words."
    assert "80% model likelihood" in window.status.text()
    page.file_history_list.setCurrentRow(-1)
    page.delete_file_history()
    assert history.rows("one.wav")[0]["id"] == item
    page.file_history_list.setCurrentRow(0)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    page.delete_file_history()
    page.clear_file_history()
    assert len(history.rows()) == 2
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    page.delete_file_history()
    assert not history.rows("one.wav")
    page.clear_file_history()
    assert not history.rows()


def test_settings_accepts_only_local_file_drops_and_selects_file_page(monkeypatch, tmp_path):
    window = settings_window(monkeypatch)
    mime = QMimeData()
    remote = QUrl("https://example.test/remote.wav")
    source = tmp_path / "incoming.wav"

    def events(urls):
        mime.setUrls(urls)
        args = (
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        enter = QDragEnterEvent(QPointF(0, 0), *args)
        drop = QDropEvent(QPointF(0, 0), *args)
        window.dragEnterEvent(enter)
        window.dropEvent(drop)
        return enter, drop

    for event in events([remote]):
        assert not event.isAccepted()
    assert not window.file_page.file_path.text()
    for event in events([remote, QUrl.fromLocalFile(str(source))]):
        assert event.isAccepted()
    assert window.file_page.file_path.text() == str(source)
    assert window.nav.currentItem().text() == "File Transcription"
    assert not window.isVisible()


def test_file_workflow_stops_on_invalid_worker_result_or_export_error(monkeypatch, tmp_path):
    window = settings_window(monkeypatch)
    page = window.file_page
    source = tmp_path / "input.wav"
    source.touch()
    page.file_path.setText(str(source))
    page.transcribe_file()
    process = page.run.process
    process.stderr = b"transcription_progress {not json}\n"
    process.readyReadStandardError.emit()
    process.stdout = b'{"text":"metadata is missing"}'
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert not FileHistory().rows()
    assert page.file_result is None
    assert page.file_text.toPlainText() == "Invalid result duration"

    page.show_file_result(result() | {"processing_seconds": 0})
    assert "realtime" not in window.status.text()
    path = tmp_path / "missing" / "transcript.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    page.export_file_result("txt")
    assert page.file_progress_label.text() == "File export failed"
    assert not path.exists()


def test_unsaved_settings_error_never_starts_file_transcription(monkeypatch, tmp_path):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    monkeypatch.setattr(gui.Window, "save", lambda self: False)
    window = gui.Window()
    window.timer.stop()
    source = tmp_path / "chosen.wav"
    source.touch()
    window.file_page.file_path.setText(str(source))
    window.file_page.transcribe_file()
    assert not window.busy
