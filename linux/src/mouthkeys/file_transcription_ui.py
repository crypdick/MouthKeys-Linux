"""File transcription and history UI; owns one isolated CLI process at a time."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import Config
from .file_history import FileHistory, FileHistoryEntry, FileResult, parse_file_result
from .file_transcription import export_result, result_text
from .processes import python_command


@dataclass
class FileRun:
    process: QProcess
    config: Config
    filename: str
    progress_buffer: bytes = b""
    errors: str = ""
    cancelled: bool = False


class FileTranscriptionPage(QWidget):
    def __init__(
        self, parent: QWidget, save_settings: Callable[[], bool], show_status: Callable[[str], None]
    ) -> None:
        super().__init__(parent)
        self.save_settings = save_settings
        self.show_status = show_status
        layout = QVBoxLayout(self)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Choose an audio file or drop one here")
        layout.addWidget(self.file_path)
        buttons = QHBoxLayout()
        for label, handler in (
            ("Choose file", self.choose_file),
            ("Transcribe", self.transcribe_file),
            ("Cancel", self.cancel_file),
            ("Copy result", self.copy_file_result),
            ("Export text", lambda: self.export_file_result("txt")),
            ("Export JSON", lambda: self.export_file_result("json")),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.file_result: FileResult | None = None
        self.run: FileRun | None = None
        self.file_rows: list[FileHistoryEntry] = []
        self.file_progress_label = QLabel("No transcription running")
        layout.addWidget(self.file_progress_label)
        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 1)
        self.file_progress.setValue(0)
        self.file_progress.setFormat("Ready")
        layout.addWidget(self.file_progress)
        self.file_text = QTextEdit()
        self.file_text.setReadOnly(True)
        self.file_text.setPlaceholderText(
            "File transcription stays separate from dictation. Speaker labels "
            "and expected count are configured in App Settings. Models must be"
            " downloaded explicitly."
        )
        split = QHBoxLayout()
        browser = QWidget()
        history_layout = QVBoxLayout(browser)
        self.file_history_search = QLineEdit()
        self.file_history_search.setPlaceholderText("Search file history")
        self.file_history_search.textChanged.connect(self.refresh_file_history)
        history_layout.addWidget(self.file_history_search)
        self.file_history_list = QListWidget()
        self.file_history_list.currentRowChanged.connect(self.select_file_history)
        history_layout.addWidget(self.file_history_list)
        for text, handler in (
            ("Delete selected result", self.delete_file_history),
            ("Clear file history", self.clear_file_history),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            history_layout.addWidget(button)
        split.addWidget(browser, 1)
        split.addWidget(self.file_text, 2)
        layout.addLayout(split)
        self.refresh_file_history()

    def refresh_file_history(self, *_: object) -> None:
        self.file_rows = FileHistory().rows(self.file_history_search.text())
        self.file_history_list.clear()
        for row in self.file_rows:
            when = datetime.fromtimestamp(row["created"], UTC).astimezone().strftime("%m-%d %H:%M")
            self.file_history_list.addItem(f"{when} · {row['name']}\n{row['result']['text'][:70]}")

    def select_file_history(self, index: int) -> None:
        if 0 <= index < len(self.file_rows):
            self.show_file_result(self.file_rows[index]["result"])

    def show_file_result(self, result: FileResult) -> None:
        confidence = result.get("confidence")
        self.file_result = result
        self.file_text.setPlainText(result_text(result))
        self.show_status(
            f"{result['duration']:.1f}s audio · {result['processing_seconds']:.1f}s processing"
            + (
                f" · {result['duration'] / result['processing_seconds']:.1f}× realtime"
                if result["processing_seconds"] > 0
                else ""
            )
            + (
                f" · {confidence:.0%} model likelihood (not calibrated confidence)"
                if confidence is not None
                else ""
            )
        )

    def delete_file_history(self) -> None:
        index = self.file_history_list.currentRow()
        if (
            0 <= index < len(self.file_rows)
            and QMessageBox.question(self, "Delete file result", "Delete selected saved result?")
            == QMessageBox.StandardButton.Yes
        ):
            FileHistory().delete(self.file_rows[index]["id"])
            self.refresh_file_history()

    def clear_file_history(self) -> None:
        if (
            QMessageBox.question(self, "Clear file history", "Delete all saved file results?")
            == QMessageBox.StandardButton.Yes
        ):
            FileHistory().clear()
            self.refresh_file_history()

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Transcribe audio", "", "Audio/video files (*)")
        if path:
            self.file_path.setText(path)

    @property
    def running(self) -> bool:
        return self.run is not None

    def transcribe_file(self) -> None:
        if self.running or not self.save_settings():
            return
        filename = self.file_path.text()
        if not Path(filename).is_file():
            self.show_status("Choose an existing audio file")
            return
        self.file_result = None
        process = QProcess(self)
        run = self.run = FileRun(process, Config.load(), filename)
        program, *arguments = python_command(
            "mouthkeys", "transcribe", filename, "--json", "--progress"
        )
        process.setProgram(program)
        process.setArguments(arguments)
        self.file_text.setPlainText("Transcribing…")
        self.update_file_progress({"stage": "decode"})
        process.readyReadStandardError.connect(lambda: self.read_file_progress(run))
        process.finished.connect(lambda code, status: self.file_finished(run, code, status))
        process.errorOccurred.connect(lambda error: self.file_process_error(run, error))
        process.start()

    def file_process_error(self, run: FileRun, error: QProcess.ProcessError) -> None:
        if self.run is run and error == QProcess.ProcessError.FailedToStart:
            self.file_text.setPlainText(run.process.errorString())
            run.process.deleteLater()
            self.run = None
            self.stop_file_progress("File transcription could not start")

    def update_file_progress(self, event: object) -> None:
        labels = {
            "decode": "Decoding audio…",
            "diarization": "Finding speaker turns…",
            "transcribe": "Transcribing audio…",
            "finalize": "Assembling transcript…",
            "save": "Saving file history…",
            "export": "Exporting transcript…",
            "complete": "Transcription complete",
        }
        if not isinstance(event, dict):
            return
        stage = event.get("stage")
        if not isinstance(stage, str) or stage not in labels:
            return
        text = labels[stage]
        if stage == "complete":
            self.file_progress.setRange(0, 1)
            self.file_progress.setValue(1)
            self.file_progress.setFormat("Complete")
        elif "total" in event:
            completed, total = event.get("completed"), event["total"]
            if not (
                type(completed) is int
                and type(total) is int
                and 0 <= completed <= total <= 1000000
                and total > 0
            ):
                return
            self.file_progress.setRange(0, total)
            self.file_progress.setValue(completed)
            self.file_progress.setFormat("%v / %m chunks")
            text += f" {completed} / {total} chunks"
            if event.get("kind") == "uncovered":
                text += " (uncovered audio)"
        else:
            self.file_progress.setRange(0, 0)
        self.file_progress_label.setText(text)

    def stop_file_progress(self, text: str) -> None:
        self.file_progress.setRange(0, 1)
        self.file_progress.setValue(0)
        self.file_progress.setFormat("Stopped")
        self.file_progress_label.setText(text)

    def read_file_progress(self, run: FileRun) -> None:
        if self.run is not run:
            return
        data = run.progress_buffer + bytes(run.process.readAllStandardError().data())
        lines = data.split(b"\n")
        run.progress_buffer = lines.pop()[-8192:]
        for line in lines:
            if line.startswith(b"transcription_progress ") and len(line) <= 8192:
                try:
                    event = json.loads(line.split(b" ", 1)[1])
                except ValueError:
                    pass
                else:
                    self.update_file_progress(event)
                    continue
            run.errors = (run.errors + line.decode(errors="replace") + "\n")[-16384:]

    def file_finished(self, run: FileRun, code: int, status: QProcess.ExitStatus) -> None:
        if self.run is not run:
            return
        self.read_file_progress(run)
        self.run = None
        output = bytes(run.process.readAllStandardOutput().data())
        error = (run.errors + run.progress_buffer.decode(errors="replace")).strip()
        run.process.deleteLater()
        try:
            if run.cancelled or code or status == QProcess.ExitStatus.CrashExit:
                raise ValueError(error or "File transcription cancelled or failed")
            result = parse_file_result(json.loads(output))
            if run.config.save_history == "on" and run.config.save_file_history == "on":
                self.update_file_progress({"stage": "save"})
                FileHistory().add(run.filename, result, run.config.file_history_limit)
                self.refresh_file_history()
            self.show_file_result(result)
            self.update_file_progress({"stage": "complete"})
        # Report to the Settings UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.file_result = None
            self.file_text.setPlainText(str(exc))
            self.stop_file_progress("File transcription cancelled or failed")

    def cancel_file(self) -> None:
        if self.run is not None:
            self.run.cancelled = True
            self.file_progress_label.setText("Cancelling file transcription…")
            self.run.process.kill()

    def copy_file_result(self) -> None:
        if self.file_result is not None:
            QApplication.clipboard().setText(result_text(self.file_result))
            self.show_status("Copied file transcription")

    def export_file_result(self, kind: Literal["json", "txt"]) -> None:
        if self.file_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export transcription", f"transcript.{kind}")
        if path:
            self.update_file_progress({"stage": "export"})
            try:
                export_result(self.file_result, path, kind)
                self.update_file_progress({"stage": "complete"})
            # Report to the Settings UI without duplicating private data in logs.
            except Exception as exc:  # allow: exception-handling  # noqa: BLE001
                self.stop_file_progress("File export failed")
                self.show_status(str(exc))
