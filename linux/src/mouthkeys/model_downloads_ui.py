"""Manage explicit model downloads and cache actions from Settings."""

from collections.abc import Callable
from functools import partial
from typing import Literal

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .engines import delete_model
from .model_progress import DownloadOutput
from .processes import python_command

DownloadCommand = Literal["download", "download-speakers"]


class ModelDownloads(QWidget):
    def __init__(
        self,
        save: Callable[[], bool],
        selected_model: Callable[[], tuple[str, str, str]],
        show_status: Callable[[str], None],
        warmup: Callable[[], None],
    ) -> None:
        super().__init__()
        self.save = save
        self.selected_model = selected_model
        self.show_status = show_status
        self.warmup = warmup
        self.process: QProcess | None = None
        self.output = DownloadOutput()
        self.cancelled = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.model_button = QPushButton("Download selected model")
        self.model_button.clicked.connect(self.download_model)
        layout.addWidget(self.model_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.cancel_button = QPushButton("Cancel model download")
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)
        self.delete_button = QPushButton("Delete selected cached model")
        self.delete_button.clicked.connect(self.delete_model)
        layout.addWidget(self.delete_button)
        self.speaker_button = QPushButton("Download speaker models")
        self.speaker_button.clicked.connect(self.download_speakers)
        layout.addWidget(self.speaker_button)
        load_button = QPushButton("Load model into memory")
        load_button.clicked.connect(warmup)
        layout.addWidget(load_button)

    @property
    def running(self) -> bool:
        return self.process is not None

    def download_model(self) -> None:
        if not self.running and self.save():
            self._start("download")

    def download_speakers(self) -> None:
        if not self.running:
            self._start("download-speakers")

    def delete_model(self) -> None:
        if self.running:
            return
        model, engine, revision = self.selected_model()
        if (
            QMessageBox.question(
                self,
                "Delete model",
                f"Delete cached model {model}? Active model memory is released on service restart.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            delete_model(model, engine, revision=revision)
        # Report engine and filesystem errors in Settings, not in private logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.show_status(str(exc))
            return
        self.show_status("Cached model deleted")

    def cancel(self) -> None:
        if self.process is not None:
            self.cancelled = True
            self.process.kill()

    def _start(self, command: DownloadCommand) -> None:
        process = QProcess(self)
        self.process = process
        self.output = DownloadOutput()
        self.cancelled = False
        self.model_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        self.speaker_button.setEnabled(False)
        if command == "download":
            self.model_button.setText("Downloading…")
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(partial(self._read_output, process))
        else:
            self.speaker_button.setText("Downloading speaker models…")
            process.setStandardOutputFile(QProcess.nullDevice())
            process.readyReadStandardError.connect(partial(self._read_errors, process))
        self.progress.resetFormat()
        self.progress.setRange(0, 0)
        process.finished.connect(partial(self._finished, process, command))
        process.errorOccurred.connect(partial(self._failed_start, process))
        program, *args = python_command("mouthkeys", command)
        process.start(program, args)

    def _read_output(self, process: QProcess) -> None:
        if self.process is not process:
            return
        for event in self.output.feed(bytes(process.readAllStandardOutput().data())):
            total, completed = event["total"], event["completed"]
            if total:
                self.progress.setRange(0, 1000)
                self.progress.setValue(round(min(1.0, completed / total) * 1000))
            else:
                self.progress.setRange(0, 0)
            self.progress.setFormat(
                f"{event['phase']} · {completed}/{total or '?'} {event['unit']}"
            )

    def _read_errors(self, process: QProcess) -> None:
        if self.process is process:
            self.output.append_tail(bytes(process.readAllStandardError().data()))

    def _release(self, process: QProcess, *, success: bool) -> None:
        self.process = None
        process.deleteLater()
        self.model_button.setEnabled(True)
        self.model_button.setText("Download selected model")
        self.speaker_button.setEnabled(True)
        self.speaker_button.setText("Download speaker models")
        self.delete_button.setEnabled(True)
        self.progress.resetFormat()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)

    def _finished(
        self,
        process: QProcess,
        command: DownloadCommand,
        code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        if self.process is not process:
            return
        if command == "download":
            self._read_output(process)
        else:
            self._read_errors(process)
        success = not self.cancelled and code == 0 and exit_status == QProcess.ExitStatus.NormalExit
        failure = self.output.tail.decode(errors="replace").strip()
        if self.cancelled:
            failure = "Download cancelled"
        self._release(process, success=success)
        if command == "download-speakers":
            self.show_status(
                "Speaker models downloaded" if success else failure or "Download failed"
            )
        else:
            if success:
                self.warmup()
            QMessageBox.information(
                self,
                "Model download",
                "Model downloaded" if success else failure or "Download failed",
            )

    def _failed_start(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self.process is process:
            message = process.errorString()
            self._release(process, success=False)
            self.show_status(message)
