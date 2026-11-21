"""Preview a specific backup before issuing an import request."""

import json
from collections.abc import Callable

from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from . import operations
from .qt_control import Response, Send, Task


class BackupPage(QWidget):
    def __init__(
        self,
        send: Send,
        show_status: Callable[[str], None],
        show_result: Callable[[Response], None],
    ) -> None:
        super().__init__()
        self.send = send
        self.show_status = show_status
        self.show_result = show_result
        self.jobs: set[Task] = set()
        self.open_reviews = 0
        layout = QVBoxLayout(self)
        note = QLabel(
            "Back up settings, dictionary, and transcript history. API keys "
            "are excluded. Restore merges history by UUID; it does not delete "
            "existing transcripts. Import previews native and upstream backups, including "
            "unsupported platform settings. Recordings are not retained."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        for text, handler in (
            ("Export backup", self.export_backup),
            ("Import backup", self.import_backup),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            layout.addWidget(button)
        layout.addStretch()

    @property
    def running(self) -> bool:
        return bool(self.jobs) or self.open_reviews > 0

    def import_backup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import backup", "", "JSON (*.json)")
        if not path:
            return
        task = Task(lambda: operations.backup_preview(path), self)
        self.jobs.add(task)
        task.done.connect(lambda response: self.review_import(path, response))
        task.finished.connect(lambda: self.jobs.discard(task))
        task.finished.connect(task.deleteLater)
        task.start()

    def review_import(self, path: str, response: Response) -> None:
        if "error" in response:
            self.show_status(str(response["error"]))
            return
        report = response.get("result")
        if not isinstance(report, dict):
            self.show_status("Backup preview unavailable")
            return
        digest, kind = report.get("file_sha256"), report.get("format")
        if not isinstance(digest, str) or not digest or not isinstance(kind, str):
            self.show_status("Backup preview unavailable")
            return
        self.open_reviews += 1
        try:
            if self.confirm_import(kind, report):
                self.send(
                    "backup_import", {"path": path, "expected_sha256": digest}, self.show_result
                )
        finally:
            self.open_reviews -= 1

    def confirm_import(self, kind: str, report: dict[str, object]) -> bool:
        counts = report.get("counts")
        counts = counts if isinstance(counts, dict) else {}
        warnings = report.get("warnings")
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Review backup import")
        dialog.setText(
            f"{kind.capitalize()} backup: "
            f"{counts.get('history', 0)} transcripts, "
            f"{counts.get('profiles', 0)} profiles, "
            f"{counts.get('replacements', 0)} replacement rules."
        )
        dialog.setInformativeText(
            f"{warning_count} migration warnings. "
            "Review details for settings changes and unsupported data before importing. "
            "History merges by UUID; existing recordings remain. Reopen Settings afterward."
        )
        dialog.setDetailedText(json.dumps(report, ensure_ascii=False, indent=2))
        dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        return dialog.exec() == QMessageBox.StandardButton.Yes

    def export_backup(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export backup", "mouthkeys-backup.json", "JSON (*.json)"
        )
        if path:
            self.send("backup_export", {"path": path}, self.show_result)
