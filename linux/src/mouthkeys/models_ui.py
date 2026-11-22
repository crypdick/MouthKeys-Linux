"""Local-only model browser. Network and activation require explicit buttons."""

import json
import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol, cast

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import benchmark, engines
from .config import Config
from .qt_control import Response, Task
from .settings_fields import SettingsField, field_value

_INFERENCE_FIELDS = (
    "device",
    "compute_type",
    "language",
    "asr_max_tokens",
    "native_streaming",
    "streaming_latency",
)


class _ModelSettings(Protocol):
    fields: Mapping[str, SettingsField]
    jobs: set[QThread]

    def save(self) -> bool: ...
    def send(self, action: str) -> None: ...
    def download_model(self) -> None: ...
    def delete_model(self) -> None: ...


class ModelBrowser(QDialog):
    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self.settings = cast("_ModelSettings", window)
        self.benchmark_task: Task | None = None
        self.measurement_context: tuple[str, str | None] | None = None
        self.setWindowTitle("Speech models")
        self.resize(1100, 550)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Inspect without downloading. Sort by clicking a column heading. "
            "Accuracy depends on language/audio; unknown memory and speed are not guessed."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.resident = QLabel("Resident model: service state unavailable")
        self.resident.setWordWrap(True)
        layout.addWidget(self.resident)
        filters = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Filter model name, provider, or language")
        self.query.textChanged.connect(self.filter_rows)
        filters.addWidget(self.query)
        self.state = QComboBox()
        self.state.addItems(["All models", "Installed", "Not downloaded"])
        self.state.currentTextChanged.connect(self.filter_rows)
        filters.addWidget(self.state)
        layout.addLayout(filters)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Model",
                "Languages",
                "Weight bytes",
                "Memory / notes",
                "Streaming / vocabulary",
                "Installed",
                "Resident",
                "Speed (local)",
                "Accuracy (local)",
                "Word timing",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        actions = QHBoxLayout()
        for label, action in (
            ("Activate selected", self.activate),
            ("Download selected", self.download),
            ("Delete cached selected", self.delete),
            ("Benchmark selected…", self.benchmark),
            ("Refresh", self.refresh),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.refresh()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_resident)
        self.timer.start(1000)

    def _field_values(self, *keys: str) -> dict[str, Any]:
        fields = self.settings.fields
        return {key: field_value(fields[key]) for key in keys if key in fields}

    def refresh(self) -> None:
        self.engine = cast("QComboBox", self.settings.fields["engine"]).currentData()
        config = replace(
            Config(), engine=self.engine, **self._field_values("model", *_INFERENCE_FIELDS)
        )
        models = engines.model_catalog(
            self.engine,
            revision=self._revision().text(),
            measurement_config=config,
            measurement_context=self.measurement_context,
        )
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            values = [
                model["name"],
                model["languages"],
                model.get("weight_bytes") or 0,
                model.get("memory_note", "Model/device dependent"),
                f"streaming: {'yes' if model.get('native_streaming') else 'provisional/offline'}; "
                f"vocabulary: {model.get('vocabulary_method', 'backend dependent')}",
                "Yes" if model["installed"] else "No",
                "No",
                model.get("speed_note", "Not benchmarked on this hardware"),
                model.get("accuracy_note", "Not evaluated for your language/audio"),
                "Native" if model.get("word_timing") else "Unavailable",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem()
                if col == 2 and value:
                    item.setData(Qt.ItemDataRole.DisplayRole, value)
                else:
                    item.setText("Unknown" if col == 2 else str(value))
                item.setData(Qt.ItemDataRole.UserRole, model)
                item.setToolTip(model.get("source", "") + "\n" + str(value))
                if model.get("measurement"):
                    item.setToolTip(
                        item.toolTip()
                        + "\n"
                        + json.dumps(model["measurement"], ensure_ascii=False, indent=2)
                    )
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.filter_rows()
        self.update_resident()

    def _revision(self) -> QLineEdit:
        return cast("QLineEdit", self.settings.fields["model_revision"])

    def _row_model(self, row: int) -> dict[str, Any]:
        # Published rows have all catalog cells; the table itself is read-only.
        item = cast("QTableWidgetItem", self.table.item(row, 0))
        return cast("dict[str, Any]", item.data(Qt.ItemDataRole.UserRole))

    def update_resident(self) -> None:
        loaded: object = getattr(self.settings, "loaded_model", None)
        if not isinstance(loaded, dict) or not all(
            isinstance(loaded.get(key), str)
            for key in ("engine", "model", "revision", "device", "compute_type")
        ):
            loaded = None
        self.resident.setText(
            f"Resident model: {loaded['engine']} / {loaded['model']} @ {loaded['revision']} "
            f"({loaded['device']}, {loaded['compute_type']})"
            if loaded
            else "Resident model: none reported"
        )
        revision = self._revision().text()
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            model = self._row_model(row)
            active = (
                loaded
                and loaded["engine"] == self.engine
                and loaded["model"] == model["model"]
                and loaded["revision"] == model.get("revision", revision)
            )
            resident = cast("QTableWidgetItem", self.table.item(row, 6))
            resident.setText("Yes" if active else "No")
        self.table.setSortingEnabled(sorting)

    def filter_rows(self, *_: object) -> None:
        query = self.query.text().casefold()
        state = self.state.currentText()
        for row in range(self.table.rowCount()):
            model = self._row_model(row)
            text = " ".join(str(value) for value in model.values()).casefold()
            visible = query in text and (
                state == "All models"
                or (model["installed"] if state == "Installed" else not model["installed"])
            )
            self.table.setRowHidden(row, not visible)

    def _selected(self) -> dict[str, Any] | None:
        row = self.table.currentRow()
        if row < 0 or self.table.isRowHidden(row):
            return None
        item = cast("QTableWidgetItem", self.table.item(row, 0))
        return self._row_model(row) if item.isSelected() else None

    def choose(self) -> bool:
        model = self._selected()
        if model is None:
            return False
        cast("QComboBox", self.settings.fields["model"]).setEditText(model["model"])
        if model.get("revision"):
            self._revision().setText(model["revision"])
        return True

    def activate(self) -> None:
        if self.choose() and self.settings.save():
            self.settings.send("warmup")

    def download(self) -> None:
        if self.choose():
            self.settings.download_model()

    def delete(self) -> None:
        if self.choose():
            self.settings.delete_model()
            self.refresh()

    def benchmark(self) -> None:
        if self.benchmark_task is not None or (model := self._selected()) is None:
            return
        if not model["installed"]:
            QMessageBox.information(
                self, "Offline benchmark", "Download this model explicitly first."
            )
            return
        fixture, _ = QFileDialog.getOpenFileName(self, "Choose benchmark WAV", "", "WAV (*.wav)")
        if not fixture:
            return
        reference, accepted = QInputDialog.getMultiLineText(
            self,
            "Optional reference",
            "Exact reference transcript for this WAV, or leave blank.\n"
            "Results describe this fixture only. Text and file paths are not saved.",
        )
        if not accepted:
            return
        try:
            config = replace(
                Config.load(),
                engine=self.engine,
                model=model["model"],
                model_revision=model["revision"],
                **self._field_values(*_INFERENCE_FIELDS),
            ).validate()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Benchmark", str(exc))
            return
        cancelled = threading.Event()
        progress = QProgressDialog("Loading and decoding offline WAV…", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.canceled.connect(cancelled.set)
        progress.show()
        task = Task(
            lambda: benchmark.run_benchmark(
                config,
                fixture,
                reference=reference if reference.strip() else None,
                cancel_event=cancelled,
            ),
            self,
            cancelled,
        )
        self.benchmark_task = task
        self.settings.jobs.add(task)

        def received(response: Response) -> None:
            progress.close()
            if "error" in response:
                QMessageBox.warning(self, "Benchmark", str(response["error"]))
            else:
                report = cast("dict[str, Any]", response["result"])
                self.measurement_context = (
                    report["fixture_sha256"],
                    (report.get("accuracy") or {}).get("reference_sha256"),
                )
                self.refresh()
                summary = (
                    f"Decode: {report['decode_seconds']:.3f}s for "
                    f"{report['audio_seconds']:.3f}s audio; "
                    f"RTF {report['real_time_factor']:.3f}.\n"
                    f"Cold load: {report['load_seconds']:.3f}s.\n"
                    f"{report['scope']}\n{report['memory_scope']}"
                )
                box = QMessageBox(
                    QMessageBox.Icon.Information,
                    "Benchmark saved",
                    summary,
                    QMessageBox.StandardButton.Ok,
                    self,
                )
                box.setDetailedText(json.dumps(report, indent=2))
                box.exec()

        def finished() -> None:
            self.settings.jobs.discard(task)
            self.benchmark_task = None
            progress.deleteLater()

        task.done.connect(received)
        task.finished.connect(finished)
        task.finished.connect(task.deleteLater)
        task.start()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self.benchmark_task is not None:
            event.ignore()
            return
        super().closeEvent(event)
