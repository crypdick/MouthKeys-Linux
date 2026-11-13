"""Local-only model browser. Network and activation require explicit buttons."""

import json
import threading
from dataclasses import replace

from PySide6.QtCore import Qt, QTimer
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
)


class ModelBrowser(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.benchmark_task = None
        self.measurement_context = None
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

    def refresh(self):
        from .config import Config
        from .engines import model_catalog
        from .gui import field_value

        engine = self.window.fields["engine"].currentData()
        self.engine = engine
        values = {
            key: field_value(self.window.fields[key])
            for key in (
                "model",
                "language",
                "device",
                "compute_type",
                "asr_max_tokens",
                "native_streaming",
                "streaming_latency",
            )
            if key in self.window.fields
        }
        config = replace(Config(), **values, engine=engine)
        models = model_catalog(
            engine,
            revision=self.window.fields["model_revision"].text(),
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

    def update_resident(self):
        loaded = getattr(self.window, "loaded_model", None)
        self.resident.setText(
            f"Resident model: {loaded['engine']} / {loaded['model']} @ {loaded['revision']} "
            f"({loaded['device']}, {loaded['compute_type']})"
            if loaded
            else "Resident model: none reported"
        )
        revision = self.window.fields["model_revision"].text()
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        for row in range(self.table.rowCount()):
            model = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            active = (
                loaded
                and loaded["engine"] == self.engine
                and loaded["model"] == model["model"]
                and loaded["revision"] == model.get("revision", revision)
            )
            self.table.item(row, 6).setText("Yes" if active else "No")
        self.table.setSortingEnabled(sorting)

    def filter_rows(self, *_):
        query = self.query.text().casefold()
        for row in range(self.table.rowCount()):
            model = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            text = " ".join(str(value) for value in model.values()).casefold()
            state = self.state.currentText()
            visible = query in text and (
                state == "All models"
                or (model["installed"] if state == "Installed" else not model["installed"])
            )
            self.table.setRowHidden(row, not visible)

    def choose(self):
        row = self.table.currentRow()
        if row < 0:
            return False
        model = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.window.fields["model"].setEditText(model["model"])
        if model.get("revision"):
            self.window.fields["model_revision"].setText(model["revision"])
        return True

    def activate(self):
        if self.choose() and self.window.save():
            self.window.send("warmup")

    def download(self):
        if self.choose():
            self.window.download_model()

    def delete(self):
        if self.choose():
            self.window.delete_model()
            self.refresh()

    def benchmark(self):
        from .assistant_ui import AssistantTask
        from .benchmark import run_benchmark
        from .config import Config
        from .gui import field_value

        if self.benchmark_task is not None or self.table.currentRow() < 0:
            return
        model = self.table.item(self.table.currentRow(), 0).data(Qt.ItemDataRole.UserRole)
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
            settings = {
                key: field_value(self.window.fields[key])
                for key in (
                    "device",
                    "compute_type",
                    "language",
                    "asr_max_tokens",
                    "native_streaming",
                    "streaming_latency",
                )
                if key in self.window.fields
            }
            config = replace(
                Config.load(),
                engine=self.engine,
                model=model["model"],
                model_revision=model["revision"],
                **settings,
            ).validate()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Benchmark", str(exc))
            return
        cancelled = threading.Event()
        progress = QProgressDialog("Loading and decoding offline WAV…", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.canceled.connect(cancelled.set)
        progress.show()
        task = AssistantTask(
            lambda: run_benchmark(
                config,
                fixture,
                reference=reference if reference.strip() else None,
                cancel_event=cancelled,
            ),
            self,
            cancelled,
        )
        self.benchmark_task = task
        self.window.jobs.add(task)

        def received(response):
            progress.close()
            if "error" in response:
                QMessageBox.warning(self, "Benchmark", response["error"])
            else:
                report = response["result"]
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

        def finished():
            self.window.jobs.discard(task)
            self.benchmark_task = None
            progress.deleteLater()

        task.done.connect(received)
        task.finished.connect(finished)
        task.finished.connect(task.deleteLater)
        task.start()

    def closeEvent(self, event):
        if self.benchmark_task is not None:
            event.ignore()
            return
        super().closeEvent(event)
