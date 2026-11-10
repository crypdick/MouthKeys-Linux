"""Local-only model browser. Network and activation require explicit buttons."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class ModelBrowser(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Speech models")
        self.resize(900, 550)
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
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Model",
                "Languages",
                "Weight bytes",
                "Memory / notes",
                "Streaming / vocabulary",
                "Installed",
                "Resident",
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
        from .engines import model_catalog

        engine = self.window.fields["engine"].currentData()
        self.engine = engine
        models = model_catalog(engine, revision=self.window.fields["model_revision"].text())
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            values = [
                model["name"],
                model["languages"],
                model.get("weight_bytes") or 0,
                model.get("memory_note", "Model/device dependent"),
                f"streaming: {'yes' if model.get('native_streaming') else 'provisional/offline'}; "
                f"vocabulary: {'yes' if model.get('vocabulary') else 'backend dependent'}",
                "Yes" if model["installed"] else "No",
                "No",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem()
                if col == 2 and value:
                    item.setData(Qt.ItemDataRole.DisplayRole, value)
                else:
                    item.setText("Unknown" if col == 2 else str(value))
                item.setData(Qt.ItemDataRole.UserRole, model)
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
