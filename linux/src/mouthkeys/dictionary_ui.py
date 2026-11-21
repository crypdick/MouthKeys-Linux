"""Edit local rules, stage voice training, and transfer dictionaries from Settings."""

import re
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .dictionary import Dictionary, DictionaryData, FormattingRule, Replacement
from .qt_control import Response, Send

_CONTROL_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\"}


def _escape_symbol(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def _unescape_symbol(value: str) -> str:
    return re.sub(r"\\([\\nt])", lambda match: _CONTROL_ESCAPES[match[1]], value)


def _append_cells(
    table: QTableWidget,
    aliases: str,
    value: str,
    *,
    enabled: bool = True,
    identity: str | None = None,
) -> tuple[int, QTableWidgetItem]:
    first = QTableWidgetItem(aliases)
    if identity:
        first.setData(Qt.ItemDataRole.UserRole, identity)
    row = table.rowCount()
    table.insertRow(row)
    for column, item in enumerate(
        (first, QTableWidgetItem(value), QTableWidgetItem("on" if enabled else "off"))
    ):
        table.setItem(row, column, item)
    return row, first


def _rules(table: QTableWidget) -> Iterator[tuple[list[str], str, bool, str | None]]:
    for row in range(table.rowCount()):
        aliases, value, enabled = (
            item.text() if (item := table.item(row, column)) else "" for column in range(3)
        )
        if enabled not in ("on", "off"):
            raise ValueError("Enabled must be on or off")
        trigger = table.item(row, 0)
        identity = trigger.data(Qt.ItemDataRole.UserRole) if trigger else None
        yield (
            [phrase.strip() for phrase in aliases.split(",") if phrase.strip()],
            value,
            enabled == "on",
            identity if isinstance(identity, str) else None,
        )


def _formatting(table: QTableWidget) -> list[FormattingRule]:
    return [
        {"aliases": aliases, "symbol": _unescape_symbol(value), "enabled": enabled}
        for aliases, value, enabled, _ in _rules(table)
    ]


class DictionaryPage(QWidget):
    vocabulary_changed = Signal()

    def __init__(self, send: Send, show_status: Callable[[str], None]) -> None:
        super().__init__()
        self.send = send
        self.show_status = show_status
        self.training_id: str | None = None
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Teach replacements, then customize phrases used after your "
            "Literal prefix. Separate aliases with commas. Disable a rule "
            "without deleting it."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.tabs = QTabWidget()
        self.tables: dict[str, QTableWidget] = {}
        for name in ("replacements", "punctuation", "actions"):
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(
                ["Spoken phrases (comma-separated)", "Replacement", "Enabled (on/off)"]
            )
            table.horizontalHeader().setStretchLastSection(True)
            table.setColumnWidth(0, 260)
            table.setColumnWidth(1, 160)
            self.tables[name] = table
            self.tabs.addTab(table, name.capitalize())
        layout.addWidget(self.tabs)
        training = QHBoxLayout()
        self.training_replacement = QLineEdit()
        self.training_replacement.setPlaceholderText("Correct word or phrase to teach")
        training.addWidget(self.training_replacement)
        train = QPushButton("Train by voice")
        train.clicked.connect(self.train)
        training.addWidget(train)
        stop_training = QPushButton("Stop training")
        stop_training.clicked.connect(lambda: self.send("stop", None, None))
        training.addWidget(stop_training)
        layout.addLayout(training)
        buttons = QHBoxLayout()
        for text, handler in (
            ("Add rule", self.add_rule),
            ("Delete rule", self.delete_rule),
            ("Save", self.save),
            ("Import", self.import_dictionary),
            ("Export", self.export_dictionary),
            ("Reload", self.reload),
            ("Reset spoken formatting", self.reset_spoken_formatting),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.load_rules(Dictionary.load())

    def train(self) -> None:
        if not self.training_replacement.text().strip():
            self.show_status("Enter the correct replacement first, then say it aloud")
            return
        self.send("start", {"mode": "training"}, self.training_started)

    def training_started(self, response: Response) -> None:
        result = response.get("result")
        identity = result.get("id") if isinstance(result, dict) else None
        if isinstance(identity, str):
            self.training_id = identity

    def update_training(self, status: Mapping[str, object]) -> None:
        proposal = status.get("assistant_proposal")
        if (
            not self.training_id
            or status.get("id") != self.training_id
            or not isinstance(proposal, dict)
            or proposal.get("mode") != "training"
        ):
            return
        self.training_id = None
        heard = proposal.get("text")
        trigger = heard.strip().rstrip(".!?") if isinstance(heard, str) else ""
        replacement = self.training_replacement.text().strip()
        if not trigger:
            self.show_status("No recognized training phrase; try again")
            return
        _append_cells(self.tables["replacements"], trigger, replacement)
        self.tabs.setCurrentIndex(0)
        self.show_status("Training captured. Review recognized phrase and Save dictionary.")

    def load_rules(self, data: DictionaryData) -> None:
        self.baseline = deepcopy(data)
        replacements = self.tables["replacements"]
        replacements.setRowCount(0)
        for entry in data["replacements"]:
            _append_cells(
                replacements,
                ", ".join(entry["triggers"]),
                entry["replacement"],
                enabled=entry.get("enabled", True),
                identity=entry.get("id"),
            )
        for name, entries in (("punctuation", data["punctuation"]), ("actions", data["actions"])):
            table = self.tables[name]
            table.setRowCount(0)
            for rule in entries:
                _append_cells(
                    table,
                    ", ".join(rule["aliases"]),
                    _escape_symbol(rule["symbol"]),
                    enabled=rule.get("enabled", True),
                )

    def add_rule(self) -> None:
        table = self.tabs.currentWidget()
        if isinstance(table, QTableWidget):
            row, first = _append_cells(table, "", "")
            table.setCurrentCell(row, 0)
            table.editItem(first)

    def delete_rule(self) -> None:
        table = self.tabs.currentWidget()
        if isinstance(table, QTableWidget) and table.currentRow() >= 0:
            table.removeRow(table.currentRow())

    def save(self) -> bool:
        try:
            replacements: list[Replacement] = []
            for triggers, value, enabled, identity in _rules(self.tables["replacements"]):
                entry: Replacement = {
                    "triggers": triggers,
                    "replacement": value,
                    "enabled": enabled,
                }
                if identity:
                    entry["id"] = identity
                replacements.append(entry)
            data: DictionaryData = {
                "replacements": replacements,
                "punctuation": _formatting(self.tables["punctuation"]),
                "actions": _formatting(self.tables["actions"]),
            }
            Dictionary.save(data, expected=self.baseline)
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))
            return False
        self.baseline = data
        self.show_status("Dictionary saved")
        return True

    def reload(self) -> None:
        try:
            self.load_rules(Dictionary.load())
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))

    def reset_spoken_formatting(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Reset spoken formatting",
                "Restore default punctuation/action rules? Unsaved edits will be discarded.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            current, defaults = Dictionary.load(), Dictionary.defaults()
            data: DictionaryData = {
                "replacements": current["replacements"],
                "punctuation": defaults["punctuation"],
                "actions": defaults["actions"],
            }
            Dictionary.save(data, expected=current)
            self.load_rules(data)
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))

    def import_dictionary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import dictionary", "", "JSON (*.json)")
        if not path:
            return
        mode, accepted = QInputDialog.getItem(
            self,
            "Import dictionary",
            "Import replacements and custom words:",
            ["Merge with existing data", "Replace supplied sections"],
            current=0,
            editable=False,
        )
        if not accepted:
            return
        try:
            self.load_rules(
                Dictionary.import_file(
                    path, mode="merge" if mode.startswith("Merge") else "replace"
                )
            )
            self.vocabulary_changed.emit()
            self.show_status("Dictionary and supplied vocabulary imported")
        # Report transfer failure without duplicating dictionary contents in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.show_status(str(exc))

    def export_dictionary(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export dictionary", "dictionary.json", "JSON (*.json)"
        )
        if not path or not self.save():
            return
        selected, accepted = QInputDialog.getItem(
            self,
            "Export dictionary",
            "Export format:",
            [
                "Linux (includes punctuation/actions)",
                "Upstream (enabled replacements + custom words)",
            ],
            current=0,
            editable=False,
        )
        if accepted:
            try:
                Dictionary.export_file(
                    path, format="native" if selected.startswith("Linux") else "upstream"
                )
                self.show_status("Dictionary and vocabulary exported")
            except (OSError, TypeError, ValueError) as exc:
                self.show_status(str(exc))
