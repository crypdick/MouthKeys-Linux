"""Microphone priority editor and capture-only input meter; never opens audio."""

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import TypeGuard

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .config import Config
from .desktop import command, source_list
from .desktop_input import session

GROUPS = {
    "microphone_preferred": "Preferred",
    "microphone_fallback": "Last resort",
    "microphone_never": "Never",
}


def discover(kind: str) -> list[tuple[str, str]]:
    if session() == "headless":
        return []
    values = json.loads(command(["pactl", "-f", "json", "list", kind]))
    if not isinstance(values, list):
        raise ValueError("Unexpected device list")  # noqa: TRY004 - discovery payload
    devices = []
    for item in values:
        if not isinstance(item, dict) or not isinstance(name := item.get("name"), str):
            continue
        description = item.get("description")
        devices.append((name, description if isinstance(description, str) else name))
    return devices


def _label(name: str, description: str | None) -> str:
    return f"{description} — {name}" if description and description != name else name


def _select_value(combo: QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        combo.addItem(value, value)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


def sync_field(field: object, value: str) -> None:
    if isinstance(field, QLineEdit):
        field.setText(value)
    elif isinstance(field, QComboBox):
        _select_value(field, value)


def _priority_name(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 512
        and value != "default"
        and not any(separator in value for separator in ",\n\r")
    )


def _combo(label: str, value: str) -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItem(label, value)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(20)
    return combo


def _labeled_row(root: QVBoxLayout, label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(widget, 1)
    root.addLayout(row)
    return row


def _meter_percent(status: Mapping[str, object]) -> int:
    meter = status.get("meter")
    value = meter.get("level") if isinstance(meter, dict) else None
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not -sys.float_info.max <= value <= sys.float_info.max
    ):
        return 0
    return round(min(1, max(0, value)) * 100)


class MicrophonePage(QWidget):
    def __init__(self, window: object) -> None:
        super().__init__()
        self.settings = window
        self.active_list: QListWidget | None = None
        self.sources: dict[str, str] = {}
        self.groups: dict[str, QListWidget] = {}
        self.edit_controls: list[QWidget] = []
        config = Config.load()

        root = self._scroll_layout()
        self.current_source = QLabel("No active capture")
        root.addWidget(self.current_source)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setAccessibleName("Input level from active recording only")
        self.meter.setFormat("Input level %p%")
        root.addWidget(self.meter)
        self._input_controls(root, config)
        self._priority_controls(root, config)
        self.sound_output = _combo("Desktop default", "default")
        _select_value(self.sound_output, config.sound_output)
        _labeled_row(root, "Sound output", self.sound_output)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        self.save_button = QPushButton("Save microphone settings")
        self.save_button.clicked.connect(self.save)
        root.addWidget(self.save_button)
        self.refresh()

    def _scroll_layout(self) -> QVBoxLayout:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        note = QLabel(
            "Use the first connected Preferred microphone, then desktop default, then "
            "Last resort. Never devices are excluded. Drag to reorder/move, or use buttons. "
            "An explicit Audio source overrides priority but never exclusions."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        return root

    def _input_controls(self, root: QVBoxLayout, config: Config) -> None:
        self.audio_source = _combo("Desktop default", "default")
        _select_value(self.audio_source, config.audio_source)
        source_row = _labeled_row(root, "Audio source", self.audio_source)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh)
        source_row.addWidget(refresh)
        self.add_picker = _combo("Choose input to add", "")
        _labeled_row(root, "Source to add", self.add_picker)
        self.edit_controls.extend([self.audio_source, self.add_picker])

    def _priority_controls(self, root: QVBoxLayout, config: Config) -> None:
        columns = QHBoxLayout()
        moves = QHBoxLayout()
        for key, title in GROUPS.items():
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            group = self._priority_list(title, source_list(getattr(config, key)))
            self.groups[key] = group
            column.addWidget(group)
            self._add_group_button(column, "Add source here", self.add_source, key)
            columns.addLayout(column)
            self._add_group_button(moves, "Move to " + title, self.move_selected, key)
        root.addLayout(columns, 1)
        row = QHBoxLayout()
        for title, action in (
            ("Move up", lambda: self.reorder(-1)),
            ("Move down", lambda: self.reorder(1)),
            ("Remove from priority", self.remove_selected),
        ):
            button = QPushButton(title)
            button.clicked.connect(action)
            self.edit_controls.append(button)
            row.addWidget(button)
        root.addLayout(row)
        root.addLayout(moves)

    def _add_group_button(
        self,
        layout: QVBoxLayout | QHBoxLayout,
        label: str,
        action: Callable[[str], None],
        key: str,
    ) -> None:
        def clicked() -> None:
            action(key)

        button = QPushButton(label)
        button.clicked.connect(clicked)
        self.edit_controls.append(button)
        layout.addWidget(button)

    def _priority_list(self, title: str, names: list[str]) -> QListWidget:
        group = QListWidget()
        group.setAccessibleName(title + " microphone priority")
        group.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        group.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        group.setDefaultDropAction(Qt.DropAction.MoveAction)
        group.setDragEnabled(True)
        group.setAcceptDrops(True)
        group.setDropIndicatorShown(True)
        group.itemSelectionChanged.connect(lambda: self.select_group(group))
        for name in names:
            self.add_item(group, name)
        return group

    def add_item(self, group: QListWidget, name: str) -> QListWidgetItem:
        item = QListWidgetItem(_label(name, self.sources.get(name)))
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setToolTip(name)
        group.addItem(item)
        return item

    def select_group(self, group: QListWidget) -> None:
        if group.selectedItems():
            self.active_list = group
            for other in self.groups.values():
                if other is not group:
                    other.clearSelection()

    def refresh(self) -> None:
        try:
            sources, sinks = discover("sources"), discover("sinks")
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            self.message.setText(
                f"Device discovery unavailable: {exc}. Custom source/sink names still work."
            )
            return
        self.sources = dict(sources)
        self._reload_picker(self.audio_source, sources, "Desktop default", "default")
        self._reload_picker(self.sound_output, sinks, "Desktop default", "default")
        self._reload_picker(self.add_picker, sources, "Choose input to add", "")
        self.message.setText("Device discovery is read-only; input meter stays empty while idle.")

    def _reload_picker(
        self, combo: QComboBox, devices: list[tuple[str, str]], label: str, default: str
    ) -> None:
        previous = self.combo_value(combo)
        combo.clear()
        combo.addItem(label, default)
        for name, description in devices:
            if combo.findData(name) < 0:
                combo.addItem(_label(name, description), name)
        _select_value(combo, previous)

    @staticmethod
    def combo_value(combo: QComboBox) -> str:
        index = combo.currentIndex()
        if index >= 0 and combo.currentText() == combo.itemText(index):
            data = combo.itemData(index)
            if isinstance(data, str):
                return data
        return combo.currentText().strip()

    def add_source(self, key: str) -> None:
        name = self.combo_value(self.add_picker)
        if not _priority_name(name):
            self.message.setText(
                "Choose or enter an actual source name; desktop default is not a priority entry."
            )
            return
        for group in self.groups.values():
            for index in reversed(range(group.count())):
                if group.item(index).data(Qt.ItemDataRole.UserRole) == name:
                    group.takeItem(index)
        group = self.groups[key]
        group.setCurrentItem(self.add_item(group, name))

    def _selection(self) -> tuple[QListWidget, int] | None:
        group = self.active_list
        if group is not None and (items := group.selectedItems()):
            return group, group.row(items[0])
        return None

    def reorder(self, direction: int) -> None:
        if (selected := self._selection()) is None:
            return
        group, index = selected
        destination = index + direction
        if 0 <= destination < group.count():
            item = group.takeItem(index)
            group.insertItem(destination, item)
            group.setCurrentItem(item)

    def move_selected(self, key: str) -> None:
        if (selected := self._selection()) is None:
            return
        group, index = selected
        item = group.takeItem(index)
        self.groups[key].addItem(item)
        self.groups[key].setCurrentItem(item)

    def remove_selected(self) -> None:
        if (selected := self._selection()) is not None:
            group, index = selected
            group.takeItem(index)

    def _values(self) -> dict[str, str]:
        values = {}
        seen: set[str] = set()
        for key, group in self.groups.items():
            names = []
            for index in range(group.count()):
                name = group.item(index).data(Qt.ItemDataRole.UserRole)
                if not _priority_name(name) or name in seen:
                    raise ValueError("Each microphone can appear only once across priority groups")
                seen.add(name)
                names.append(name)
            values[key] = ",".join(names)
        return {
            **values,
            "audio_source": self.combo_value(self.audio_source),
            "sound_output": self.combo_value(self.sound_output),
        }

    def save(self) -> bool:
        try:
            values = self._values()
            Config.from_mapping(asdict(Config.load()) | values).save()
        except (OSError, ValueError, TypeError) as exc:
            self.message.setText(str(exc))
            return False
        fields = getattr(self.settings, "fields", {})
        for key, value in values.items():
            sync_field(fields.get(key), value)
        self.message.setText(
            "Saved; priority applies next recording. Desktop defaults are unchanged."
        )
        return True

    def update_status(self, status: Mapping[str, object]) -> None:
        recording = status.get("phase") == "recording"
        self.meter.setValue(_meter_percent(status) if recording else 0)
        source = status.get("microphone")
        name = source if isinstance(source, str) else "default"
        self.current_source.setText("Active input: " + name if recording else "No active capture")
        for control in (
            *self.groups.values(),
            *self.edit_controls,
            self.sound_output,
            self.save_button,
        ):
            control.setEnabled(not recording)
