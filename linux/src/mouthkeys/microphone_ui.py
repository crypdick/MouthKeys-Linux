"""Microphone priority editor and capture-only input meter; never opens audio."""

import json
import math
import subprocess
from dataclasses import replace

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
from .delivery import session
from .desktop import source_list

GROUPS = {
    "microphone_preferred": "Preferred",
    "microphone_fallback": "Last resort",
    "microphone_never": "Never",
}


def discover(kind):
    if session() == "headless":
        return []
    result = subprocess.run(
        ["pactl", "-f", "json", "list", kind], check=True, capture_output=True, text=True, timeout=2
    )
    values = json.loads(result.stdout)
    return [
        (item["name"], item.get("description", item["name"]))
        for item in values
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def sync_field(field, value):
    if isinstance(field, QLineEdit):
        field.setText(value)
    elif isinstance(field, QComboBox):
        if field.findData(value) < 0:
            field.addItem(value, value)
        field.setCurrentIndex(field.findData(value))


class MicrophonePage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.active_list = None
        self.sources = {}
        self.groups = {}
        self.edit_controls = []
        config = Config.load()
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
        self.current_source = QLabel("No active capture")
        root.addWidget(self.current_source)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setAccessibleName("Input level from active recording only")
        self.meter.setFormat("Input level %p%")
        root.addWidget(self.meter)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Audio source"))
        self.audio_source = QComboBox()
        self.audio_source.setEditable(True)
        self.audio_source.addItem("Desktop default", "default")
        if config.audio_source != "default":
            self.audio_source.addItem(config.audio_source, config.audio_source)
            self.audio_source.setCurrentIndex(1)
        source_row.addWidget(self.audio_source, 1)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh)
        source_row.addWidget(refresh)
        root.addLayout(source_row)
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Source to add"))
        self.add_picker = QComboBox()
        self.add_picker.setEditable(True)
        self.add_picker.addItem("Choose input to add", "")
        add_row.addWidget(self.add_picker, 1)
        root.addLayout(add_row)
        self.edit_controls.extend([self.audio_source, self.add_picker])
        columns = QHBoxLayout()
        for key, title in GROUPS.items():
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            group = QListWidget()
            group.setAccessibleName(title + " microphone priority")
            group.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            group.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
            group.setDefaultDropAction(Qt.DropAction.MoveAction)
            group.setDragEnabled(True)
            group.setAcceptDrops(True)
            group.setDropIndicatorShown(True)
            group.itemSelectionChanged.connect(lambda group=group: self.select_group(group))
            for name in source_list(getattr(config, key)):
                self.add_item(group, name)
            self.groups[key] = group
            column.addWidget(group)
            add = QPushButton("Add source here")
            add.clicked.connect(lambda checked=False, key=key: self.add_source(key))
            self.edit_controls.append(add)
            column.addWidget(add)
            columns.addLayout(column)
        root.addLayout(columns, 1)
        row = QHBoxLayout()
        for title, callback in (
            ("Move up", lambda: self.reorder(-1)),
            ("Move down", lambda: self.reorder(1)),
            ("Remove from priority", self.remove_selected),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            self.edit_controls.append(button)
            row.addWidget(button)
        root.addLayout(row)
        row = QHBoxLayout()
        for key, title in GROUPS.items():
            button = QPushButton("Move to " + title)
            button.clicked.connect(lambda checked=False, key=key: self.move_selected(key))
            self.edit_controls.append(button)
            row.addWidget(button)
        root.addLayout(row)
        output = QHBoxLayout()
        output.addWidget(QLabel("Sound output"))
        self.sound_output = QComboBox()
        self.sound_output.setEditable(True)
        self.sound_output.addItem("Desktop default", "default")
        if config.sound_output != "default":
            self.sound_output.addItem(config.sound_output, config.sound_output)
            self.sound_output.setCurrentIndex(1)
        output.addWidget(self.sound_output, 1)
        root.addLayout(output)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        self.save_button = QPushButton("Save microphone settings")
        self.save_button.clicked.connect(self.save)
        root.addWidget(self.save_button)
        for combo in (self.audio_source, self.add_picker, self.sound_output):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(20)
        self.refresh()

    def add_item(self, group, name):
        description = self.sources.get(name)
        text = f"{description} — {name}" if description and description != name else name
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setToolTip(name)
        group.addItem(item)
        return item

    def select_group(self, group):
        if group.selectedItems():
            self.active_list = group
            for other in self.groups.values():
                if other is not group:
                    other.clearSelection()

    def refresh(self):
        try:
            sources, sinks = discover("sources"), discover("sinks")
            self.sources = dict(sources)
            for combo, values in ((self.audio_source, sources), (self.sound_output, sinks)):
                previous = self.combo_value(combo)
                combo.clear()
                combo.addItem("Desktop default", "default")
                for name, description in values:
                    if combo.findData(name) < 0:
                        combo.addItem(
                            f"{description} — {name}" if description != name else name, name
                        )
                if combo.findData(previous) < 0:
                    combo.addItem(previous, previous)
                combo.setCurrentIndex(combo.findData(previous))
            previous = self.combo_value(self.add_picker)
            self.add_picker.clear()
            self.add_picker.addItem("Choose input to add", "")
            for name, description in sources:
                if self.add_picker.findData(name) < 0:
                    self.add_picker.addItem(
                        f"{description} — {name}" if description != name else name, name
                    )
            if previous and self.add_picker.findData(previous) < 0:
                self.add_picker.addItem(previous, previous)
            self.add_picker.setCurrentIndex(max(0, self.add_picker.findData(previous)))
            self.message.setText(
                "Device discovery is read-only; input meter stays empty while idle."
            )
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            self.message.setText(
                f"Device discovery unavailable: {exc}. Custom source/sink names still work."
            )

    @staticmethod
    def combo_value(combo):
        index = combo.currentIndex()
        return (
            combo.itemData(index)
            if index >= 0 and combo.currentText() == combo.itemText(index)
            else combo.currentText().strip()
        )

    def add_source(self, key):
        name = self.combo_value(self.add_picker)
        if not name or name == "default" or "," in name or "\n" in name or len(name) > 512:
            self.message.setText(
                "Choose or enter an actual source name; desktop default is not a priority entry."
            )
            return
        for group in self.groups.values():
            for index in reversed(range(group.count())):
                if group.item(index).data(Qt.ItemDataRole.UserRole) == name:
                    group.takeItem(index)
        item = self.add_item(self.groups[key], name)
        self.groups[key].setCurrentItem(item)

    def reorder(self, direction):
        group = self.active_list
        if group is None or group.currentRow() < 0:
            return
        index = group.currentRow()
        destination = index + direction
        if 0 <= destination < group.count():
            item = group.takeItem(index)
            group.insertItem(destination, item)
            group.setCurrentItem(item)

    def move_selected(self, key):
        group = self.active_list
        if group is None or group.currentRow() < 0:
            return
        item = group.takeItem(group.currentRow())
        self.groups[key].addItem(item)
        self.groups[key].setCurrentItem(item)

    def remove_selected(self):
        if self.active_list is not None and self.active_list.currentRow() >= 0:
            self.active_list.takeItem(self.active_list.currentRow())

    def save(self):
        try:
            values = {}
            seen = set()
            for key, group in self.groups.items():
                names = []
                for index in range(group.count()):
                    name = group.item(index).data(Qt.ItemDataRole.UserRole)
                    if (
                        not isinstance(name, str)
                        or not name
                        or name in seen
                        or "," in name
                        or name == "default"
                        or "\n" in name
                        or "\r" in name
                        or len(name) > 512
                    ):
                        raise ValueError(
                            "Each microphone can appear only once across priority groups"
                        )
                    seen.add(name)
                    names.append(name)
                values[key] = ",".join(names)
            values["audio_source"] = self.combo_value(self.audio_source)
            values["sound_output"] = self.combo_value(self.sound_output)
            replace(Config.load(), **values).save()
            for key, value in values.items():
                field = getattr(self.window, "fields", {}).get(key)
                if field is not None:
                    sync_field(field, value)
            self.message.setText(
                "Saved; priority applies next recording. Desktop defaults are unchanged."
            )
            return True
        except (OSError, ValueError, TypeError) as exc:
            self.message.setText(str(exc))
            return False

    def update_status(self, status):
        recording = status.get("phase") == "recording"
        level = status.get("meter", {}).get("level", 0) if recording else 0
        if not isinstance(level, int | float) or not math.isfinite(level):
            level = 0
        self.meter.setValue(round(min(1, max(0, level)) * 100))
        self.current_source.setText(
            "Active input: " + status.get("microphone", "default")
            if recording
            else "No active capture"
        )
        for group in self.groups.values():
            group.setEnabled(not recording)
        for control in self.edit_controls:
            control.setEnabled(not recording)
        self.sound_output.setEnabled(not recording)
        self.save_button.setEnabled(not recording)
