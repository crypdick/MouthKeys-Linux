"Optional Qt desktop controller. The daemon owns audio and never opens windows."

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .config import CHOICES, RANGES, Config, state_dir
from .processes import python_command
from .service import ensure_daemon, ensure_overlay, request

COMPUTE_TYPES = [
    "default",
    "auto",
    "int8",
    "float32",
    "float16",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "bfloat16",
    "int16",
]

MODEL_ALIASES = {"large": "large-v3", "turbo": "large-v3-turbo"}
# Approximate accuracy for short English dictation, strongest first. Language and
# audio can change the ranking; .en and distilled models only support English.
# See https://huggingface.co/distil-whisper/distil-large-v3.5 for comparisons.
MODEL_ACCURACY_ORDER = (
    "large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "distil-large-v3",
    "large-v2",
    "distil-large-v2",
    "large-v1",
    "medium.en",
    "medium",
    "distil-medium.en",
    "small.en",
    "small",
    "distil-small.en",
    "base.en",
    "base",
    "tiny.en",
    "tiny",
)


def model_choices():
    from faster_whisper import available_models

    available = {MODEL_ALIASES.get(name, name) for name in available_models()}
    ranked = [name for name in MODEL_ACCURACY_ORDER if name in available]
    return ranked + sorted(available.difference(MODEL_ACCURACY_ORDER))


def language_choices():
    # Match the tokenizer in our installed, version-bounded transcription engine.
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    names = {"haw": "Hawaiian", "ht": "Haitian Creole", "jw": "Javanese", "yue": "Cantonese"}
    choices = []
    for code in _LANGUAGE_CODES:
        locale = QLocale(code)
        name = names.get(code, QLocale.languageToString(locale.language()))
        choices.append((f"{name if name != 'C' else code} ({code})", code))
    return [("Detect automatically", "auto"), *sorted(choices)]


def audio_sources():
    import subprocess

    from .delivery import session

    if session() == "headless":
        return ["default"]
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return [
            "default",
            *[line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2],
        ]
    except (OSError, subprocess.SubprocessError):
        return ["default"]


def field_value(field):
    if isinstance(field, QSpinBox):
        return field.value()
    if isinstance(field, QLineEdit):
        return field.text()
    return field.currentText() if field.isEditable() else field.currentData()


def setting_group(key):
    if key.startswith("api_"):
        return "Local API"
    if key.startswith("ai_"):
        return "AI cleanup"
    if key.startswith("spoken_send") or key == "send_after_pause":
        return "Spoken Send"
    if key.startswith(("microphone_", "sound_")) or key in ("audio_source", "pause_media"):
        return "Microphone & media"
    if key.startswith(("paste_", "clipboard_")) or key in (
        "delivery",
        "insertion_mode",
        "restore_clipboard",
        "copy_to_clipboard",
    ):
        return "Text delivery"
    if key.startswith("overlay_") or key in (
        "recording_panel",
        "theme",
        "preview_chars",
        "visualizer_sensitivity",
        "live_preview",
    ):
        return "Appearance"
    if key.startswith("save_") or key in ("audio_budget_mb", "typing_wpm", "weekends_break_streak"):
        return "History & stats"
    if key in (
        "tray",
        "launch_at_startup",
        "show_window_at_login",
        "show_microphone_alerts",
        "show_paste_alerts",
    ):
        return "Desktop"
    if key in (
        "model",
        "language",
        "device",
        "compute_type",
        "engine",
        "max_seconds",
        "skip_silent_recordings",
        "queue_limit",
        "file_speaker_labels",
        "file_speaker_count",
    ):
        return "Speech"
    return "Text formatting"


class Request(QThread):
    done = Signal(object)

    def __init__(self, action, data=None, parent=None):
        super().__init__(parent)
        self.action, self.data = action, data or {}

    def run(self):
        try:
            self.done.emit({"result": request(self.action, **self.data)})
        except Exception as exc:
            self.done.emit({"error": str(exc)})


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MouthKeys")
        icon_path = str(Path(__file__).parent / "assets" / "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1000, 680)
        self.setMinimumSize(820, 560)
        self.jobs = set()
        self.polling = False
        self.rows = []
        self.download = None
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        navigation = QVBoxLayout(sidebar)
        navigation.setContentsMargins(20, 28, 20, 20)
        navigation.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(
            QPixmap(icon_path).scaled(
                64,
                64,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        navigation.addWidget(mark)
        brand = QLabel("MOUTHKEYS")
        brand.setObjectName("brand")
        navigation.addWidget(brand)
        tagline = QLabel("WICKED FAST VOICE TO TEXT")
        tagline.setObjectName("placard")
        navigation.addWidget(tagline)
        navigation.addSpacing(28)
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        self.nav.addItems(
            [
                "History",
                "App Settings",
                "Custom Dictionary",
                "File Transcription",
                "Stats",
                "Backup & Audio",
                "Help & Diagnostics",
                "AI Prompts",
                "Command / Edit",
            ]
        )
        navigation.addWidget(self.nav)
        footer = QLabel("LOCAL DICTATION  /  LINUX")
        footer.setObjectName("placard")
        navigation.addWidget(footer)
        root.addWidget(sidebar)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(16)
        self.page_title = QLabel("History")
        self.page_title.setObjectName("pageTitle")
        layout.addWidget(self.page_title)
        root.addWidget(content, 1)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #ff4f1f; padding: 12px 0")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.toggle = QPushButton("Record / Stop")
        self.toggle.clicked.connect(lambda: self.send("toggle"))
        buttons.addWidget(self.toggle)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(lambda: self.send("cancel"))
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        hint = QLabel(
            "Bind a desktop shortcut to mouthkeys toggle to dictate without moving focus.\n"
            "On Wayland, finish speaking, then paste with Ctrl+V (Ctrl+Shift+V in terminals)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        tabs = QStackedWidget()
        self.nav.currentRowChanged.connect(tabs.setCurrentIndex)
        self.nav.currentTextChanged.connect(self.page_title.setText)
        layout.addWidget(tabs)
        history = QWidget()
        history_layout = QVBoxLayout(history)
        self.history_limit = 100
        self.history_request = None
        self.history_serial = 0
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search transcripts, applications, and windows")
        self.search.textChanged.connect(lambda: self.refresh_history())
        history_layout.addWidget(self.search)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.currentRowChanged.connect(self.select)
        history_split = QHBoxLayout()
        history_split.addWidget(self.list, 1)
        history_layout.addLayout(history_split, 1)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        history_split.addWidget(self.preview, 2)
        self.preview.setPlaceholderText(
            "No History Yet\n\nYour transcriptions will appear here. Start dictating to begin."
        )
        row = QHBoxLayout()
        for text, action in (("Copy", "copy"), ("Reprocess audio", "reprocess")):
            button = QPushButton(text)
            button.clicked.connect(
                lambda checked=False, action=action: self.selected_action(action)
            )
            row.addWidget(button)
        clear = QPushButton("Clear history")
        clear.clicked.connect(self.clear_history)
        row.addWidget(clear)
        delete = QPushButton("Delete selected")
        delete.clicked.connect(self.delete_selected)
        row.addWidget(delete)
        history_layout.addLayout(row)
        extras = QHBoxLayout()
        for text, handler in (
            ("Load more", self.load_more),
            ("Open audio", self.open_audio),
            ("Export transcript + audio", self.export_pair),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            extras.addWidget(button)
        history_layout.addLayout(extras)
        tabs.addWidget(history)
        settings = QWidget()
        form = QFormLayout(settings)
        self.app_version = QLabel(__version__)
        self.app_version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("App version", self.app_version)
        self.service_version = QLabel("Connecting…")
        self.service_version.setWordWrap(True)
        self.service_version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Recording service version", self.service_version)
        self.settings_category = QComboBox()
        self.settings_category.addItems(
            [
                "All settings",
                "Speech",
                "Microphone & media",
                "Text delivery",
                "Spoken Send",
                "Text formatting",
                "AI cleanup",
                "Appearance",
                "History & stats",
                "Desktop",
                "Local API",
            ]
        )
        form.addRow("Category", self.settings_category)
        self.settings_rows = []
        self.settings_form = form
        self.fields = {}
        config = asdict(Config.load())

        choices = {
            **CHOICES,
            "model": model_choices(),
            "language": language_choices(),
            "compute_type": COMPUTE_TYPES,
            "audio_source": audio_sources(),
            "device": ["cpu", "cuda", "auto"],
            "recording_panel": [("Show panel", "on"), ("Hide panel", "off")],
            "delivery": ["clipboard", "x11", "wayland"],
            "paste_backend": ["auto", "xdotool", "wtype", "ydotool"],
            "paste_shortcut": ["ctrl+shift+v", "ctrl+v"],
        }
        from .profiles import load_profiles

        choices["ai_profile"] = [("Default settings", "")] + [
            (profile["name"], profile["id"]) for profile in load_profiles()
        ]
        labels = {
            "max_seconds": "Maximum recording length",
            "audio_budget_mb": "Audio storage budget",
            "recording_panel": "Recording panel",
            "ai_key_env": "AI API key environment variable",
            "ai_endpoint": "AI cleanup endpoint (full URL)",
            "spoken_send": "Spoken Send",
            "send_after_pause": "Send after a pause",
            "spoken_send_key": "Send key",
            "save_history": "Save transcription history",
            "save_audio": "Save all history audio",
            "microphone_preferred": "Preferred microphones (ordered, comma-separated)",
            "microphone_fallback": "Last-resort microphones (ordered, comma-separated)",
            "microphone_never": "Never use microphones (comma-separated)",
            "typing_wpm": "Typing speed for time-saved estimate (WPM)",
            "file_speaker_count": "File speaker count (0 = automatic)",
        }
        help_text = {
            "max_seconds": (
                "At the limit, recording stops and text is copied without automatic paste."
            ),
            "audio_budget_mb": (
                "Oldest recovery audio, including failures, is deleted above this budget. "
                "Transcripts stay. 1,000 MB = 1 GB."
            ),
            "recording_panel": (
                "Hide the panel if your compositor tiles it or takes keyboard focus. "
                "Dictation still works; the panel also shows loading and shortcut errors."
            ),
            "spoken_send": (
                "Opt-in: send configured key only after automatic insertion into "
                "an unchanged X11 destination."
            ),
            "delivery": (
                "Wayland auto-paste requires Sway/Hyprland focus checks and wtype/ydotool; "
                "compositor support varies."
            ),
            "ai_enabled": (
                "When on, recorded transcript is sent to the configured local or "
                "remote AI endpoint."
            ),
            "ai_key_env": (
                "Set this variable in the recording service environment; API keys "
                "are never stored in backups."
            ),
            "save_history": (
                "Off removes completed attempts after delivery; failed raw text is"
                " not retained either."
            ),
            "save_audio": (
                "Off retains only recovery audio and latest successful recording; "
                "on keeps audio within budget."
            ),
            "restore_clipboard": (
                "Restoration occurs only after supported automatic paste and only "
                "if clipboard ownership is unchanged."
            ),
            "api_enabled": (
                "Opt-in loopback HTTP API; requires a token in the configured "
                "environment variable. Restart service after changes."
            ),
        }
        for key, value in config.items():
            # faster-whisper gives existing local directories precedence over aliases.
            if key == "model" and value in MODEL_ALIASES and not Path(value).is_dir():
                value = MODEL_ALIASES.get(value, value)
            if key in RANGES:
                field = QSpinBox()
                field.setRange(*RANGES[key])
                if key in ("max_seconds", "audio_budget_mb"):
                    field.setSuffix(" seconds" if key == "max_seconds" else " MB")
                field.setValue(value)
            elif key in choices:
                field = QComboBox()
                for option in choices[key]:
                    label, data = option if isinstance(option, tuple) else (option, option)
                    field.addItem(label, data)
                # Preserve custom model paths, microphone names, and prior CLI settings.
                if field.findData(value) < 0:
                    field.addItem(value, value)
                field.setCurrentIndex(field.findData(value))
                if key in ("model", "audio_source"):
                    field.setEditable(True)
                    field.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            else:
                field = QLineEdit(value)
            self.fields[key] = field
            self.settings_rows.append((form.rowCount(), setting_group(key)))
            form.addRow(labels.get(key, key.replace("_", " ").capitalize()), field)
            if key in help_text:
                field.setToolTip(help_text[key])
                hint = QLabel(help_text[key])
                hint.setWordWrap(True)
                self.settings_rows.append((form.rowCount(), setting_group(key)))
                form.addRow("", hint)
        self.fields["engine"].currentTextChanged.connect(self.refresh_model_choices)
        if config["engine"] != "whisper":
            self.refresh_model_choices(config["engine"])
        self.models_info = QLabel(
            "Download only on request. Install optional engine extras before switching engines."
        )
        self.models_info.setWordWrap(True)
        self.settings_rows.append((form.rowCount(), "Speech"))
        form.addRow(self.models_info)
        catalog_button = QPushButton("Inspect model catalog / installed state")
        catalog_button.clicked.connect(self.inspect_models)
        self.settings_rows.append((form.rowCount(), "Speech"))
        form.addRow(catalog_button)
        self.fields["model"].setToolTip(
            "Approximate transcription accuracy, strongest first (English dictation).\n"
            "Accuracy varies by language and audio; .en and distilled models "
            "are English-only.\n"
            "Choose a model or enter a local model path / Hugging Face ID."
        )
        self.fields["audio_source"].setToolTip(
            "Choose a microphone or enter a PulseAudio source name. Default follows your desktop."
        )
        self.fields["compute_type"].setToolTip(
            "default follows the model's stored precision, with hardware fallbacks.\n"
            "auto selects the fastest supported type for your device.\n"
            "int8 reduces memory use; default may use more memory and run slower on CPU."
        )
        save = QPushButton("Save settings")
        save.clicked.connect(self.save)
        form.addRow(save)
        self.download_button = QPushButton("Download selected model")
        self.download_button.clicked.connect(self.download_model)
        self.settings_rows.append((form.rowCount(), "Speech"))
        form.addRow(self.download_button)
        for label, handler in (
            ("Cancel model download", self.cancel_download),
            ("Delete selected cached model", self.delete_model),
            ("Download speaker models", self.download_speakers),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            self.settings_rows.append((form.rowCount(), "Speech"))
            form.addRow(button)
        warm = QPushButton("Load model into memory")
        warm.clicked.connect(lambda: self.send("warmup"))
        self.settings_rows.append((form.rowCount(), "Speech"))
        form.addRow(warm)
        self.settings_category.currentTextChanged.connect(self.filter_settings)
        self.settings_category.setCurrentText("Speech")
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)
        tabs.addWidget(settings_scroll)
        tabs.addWidget(self.dictionary_page())
        tabs.addWidget(self.file_page())
        tabs.addWidget(self.stats_page())
        tabs.addWidget(self.backup_page())
        tabs.addWidget(self.help_page())
        tabs.addWidget(self.profiles_page())
        from .assistant_ui import AssistantPage

        self.assistant_page = AssistantPage(self)
        tabs.addWidget(self.assistant_page)
        self.nav.setCurrentRow(0)
        self.apply_theme()
        QApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(700)
        self.poll()

    def refresh_history(self):
        query = (self.search.text(), self.history_limit)
        if self.history_request == query:
            return
        self.history_request = query
        self.history_serial += 1
        serial = self.history_serial

        def update(response):
            if self.history_serial == serial:
                self.history_request = None
                if query == (self.search.text(), self.history_limit):
                    self.update_history(response)

        self.send(
            "history",
            {"query": query[0], "limit": query[1]},
            callback=update,
        )

    def filter_settings(self, category):
        for row, group in self.settings_rows:
            self.settings_form.setRowVisible(row, category == "All settings" or group == category)

    def refresh_model_choices(self, engine):
        field = self.fields["model"]
        current = field_value(field)
        if engine == "whisper":
            names = model_choices()
        else:
            from .engines import model_catalog

            names = [entry["model"] for entry in model_catalog(engine)]
        field.clear()
        for name in names:
            field.addItem(name, name)
        if current and current not in names:
            field.addItem(current, current)
        field.setCurrentIndex(field.findData(current))

    def inspect_models(self):
        from .engines import model_catalog

        try:
            catalog = model_catalog(field_value(self.fields["engine"]))
            lines = [
                f"{entry['name']} · {entry['languages']} · "
                f"{'Installed' if entry['installed'] else 'Not downloaded'}"
                for entry in catalog
            ]
            QMessageBox.information(self, "Speech models", "\n".join(lines))
        except Exception as exc:
            self.status.setText(str(exc))

    def load_more(self):
        self.history_limit = min(10000, self.history_limit + 100)
        self.refresh_history()

    def selected_id(self):
        index = self.list.currentRow()
        return self.rows[index]["id"] if 0 <= index < len(self.rows) else None

    def delete_selected(self):
        item_id = self.selected_id()
        if (
            item_id
            and QMessageBox.question(
                self, "Delete transcript", "Delete selected transcript and audio?"
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.send("delete", {"id": item_id}, callback=lambda _: self.refresh_history())

    def open_audio(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from .storage import History

        item_id = self.selected_id()
        if item_id:
            path = History().recording(item_id)
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                self.status.setText("No retained audio for this entry")

    def export_pair(self):
        item_id = self.selected_id()
        if item_id:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export transcript and audio", "dictation.zip", "ZIP (*.zip)"
            )
            if path:
                self.send(
                    "export_audio", {"id": item_id, "path": path}, callback=self.operation_result
                )

    def operation_result(self, response):
        self.status.setText(str(response.get("error", response.get("result", "Done"))))

    def dictionary_page(self):
        from .dictionary import Dictionary

        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel(
            "Teach replacements, then customize phrases used after your "
            "Literal prefix. Separate aliases with commas. Disable a rule "
            "without deleting it."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.dictionary_tabs = QTabWidget()
        self.dictionary_tables = {}
        for name in ("replacements", "punctuation", "actions"):
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels(
                ["Spoken phrases (comma-separated)", "Replacement", "Enabled (on/off)"]
            )
            table.horizontalHeader().setStretchLastSection(True)
            table.setColumnWidth(0, 260)
            table.setColumnWidth(1, 160)
            self.dictionary_tables[name] = table
            self.dictionary_tabs.addTab(table, name.capitalize())
        layout.addWidget(self.dictionary_tabs)
        training = QHBoxLayout()
        self.training_replacement = QLineEdit()
        self.training_replacement.setPlaceholderText("Correct word or phrase to teach")
        training.addWidget(self.training_replacement)
        self.training_id = None
        train = QPushButton("Train by voice")
        train.clicked.connect(self.train_dictionary)
        training.addWidget(train)
        stop_training = QPushButton("Stop training")
        stop_training.clicked.connect(lambda: self.send("stop"))
        training.addWidget(stop_training)
        layout.addLayout(training)
        buttons = QHBoxLayout()
        for text, handler in (
            ("Add rule", self.add_dictionary_rule),
            ("Delete rule", self.delete_dictionary_rule),
            ("Save", self.save_dictionary),
            ("Import", self.import_dictionary),
            ("Export", self.export_dictionary),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.fill_dictionary(Dictionary.load())
        return page

    def train_dictionary(self):
        if not self.training_replacement.text().strip():
            self.status.setText("Enter the correct replacement first, then say it aloud")
            return

        def started(response):
            if "result" in response:
                self.training_id = response["result"]["id"]

        self.send("start", {"mode": "training"}, callback=started)

    def update_training(self, result):
        proposal = result.get("assistant_proposal")
        if (
            not self.training_id
            or result.get("id") != self.training_id
            or not proposal
            or proposal.get("mode") != "training"
        ):
            return
        self.training_id = None
        trigger = proposal["text"].strip().rstrip(".!?")
        replacement = self.training_replacement.text().strip()
        if not trigger:
            self.status.setText("No recognized training phrase; try again")
            return
        table = self.dictionary_tables["replacements"]
        row = table.rowCount()
        table.insertRow(row)
        for col, text in enumerate((trigger, replacement, "on")):
            table.setItem(row, col, QTableWidgetItem(text))
        self.dictionary_tabs.setCurrentIndex(0)
        self.status.setText("Training captured. Review recognized phrase and Save dictionary.")

    def fill_dictionary(self, data):
        for name, table in self.dictionary_tables.items():
            table.setRowCount(0)
            for entry in data[name]:
                row = table.rowCount()
                table.insertRow(row)
                aliases = entry["triggers" if name == "replacements" else "aliases"]
                value = entry["replacement" if name == "replacements" else "symbol"]
                # Escaped control characters keep newline/tab rules editable in one cell.
                if name != "replacements":
                    value = value.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
                for column, text in enumerate(
                    (", ".join(aliases), value, "on" if entry.get("enabled", True) else "off")
                ):
                    table.setItem(row, column, QTableWidgetItem(text))

    def add_dictionary_rule(self):
        table = self.dictionary_tabs.currentWidget()
        row = table.rowCount()
        table.insertRow(row)
        for col, text in enumerate(("", "", "on")):
            table.setItem(row, col, QTableWidgetItem(text))
        table.setCurrentCell(row, 0)
        table.editItem(table.item(row, 0))

    def delete_dictionary_rule(self):
        table = self.dictionary_tabs.currentWidget()
        if table.currentRow() >= 0:
            table.removeRow(table.currentRow())

    def save_dictionary(self):
        from .dictionary import Dictionary

        try:
            data = {}
            for name, table in self.dictionary_tables.items():
                entries = []
                for row in range(table.rowCount()):
                    values = [
                        table.item(row, col).text() if table.item(row, col) else ""
                        for col in range(3)
                    ]
                    if values[2] not in ("on", "off"):
                        raise ValueError("Enabled must be on or off")
                    value = values[1]
                    if name != "replacements":
                        import re

                        value = re.sub(
                            r"\\([\\nt])", lambda m: {"n": "\n", "t": "\t", "\\": "\\"}[m[1]], value
                        )
                    entries.append(
                        {
                            "triggers" if name == "replacements" else "aliases": [
                                v.strip() for v in values[0].split(",") if v.strip()
                            ],
                            "replacement" if name == "replacements" else "symbol": value,
                            "enabled": values[2] == "on",
                        }
                    )
                data[name] = entries
            Dictionary.save(data)
            self.status.setText("Dictionary saved")
            return True
        except Exception as exc:
            self.status.setText(str(exc))
            return False

    def import_dictionary(self):
        from .dictionary import Dictionary

        path, _ = QFileDialog.getOpenFileName(self, "Import dictionary", "", "JSON (*.json)")
        if (
            path
            and QMessageBox.question(self, "Import dictionary", "Replace current dictionary?")
            == QMessageBox.StandardButton.Yes
        ):
            try:
                self.fill_dictionary(Dictionary.import_file(path))
            except Exception as exc:
                self.status.setText(str(exc))

    def export_dictionary(self):
        from .dictionary import Dictionary

        path, _ = QFileDialog.getSaveFileName(
            self, "Export dictionary", "dictionary.json", "JSON (*.json)"
        )
        if path and self.save_dictionary():
            Dictionary.export_file(path)

    def file_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Choose an audio file or drop one here")
        layout.addWidget(self.file_path)
        buttons = QHBoxLayout()
        for label, handler in (
            ("Choose file", self.choose_file),
            ("Transcribe", self.transcribe_file),
            ("Cancel", self.cancel_file),
            ("Export text", lambda: self.export_file_result("text")),
            ("Export JSON", lambda: self.export_file_result("json")),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.file_result = None
        self.file_process = None
        self.file_text = QTextEdit()
        self.file_text.setReadOnly(True)
        self.file_text.setPlaceholderText(
            "File transcription stays separate from dictation. Speaker labels "
            "and expected count are configured in App Settings. Models must be"
            " downloaded explicitly."
        )
        layout.addWidget(self.file_text)
        self.setAcceptDrops(True)
        return page

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.file_path.setText(urls[0].toLocalFile())
            self.nav.setCurrentRow(3)
            event.acceptProposedAction()

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Transcribe audio", "", "Audio/video files (*)")
        if path:
            self.file_path.setText(path)

    def transcribe_file(self):
        if self.file_process is not None or not self.save():
            return
        if not Path(self.file_path.text()).is_file():
            self.status.setText("Choose an existing audio file")
            return
        self.file_result = None
        self.file_process = QProcess(self)
        self.file_process.setProgram(sys.executable)
        self.file_process.setArguments(
            python_command("mouthkeys", "transcribe", self.file_path.text(), "--json")[1:]
        )
        self.file_text.setPlainText("Transcribing…")
        self.file_process.finished.connect(self.file_finished)
        self.file_process.errorOccurred.connect(self.file_process_error)
        self.file_process.start()

    def file_process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.file_process is not None:
            self.file_text.setPlainText(self.file_process.errorString())
            self.file_process.deleteLater()
            self.file_process = None

    def file_finished(self, code, _):
        process, self.file_process = self.file_process, None
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode(errors="replace")
        error = bytes(process.readAllStandardError()).decode(errors="replace")
        process.deleteLater()
        try:
            if code:
                raise ValueError(error or "File transcription cancelled or failed")
            self.file_result = json.loads(output)
            result = self.file_result
            segments = result.get("segments", [])
            lines = [
                f"{s['start']:.1f}–{s['end']:.1f}  {s.get('speaker', '')}  {s['text']}"
                for s in segments
            ]
            self.file_text.setPlainText(result["text"] + "\n\n" + "\n".join(lines))
            self.status.setText(
                f"{result['duration']:.1f}s audio · {result['processing_seconds']:.1f}s processing"
            )
        except Exception as exc:
            self.file_result = None
            self.file_text.setPlainText(str(exc))

    def cancel_file(self):
        if self.file_process is not None:
            self.file_process.kill()

    def export_file_result(self, kind):
        if self.file_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export transcription", "transcript." + ("json" if kind == "json" else "txt")
        )
        if path:
            from .file_transcription import export_result

            export_result(self.file_result, path, "json" if kind == "json" else "txt")

    def stats_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        refresh = QPushButton("Refresh stats")
        refresh.clicked.connect(self.refresh_stats)
        layout.addWidget(refresh)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText(
            "Refresh to show local dictation activity. Typing speed and "
            "weekend streak rules are configurable in App Settings."
        )
        layout.addWidget(self.stats_text)
        return page

    def refresh_stats(self):
        def update(response):
            if "result" not in response:
                return
            result = response["result"]
            lines = [
                f"{result['words']:,} words  ·  {result['transcriptions']:,} dictations",
                f"{result['current_streak']} day streak  ·  "
                f"{result['time_saved_seconds'] / 60:.1f} minutes saved",
                "",
            ]
            for name, tally in result["windows"].items():
                lines.append(
                    f"{name}: {tally['words']:,} words / {tally['transcriptions']:,} dictations"
                )
            lines.extend(
                [
                    "",
                    f"Longest dictation: {result['longest_words']:,} words",
                    f"Most words in one day: {result['most_words_in_day']:,}",
                    f"Average length: {result['average_words']} words",
                    "",
                    "Daily words:",
                ]
            )
            lines.extend(f"{day}: {words:,}" for day, words in result["daily_words"].items())
            lines.extend(
                ["", f"Top apps: {result['top_apps']}", f"Peak hours: {result['peak_hour']}"]
            )
            self.stats_text.setPlainText("\n".join(lines))

        self.send("stats", callback=update)

    def backup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Back up settings, dictionary, and transcript history. API keys "
            "are excluded. Restore merges history by UUID; it does not delete "
            "existing recordings. Audio exports are separate ZIP files."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        for text, handler in (
            ("Export backup", lambda: self.backup_action(False)),
            ("Import backup", lambda: self.backup_action(True)),
            ("Export all audio + transcripts", self.export_all_audio),
            ("Delete all saved audio", self.delete_all_audio),
            ("Refresh audio usage", self.audio_usage),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            layout.addWidget(button)
        self.audio_usage_label = QLabel("Audio usage: refresh to inspect")
        layout.addWidget(self.audio_usage_label)
        layout.addStretch()
        return page

    def audio_usage(self):
        def update(response):
            if "result" in response:
                value = response["result"]
                self.audio_usage_label.setText(
                    f"{value['bytes'] / 1000000:.1f} MB / "
                    f"{Config.load().audio_budget_mb} MB budget · {value['recordings']} recordings"
                )

        self.send("audio_usage", callback=update)

    def backup_action(self, importing):
        if importing:
            path, _ = QFileDialog.getOpenFileName(self, "Import backup", "", "JSON (*.json)")
            if (
                not path
                or QMessageBox.question(
                    self,
                    "Restore backup",
                    (
                        "Replace settings and dictionary, and merge transcript history? "
                        "Reopen Settings after restore."
                    ),
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export backup", "mouthkeys-backup.json", "JSON (*.json)"
            )
        if path:
            self.send(
                "backup_import" if importing else "backup_export",
                {"path": path},
                callback=self.operation_result,
            )

    def export_all_audio(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", "mouthkeys-audio.zip", "ZIP (*.zip)"
        )
        if path:
            self.send("export_audio", {"path": path}, callback=self.operation_result)

    def delete_all_audio(self):
        if (
            QMessageBox.question(
                self, "Delete saved audio", "Delete all audio while keeping transcripts?"
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.send("delete_audio", callback=lambda _: self.audio_usage())

    def help_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Bind mouthkeys toggle / start / stop / cancel / paste / reprocess"
            " in i3 or desktop keyboard settings. Hold-to-talk can use "
            "separate key-press start and key-release stop bindings. Optional "
            "speech engines need their package extras and explicit model "
            "downloads. Open README for compositor-specific setup."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        for label, handler in (
            ("Show diagnostic log", self.show_log),
            ("Report issue", self.report_issue),
            ("Open Linux guide", self.open_guide),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            layout.addWidget(button)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return page

    def show_log(self):
        path = state_dir() / "daemon.log"
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 100000))
                self.log_text.setPlainText(stream.read().decode(errors="replace"))
        else:
            self.log_text.setPlainText("No diagnostic log yet")

    def report_issue(self):
        import platform
        from urllib.parse import urlencode

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        url = "https://github.com/crypdick/MouthKeys-Linux/issues/new?" + urlencode(
            {"body": f"MouthKeys {__version__}\nLinux {platform.release()}\n\nDescribe issue:\n"}
        )
        QDesktopServices.openUrl(QUrl(url))

    def open_guide(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(
            QUrl("https://github.com/crypdick/MouthKeys-Linux/blob/main/linux/README.md")
        )

    def cancel_download(self):
        if self.download is not None:
            self.download.kill()

    def delete_model(self):
        from .engines import delete_model

        model = field_value(self.fields["model"])
        if (
            QMessageBox.question(
                self,
                "Delete model",
                f"Delete cached model {model}? Active model memory is released on service restart.",
            )
            == QMessageBox.StandardButton.Yes
        ):
            try:
                delete_model(model, field_value(self.fields["engine"]))
                self.status.setText("Cached model deleted")
            except Exception as exc:
                self.status.setText(str(exc))

    def download_speakers(self):
        if self.download is not None:
            return
        self.download = QProcess(self)
        self.download.setProgram(sys.executable)
        self.download.setArguments(python_command("mouthkeys", "download-speakers")[1:])

        def finished(code, _):
            error = bytes(self.download.readAllStandardError()).decode(errors="replace")
            self.download.deleteLater()
            self.download = None
            self.status.setText(
                "Speaker models downloaded" if code == 0 else error or "Download cancelled"
            )

        self.download.finished.connect(finished)
        self.download.errorOccurred.connect(self.download_error)
        self.download.start()

    def profiles_page(self):
        from .profiles import PROFILE_FIELDS

        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel(
            "Save named cleanup prompts and provider settings. Choose active profile in "
            "App Settings. Enable AI cleanup separately. Secrets stay in environment variables."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.profile_picker = QComboBox()
        layout.addWidget(self.profile_picker)
        form = QFormLayout()
        self.profile_name = QLineEdit()
        form.addRow("Name", self.profile_name)
        self.profile_fields = {}
        defaults = asdict(Config.load())
        for name in PROFILE_FIELDS:
            if name == "ai_prompt":
                field = QTextEdit()
                field.setPlainText(defaults[name])
                field.setMinimumHeight(110)
                field.setMaximumHeight(150)
            elif name in RANGES:
                field = QSpinBox()
                field.setRange(*RANGES[name])
                field.setValue(defaults[name])
            elif name in CHOICES:
                field = QComboBox()
                for value in CHOICES[name]:
                    field.addItem(value, value)
                field.setCurrentIndex(field.findData(defaults[name]))
            else:
                field = QLineEdit(defaults[name])
            form.addRow(name.removeprefix("ai_").replace("_", " ").capitalize(), field)
            self.profile_fields[name] = field
        layout.addLayout(form)
        buttons = QHBoxLayout()
        for label, handler in (
            ("Use provider endpoint", self.profile_endpoint),
            ("Save profile", self.save_profile),
            ("Delete profile", self.delete_profile),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.profile_picker.currentIndexChanged.connect(self.select_profile)
        self.refresh_profiles()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def refresh_profiles(self, selected=None):
        from .profiles import load_profiles

        self.profile_picker.blockSignals(True)
        self.profile_picker.clear()
        self.profile_picker.addItem("New profile", "")
        profiles = load_profiles()
        for profile in profiles:
            self.profile_picker.addItem(profile["name"], profile["id"])
        self.profile_picker.setCurrentIndex(max(0, self.profile_picker.findData(selected or "")))
        self.profile_picker.blockSignals(False)
        active = self.fields["ai_profile"].currentData()
        self.fields["ai_profile"].clear()
        self.fields["ai_profile"].addItem("Default settings", "")
        for profile in profiles:
            self.fields["ai_profile"].addItem(profile["name"], profile["id"])
        self.fields["ai_profile"].setCurrentIndex(
            max(0, self.fields["ai_profile"].findData(active))
        )
        self.select_profile()

    def select_profile(self, *_):
        from .profiles import load_profiles

        selected = self.profile_picker.currentData()
        profile = next((p for p in load_profiles() if p["id"] == selected), None)
        self.profile_name.setText(profile["name"] if profile else "")
        values = profile or asdict(Config.load())
        for name, field in self.profile_fields.items():
            value = values[name]
            if isinstance(field, QTextEdit):
                field.setPlainText(value)
            elif isinstance(field, QLineEdit):
                field.setText(value)
            elif isinstance(field, QSpinBox):
                field.setValue(value)
            else:
                field.setCurrentIndex(field.findData(value))

    def profile_endpoint(self):
        from .ai_providers import PROVIDER_ENDPOINTS

        provider = field_value(self.profile_fields["ai_provider"])
        self.profile_fields["ai_endpoint"].setText(PROVIDER_ENDPOINTS[provider])

    def save_profile(self):
        import uuid

        from .profiles import load_profiles, save_profiles

        try:
            item_id = self.profile_picker.currentData() or str(uuid.uuid4())
            values = {
                name: field.toPlainText() if isinstance(field, QTextEdit) else field_value(field)
                for name, field in self.profile_fields.items()
            }
            profile = {"id": item_id, "name": self.profile_name.text(), **values}
            profiles = [p for p in load_profiles() if p["id"] != item_id]
            save_profiles([*profiles, profile])
            self.refresh_profiles(item_id)
            self.status.setText("Profile saved; select it in App Settings to activate")
        except Exception as exc:
            self.status.setText(str(exc))

    def delete_profile(self):
        from .profiles import load_profiles, save_profiles

        selected = self.profile_picker.currentData()
        if selected and selected == Config.load().ai_profile:
            self.status.setText("Choose and save another active profile before deleting this one")
            return
        if (
            selected
            and QMessageBox.question(self, "Delete prompt", "Delete selected prompt profile?")
            == QMessageBox.StandardButton.Yes
        ):
            save_profiles([p for p in load_profiles() if p["id"] != selected])
            self.refresh_profiles()

    def apply_theme(self, *_):
        preference = Config.load().theme
        light = preference == "light" or (
            preference == "system"
            and QApplication.styleHints().colorScheme() == Qt.ColorScheme.Light
        )
        surface, ink, edge, side, field = (
            ("#ffffff", "#111214", "#111214", "#f2f2f4", "#ffffff")
            if light
            else ("#111214", "#ebebec", "#2c2e33", "#0c0d0f", "#0c0d0f")
        )
        self.setStyleSheet(
            f"QWidget {{ background: {surface}; color: {ink}; font-family: sans-serif;"
            "font-size: 13px; }"
            f"QWidget#sidebar {{ background: {side}; border-right: 1px solid {edge}; }}"
            f"#sidebar QLabel, #navigation {{ background: {side}; font-family: monospace; }}"
            "#brand { font-size: 20px; font-weight: 600; letter-spacing: 2px; }"
            "#placard { font-size: 10px; }"
            "#pageTitle { font-size: 24px; font-weight: 600; }"
            f"QPushButton {{ padding: 8px 12px; border: 1px solid {edge}; border-radius: 0; }}"
            f"QPushButton:hover {{ border-color: {ink}; }}"
            f"QPushButton:pressed {{ background: {ink}; color: {surface}; }}"
            "QPushButton:disabled { color: #777; }"
            f"QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox {{ background: {field};"
            f"border: 1px solid {edge}; padding: 8px; selection-background-color: {ink};"
            f"selection-color: {surface}; }}"
            "QListWidget#navigation { border: 0; padding: 0; }"
            "QListWidget::item { padding: 12px 8px; }"
            f"QListWidget::item:selected {{ background: {ink}; color: {surface}; }}"
        )

    def send(self, action, data=None, callback=None):
        job = Request(action, data, self)
        self.jobs.add(job)

        def done(response):
            if "error" in response:
                self.status.setText(response["error"])
            if callback:
                callback(response)

        job.done.connect(done)
        job.finished.connect(lambda: self.jobs.discard(job))
        job.finished.connect(job.deleteLater)
        job.start()

    def poll(self):
        if self.polling:
            return
        self.polling = True

        def update(response):
            self.polling = False
            if "result" in response:
                result = response["result"]
                service_version = result["version"]
                version_text = service_version
                if service_version != __version__:
                    version_text += " — restart MouthKeys after updating"
                self.service_version.setText(version_text)
                seconds = f"  {result['seconds']:.1f}s" if result["phase"] == "recording" else ""
                self.status.setText(f"{result['phase'].upper()}{seconds} · {result['message']}")
                self.toggle.setEnabled(
                    result["phase"] == "recording"
                    or result.get("can_record", result["phase"] == "idle")
                )
                self.assistant_page.update_status(result)
                self.update_training(result)
                self.refresh_history()
            else:
                self.service_version.setText("Unavailable")

        self.send("status", callback=update)

    def update_history(self, response):
        if "result" not in response or response["result"] == self.rows:
            return
        selected = self.list.currentRow()
        selected_id = self.rows[selected]["id"] if 0 <= selected < len(self.rows) else None
        self.rows = response["result"]
        self.list.clear()
        for row in self.rows:
            date = datetime.fromtimestamp(row["created"]).strftime("%m-%d %H:%M")
            label = f"{date}  /  {row['status'].upper()}\n{row['text'][:65] or row['error'][:65]}"
            self.list.addItem(QListWidgetItem(label))
        index = next((i for i, row in enumerate(self.rows) if row["id"] == selected_id), 0)
        self.list.setCurrentRow(index if self.rows else -1)

    def select(self, index):
        if 0 <= index < len(self.rows):
            row = self.rows[index]
            text = row["text"] + (f"\n\n{row['error']}" if row["error"] else "")
            if row.get("duration"):
                text += (
                    f"\n\n{row['duration']:.1f}s · {len(row['text'].split())} words · "
                    f"{len(row['text'])} characters"
                )
            if row.get("app") or row.get("window"):
                text += f"\n{row.get('app', '')} · {row.get('window', '')}"
            if row.get("raw_text") and row["raw_text"] != row["text"]:
                text += f"\n\nOriginal transcription:\n{row['raw_text']}"
            self.preview.setPlainText(text)
        else:
            self.preview.clear()

    def selected_action(self, action):
        index = self.list.currentRow()
        if 0 <= index < len(self.rows):
            self.send(action, {"id": self.rows[index]["id"]})

    def clear_history(self):
        if (
            QMessageBox.question(
                self, "Clear history", "Delete all transcripts and retained audio?"
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.send("clear")

    def save(self):
        try:
            previous = Config.load()
            values = {key: field_value(field) for key, field in self.fields.items()}
            updated = Config(**values)
            updated.save()
            if any(
                getattr(previous, key) != getattr(updated, key)
                for key in ("launch_at_startup", "show_window_at_login", "tray")
            ):
                from .cli import apply_startup

                apply_startup(updated)
            self.apply_theme()
            self.status.setText(
                "Saved; recording settings apply next time. Panel visibility applies now; "
                "audio budget is checked while idle."
            )
            return True
        except Exception as exc:
            self.status.setText(str(exc))
            return False

    def download_model(self):
        if not self.save() or self.download is not None:
            return
        self.download = QProcess(self)
        self.download.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.download.setProgram(sys.executable)
        self.download.setArguments(python_command("mouthkeys", "download")[1:])
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading…")

        def finished(code, _):
            output = bytes(self.download.readAllStandardOutput()).decode(errors="replace")
            self.download.deleteLater()
            self.download = None
            self.download_button.setEnabled(True)
            self.download_button.setText("Download selected model")
            if code == 0:
                self.send("warmup")
            QMessageBox.information(
                self, "Model download", "Model downloaded" if code == 0 else output
            )

        self.download.finished.connect(finished)
        self.download.errorOccurred.connect(self.download_error)
        self.download.start()

    def download_error(self, error):
        if error == QProcess.ProcessError.FailedToStart and self.download is not None:
            message = self.download.errorString()
            self.download.deleteLater()
            self.download = None
            self.download_button.setEnabled(True)
            self.download_button.setText("Download selected model")
            self.status.setText(message)

    def closeEvent(self, event):
        if self.jobs or self.download is not None or self.file_process is not None:
            self.status.setText("Wait for the current operation before closing")
            event.ignore()
            return
        event.accept()  # Closing the controller leaves the shortcut daemon running.


def launch():
    ensure_daemon()
    ensure_overlay()
    app = QApplication(sys.argv)
    app.setApplicationName("MouthKeys")
    window = Window()
    from .tray import create_tray

    tray = create_tray(window)
    app.setQuitOnLastWindowClosed(tray is None)
    if tray is None:
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.show()
    return app.exec()
