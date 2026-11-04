"""Optional Qt desktop controller. The daemon owns audio and never opens windows."""

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QLocale, QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .config import Config
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
    return field.currentText() if field.isEditable() else field.currentData()


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
        self.nav.addItems(["History", "App Settings"])
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
        history_layout.addLayout(row)
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
        self.fields = {}
        config = asdict(Config.load())

        choices = {
            "model": model_choices(),
            "language": language_choices(),
            "compute_type": COMPUTE_TYPES,
            "audio_source": audio_sources(),
            "device": ["cpu", "cuda", "auto"],
            "recording_panel": [("Show panel", "on"), ("Hide panel", "off")],
            "delivery": ["clipboard", "x11"],
            "paste_backend": ["auto", "xdotool", "wtype", "ydotool"],
            "paste_shortcut": ["ctrl+shift+v", "ctrl+v"],
        }
        labels = {
            "max_seconds": "Maximum recording length",
            "audio_budget_mb": "Audio storage budget",
            "recording_panel": "Recording panel",
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
        }
        for key, value in config.items():
            # faster-whisper gives existing local directories precedence over aliases.
            if key == "model" and value in MODEL_ALIASES and not Path(value).is_dir():
                value = MODEL_ALIASES.get(value, value)
            if key in ("max_seconds", "audio_budget_mb"):
                field = QSpinBox()
                field.setRange(1, 3600 if key == "max_seconds" else 1000000)
                field.setSuffix(" seconds" if key == "max_seconds" else " MB")
                field.setValue(value)
            else:
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
            self.fields[key] = field
            form.addRow(labels.get(key, key.replace("_", " ").capitalize()), field)
            if key in help_text:
                field.setToolTip(help_text[key])
                hint = QLabel(help_text[key])
                hint.setWordWrap(True)
                form.addRow("", hint)
        self.fields["model"].setToolTip(
            "Approximate transcription accuracy, strongest first (English dictation).\n"
            "Accuracy varies by language and audio; .en and distilled models are English-only.\n"
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
        form.addRow(self.download_button)
        warm = QPushButton("Load model into memory")
        warm.clicked.connect(lambda: self.send("warmup"))
        form.addRow(warm)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)
        tabs.addWidget(settings_scroll)
        self.nav.setCurrentRow(0)
        self.apply_theme()
        QApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(700)
        self.poll()

    def apply_theme(self, *_):
        light = QApplication.styleHints().colorScheme() == Qt.ColorScheme.Light
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
                service_version = result.get("version")
                version_text = service_version or "Unknown"
                if service_version != __version__:
                    version_text += " — restart MouthKeys after updating"
                self.service_version.setText(version_text)
                seconds = f"  {result['seconds']:.1f}s" if result["phase"] == "recording" else ""
                self.status.setText(f"{result['phase'].upper()}{seconds} · {result['message']}")
                self.toggle.setEnabled(result["phase"] in ("idle", "recording"))
                self.send("history", callback=self.update_history)
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
            self.preview.setPlainText(row["text"] + (f"\n\n{row['error']}" if row["error"] else ""))
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
            values = {key: field_value(field) for key, field in self.fields.items()}
            Config(**values).save()
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
            QMessageBox.information(self, "Model download", "Model ready" if code == 0 else output)

        self.download.finished.connect(finished)
        self.download.start()

    def closeEvent(self, event):
        if self.jobs or self.download is not None:
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
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.show()
    return app.exec()
