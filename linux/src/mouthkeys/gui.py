"Optional Qt desktop controller. The daemon owns audio and never opens windows."

import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from PySide6.QtCore import QLocale, Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__, desktop_entry, desktop_input
from .backup_ui import BackupPage
from .config import CHOICES, RANGES, Config, config_path, state_dir
from .control import ensure_daemon, ensure_overlay
from .desktop_input import automatic_delivery_warning
from .dictionary_ui import DictionaryPage
from .file_transcription_ui import FileTranscriptionPage
from .help_ui import HelpPage
from .history_ui import HistoryPage
from .model_downloads_ui import ModelDownloads
from .processes import python_command
from .profiles import PromptProfile, load_profiles
from .prompt_profiles_ui import KeyringAction, PromptProfilesPage
from .qt_control import Request, Response, ResponseHandler, Task
from .settings_fields import field_value
from .stats_view import StatsPage

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

    if desktop_input.session() == "headless":
        return ["default"]
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],  # noqa: S607
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


def setting_group(key):
    if "shortcut" in key or key in ("hotkey_mode", "shortcut_backend"):
        return "Desktop"
    if key in (
        "edit_profile",
        "command_profile",
        "edit_prompt",
        "command_prompt",
        "command_confirm",
        "command_continue",
    ):
        return "AI cleanup"
    if key in ("native_timeout_seconds", "file_diarization_mode"):
        return "Speech"
    if key == "mic_battery":
        return "Microphone & media"
    if key == "file_history_limit":
        return "History & stats"
    if key.startswith(("learning_", "pronunciation_")) or key in (
        "automatic_learning",
        "pronunciation_matching",
        "vocabulary_boosting",
    ):
        return "Dictionary & learning"
    if key.startswith("api_"):
        return "Local API"
    if key.startswith("ai_"):
        return "AI cleanup"
    if key.startswith("spoken_send") or key == "send_after_pause":
        return "Spoken Send"
    if key.startswith(("microphone_", "sound_")) or key in (
        "audio_source",
        "pause_media",
        "sounds_enabled",
    ):
        return "Microphone & media"
    if key.startswith(("paste_", "clipboard_")) or key in (
        "delivery",
        "destination_guard",
        "insertion_mode",
        "restore_clipboard",
        "copy_to_clipboard",
        "target_policy",
        "return_to_start",
        "verify_paste",
        "edit_delivery",
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
    if key.startswith("save_") or key in ("typing_wpm", "weekends_break_streak"):
        return "History & stats"
    if key in (
        "tray",
        "launch_at_startup",
        "show_window_at_login",
        "show_microphone_alerts",
        "show_paste_alerts",
        "show_ai_alerts",
        "hide_from_taskbar",
        "onboarding_complete",
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
        "model_revision",
        "native_streaming",
        "streaming_latency",
        "preview_interval_ms",
        "asr_max_tokens",
    ):
        return "Speech"
    return "Text formatting"


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.Tool, Config.load().hide_from_taskbar == "on")
        self.setWindowTitle("MouthKeys")
        icon_path = str(Path(__file__).parent / "assets" / "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1000, 680)
        self.setMinimumSize(820, 560)
        self.jobs: set[QThread] = set()
        self.polling = False
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._make_navigation(icon_path))
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
        self.history_page = HistoryPage(self.send)
        tabs.addWidget(self.history_page)
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
                "Dictionary & learning",
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
            "delivery": [
                ("Clipboard only", "clipboard"),
                ("X11 auto-paste", "x11"),
                ("Wayland auto-paste", "wayland"),
            ],
            "destination_guard": [
                ("Field first, window fallback", "field_then_window"),
                ("Require field or pane", "field"),
            ],
            "paste_backend": ["auto", "xdotool", "wtype", "ydotool"],
            "paste_shortcut": [
                ("Ctrl+Shift+V (Ghostty / most terminals)", "ctrl+shift+v"),
                ("Ctrl+V (most editors)", "ctrl+v"),
            ],
            "sound_preset": [
                ("Custom files", "custom"),
                ("None", "none"),
                ("Legacy MouthKeys pair", "mouthkeys"),
                *((f"Sound {index + 1}", f"fluid_sfx_{index}") for index in range(5)),
            ],
            "command_confirm": [
                ("Every command", "on"),
                ("Destructive commands (upstream heuristic)", "destructive"),
                ("No individual confirmation", "off"),
            ],
        }
        profiles = load_profiles()
        choices["ai_profile"] = [("Default settings", "")] + [
            (profile["name"], profile["id"]) for profile in profiles
        ]
        for key in ("edit_profile", "command_profile"):
            choices[key] = [("Sync to global provider", "default"), ("Off", "off")] + [
                (profile["name"], profile["id"]) for profile in profiles
            ]
        labels = {
            "max_seconds": "Maximum recording length",
            "recording_panel": "Recording panel",
            "ai_key_env": "AI API key environment variable",
            "ai_credential_source": "AI credential source",
            "destination_guard": "Automatic delivery destination checks",
            "pronunciation_aligner_model": "Optional local Whisper pronunciation aligner",
            "ai_endpoint": "AI cleanup endpoint (full URL)",
            "spoken_send": "Spoken Send",
            "send_after_pause": "Send after a pause",
            "spoken_send_key": "Send key",
            "save_history": "Save transcription history",
            "microphone_preferred": "Preferred microphones (ordered, comma-separated)",
            "microphone_fallback": "Last-resort microphones (ordered, comma-separated)",
            "microphone_never": "Never use microphones (comma-separated)",
            "typing_wpm": "Typing speed for time-saved estimate (WPM)",
            "file_speaker_count": "File speaker count (0 = automatic)",
            "command_confirm": "Confirm before running commands",
            "ai_custom_prompt_only": "Send custom dictation prompt only",
            "hide_from_taskbar": "Hide main window from taskbar / app switcher",
            "show_ai_alerts": "AI processing failure notifications",
            "sounds_enabled": "Recording sounds",
        }
        help_text = {
            "ai_custom_prompt_only": (
                "Off prepends the built-in dictation-cleaning instructions to your custom prompt. "
                "On sends only your prompt and transcript. Command/Edit prompts are separate."
            ),
            "hide_from_taskbar": (
                "Use a Linux tool-window hint for the main window. Taskbar and app-switcher "
                "behavior depends on your compositor. Reopen through the tray or mouthkeys gui."
            ),
            "command_confirm": (
                "Every command shows the exact shell command and directory for review. "
                "Destructive uses upstream's command-text heuristic, not a sandbox. "
                "No individual confirmation runs model-generated commands as your user; "
                "continuation remains bounded to 20 steps."
            ),
            "max_seconds": (
                "At the limit, recording stops and text is copied without automatic paste."
            ),
            "recording_panel": (
                "Hide the panel if your compositor tiles it or takes keyboard focus. "
                "Dictation still works; the panel also shows loading and shortcut errors."
            ),
            "spoken_send": (
                "Opt-in: send configured key only after automatic insertion into "
                "an unchanged X11 destination."
            ),
            "ai_enabled": (
                "When on, recorded transcript is sent to the configured local or "
                "remote AI endpoint."
            ),
            "ai_key_env": (
                "Set this variable in the recording service environment; API keys "
                "are never stored in backups."
            ),
            "ai_credential_source": (
                "Environment uses the named variable. Secret Service uses your desktop keyring "
                "for this exact provider endpoint; Save API key below. Settings never reads keys."
            ),
            "destination_guard": (
                "Both options check the field or pane when it can be identified. If no field was "
                "identified, window fallback can still paste or send in the same window, but "
                "cannot detect tab or pane switches. Require field or pane copies instead. "
                "Paste latest still pastes at current focus."
            ),
            "pronunciation_aligner_model": (
                "Blank disables extra alignment. A complete already-cached Whisper model/path "
                "can align engines without native word intervals for pronunciation matching. "
                "No download; separate CPU/int8 model adds RAM and processing time. "
                "Native word intervals remain preferred."
            ),
            "save_history": (
                "Off removes completed attempts after delivery; failed raw text is"
                " not retained either."
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
                if key == "max_seconds":
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
            if key == "delivery":
                self.delivery_hint = QLabel()
                self.delivery_hint.setWordWrap(True)
                self.settings_rows.append((form.rowCount(), setting_group(key)))
                form.addRow("", self.delivery_hint)
            elif key in help_text:
                field.setToolTip(help_text[key])
                hint = QLabel(help_text[key])
                hint.setWordWrap(True)
                self.settings_rows.append((form.rowCount(), setting_group(key)))
                form.addRow("", hint)
        for key in ("delivery", "paste_backend", "insertion_mode"):
            self.fields[key].currentIndexChanged.connect(self.refresh_delivery_hint)
        self.refresh_delivery_hint()
        from .regional_ui import RegionalFillerBanner

        self.regional_banner = RegionalFillerBanner(self, on_change=self.refresh_regional_offers)
        self.settings_rows.append((form.rowCount(), "Text formatting"))
        form.addRow(self.regional_banner)
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
        for label, action, category in (
            ("Verify AI connection", lambda: self.provider_action("verify"), "AI cleanup"),
            ("Discover AI models", lambda: self.provider_action("models"), "AI cleanup"),
            ("Save API key in desktop keyring", lambda: self.keyring_action("store"), "AI cleanup"),
            (
                "Delete API key from desktop keyring",
                lambda: self.keyring_action("delete"),
                "AI cleanup",
            ),
            ("Preview recording sound", lambda: self.preview_sound(False), "Microphone & media"),
            ("Preview stop sound", lambda: self.preview_sound(True), "Microphone & media"),
            ("Start configured shortcut listener", self.start_hotkeys, "Desktop"),
            ("Capture keyboard shortcut", self.capture_shortcut, "Desktop"),
            ("Reset filler words to upstream defaults", self.reset_fillers, "Text formatting"),
            ("Regional filler suggestion", self.offer_regional_filler, "Text formatting"),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            self.settings_rows.append((form.rowCount(), category))
            form.addRow(button)
        self.model_downloads = ModelDownloads(
            self.save, self.selected_model, self.status.setText, lambda: self.send("warmup")
        )
        self.settings_rows.append((form.rowCount(), "Speech"))
        form.addRow(self.model_downloads)
        self.settings_category.currentTextChanged.connect(self.filter_settings)
        self.settings_category.setCurrentText("Speech")
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)
        tabs.addWidget(settings_scroll)
        self._add_auxiliary_pages(tabs)
        self.nav.setCurrentRow(0)
        self.apply_theme()
        QApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(700)
        self.poll()

    def _add_auxiliary_pages(self, tabs: QStackedWidget) -> None:
        self.dictionary_page = DictionaryPage(self.send, self.status.setText)
        tabs.addWidget(self.dictionary_page)
        self.file_page = FileTranscriptionPage(tabs, self.save, self.status.setText)
        tabs.addWidget(self.file_page)
        self.setAcceptDrops(True)
        tabs.addWidget(StatsPage(self.send))
        self.backup_page = BackupPage(self.send, self.status.setText, self.operation_result)
        tabs.addWidget(self.backup_page)
        tabs.addWidget(HelpPage())
        self.prompt_profiles = PromptProfilesPage(
            self.send,
            self.run_task,
            self.status.setText,
            self.refresh_profile_choices,
            lambda action: self.keyring_action(action, profile=True),
        )
        profiles_scroll = QScrollArea()
        profiles_scroll.setWidgetResizable(True)
        profiles_scroll.setWidget(self.prompt_profiles)
        tabs.addWidget(profiles_scroll)
        from .assistant_ui import AssistantPage

        self.assistant_page = AssistantPage(self)
        tabs.addWidget(self.assistant_page)
        from .onboarding import SetupPage

        self.setup_page = SetupPage(self)
        tabs.addWidget(self.setup_page)
        from .learning_ui import LearningPage

        self.learning_page = LearningPage(self)
        self.dictionary_page.vocabulary_changed.connect(self.learning_page.refresh_words)
        learning_scroll = QScrollArea()
        learning_scroll.setWidgetResizable(True)
        learning_scroll.setWidget(self.learning_page)
        tabs.addWidget(learning_scroll)
        from .microphone_ui import MicrophonePage

        self.microphone_page = MicrophonePage(self)
        tabs.addWidget(self.microphone_page)

    def _make_navigation(self, icon_path: str) -> QWidget:
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
                "Backup",
                "Help & Diagnostics",
                "AI Prompts",
                "Command / Edit",
                "Getting Started",
                "Learning & Vocabulary",
                "Microphone",
            ]
        )
        navigation.addWidget(self.nav)
        footer = QLabel("LOCAL DICTATION  /  LINUX")
        footer.setObjectName("placard")
        navigation.addWidget(footer)
        return sidebar

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
        from .models_ui import ModelBrowser

        try:
            ModelBrowser(self).exec()
        # Report to the Settings UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.status.setText(str(exc))

    def operation_result(self, response: Response) -> None:
        self.status.setText(str(response.get("error", response.get("result", "Done"))))

    def reset_fillers(self):
        from .regional import UPSTREAM_FILLERS

        self.fields["filler_words"].setText(UPSTREAM_FILLERS)
        self.status.setText("Upstream filler list restored in editor; Save settings to apply")

    def offer_regional_filler(self):
        from .regional import region_offer

        config = Config.load()
        offer = region_offer(config)
        if not offer:
            self.status.setText(
                "No unanswered regional filler suggestion applies to current settings"
            )
            return
        keep = (
            QMessageBox.question(self, "Regional filler", offer["message"])
            == QMessageBox.StandardButton.Yes
        )
        words = config.filler_words
        if keep:
            words = ",".join(
                word for word in words.split(",") if word.strip().casefold() != offer["word"]
            )
        replace(config, filler_words=words, regional_filler_answered="on").save()
        self.fields["filler_words"].setText(words)
        field = self.fields["regional_filler_answered"]
        field.setCurrentIndex(field.findData("on"))
        self.refresh_regional_offers()

    def refresh_regional_offers(self):
        self.regional_banner.refresh()
        if hasattr(self, "setup_page"):
            self.setup_page.regional_banner.refresh()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.file_page.file_path.setText(urls[0].toLocalFile())
            self.nav.setCurrentRow(3)
            event.acceptProposedAction()

    def selected_model(self) -> tuple[str, str, str]:
        return (
            field_value(self.fields["model"]),
            field_value(self.fields["engine"]),
            field_value(self.fields["model_revision"]),
        )

    def delete_model(self) -> None:
        self.model_downloads.delete_model()

    def refresh_profile_choices(self, profiles: list[PromptProfile]) -> None:
        for key in ("ai_profile", "edit_profile", "command_profile"):
            picker = cast("QComboBox", self.fields[key])
            selected = picker.currentData()
            picker.clear()
            if key == "ai_profile":
                picker.addItem("Default settings", "")
            else:
                picker.addItem("Sync to global provider", "default")
                picker.addItem("Off", "off")
            for profile in profiles:
                picker.addItem(profile["name"], profile["id"])
            picker.setCurrentIndex(max(0, picker.findData(selected)))

    def run_task(self, function: Callable[[], object], callback: ResponseHandler) -> None:
        task = Task(function, self)
        self.jobs.add(task)
        task.done.connect(callback)
        task.finished.connect(lambda: self.jobs.discard(task))
        task.finished.connect(task.deleteLater)
        task.start()

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

    def send(
        self,
        action: str,
        data: Mapping[str, object] | None = None,
        callback: ResponseHandler | None = None,
    ) -> None:
        job = Request(action, data, self)
        self.jobs.add(job)

        def done(response: Response) -> None:
            if "error" in response:
                self.status.setText(str(response["error"]))
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
                self.loaded_model = result.get("loaded_model")
                service_version = result["version"]
                version_text = service_version
                if service_version != __version__:
                    version_text += " — restart MouthKeys after updating"
                self.service_version.setText(version_text)
                seconds = f"  {result['seconds']:.1f}s" if result["phase"] == "recording" else ""
                self.status.setText(f"{result['phase'].upper()}{seconds} · {result['message']}")
                self.toggle.setEnabled(
                    result["phase"] in ("starting", "recording", "switching", "capture_failed")
                    or result.get("can_record", result["phase"] == "idle")
                )
                self.assistant_page.update_status(result)
                self.setup_page.update_status(result)
                self.learning_page.update_status(result)
                self.microphone_page.update_status(result)
                self.prompt_profiles.update_status(result)
                self.dictionary_page.update_training(result)
                self.history_page.refresh_history()
            else:
                self.loaded_model = None
                self.service_version.setText("Unavailable")

        self.send("status", callback=update)

    def refresh_delivery_hint(self):
        config = Config(
            **{
                key: field_value(self.fields[key])
                for key in ("delivery", "paste_backend", "insertion_mode")
            },
        )
        warning = automatic_delivery_warning(config)
        message = (
            "Dictation copies text. Paste manually in the destination."
            if config.delivery == "clipboard"
            else "Auto-paste requires an unchanged destination. Otherwise, text is copied."
        )
        if warning:
            message = f"{warning} Clipboard fallback stays available."
            if config.insertion_mode == "direct" and config.delivery == desktop_input.session():
                message += " Direct insertion may still work in supported fields."
        self.delivery_hint.setText(f"Detected session: {desktop_input.session()}. {message}")

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
                desktop_entry.apply_startup(updated)
            self.apply_theme()
            self.refresh_regional_offers()
            if previous.hide_from_taskbar != updated.hide_from_taskbar:
                visible = self.isVisible()
                self.setWindowFlag(Qt.WindowType.Tool, updated.hide_from_taskbar == "on")
                if visible:
                    self.show()
            self.status.setText(
                "Saved; recording settings apply next time. Panel visibility applies now."
            )
            if previous.audio_source != updated.audio_source:
                self.send("select_microphone", {"source": updated.audio_source, "persist": True})
            return True
        # Report to the Settings UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.status.setText(str(exc))
            return False

    def keyring_action(self, action: KeyringAction, *, profile: bool = False) -> None:
        from PySide6.QtWidgets import QInputDialog

        from .credentials import delete_key, store_key

        fields = self.prompt_profiles.fields if profile else self.fields
        selection = self.prompt_profiles.selection if profile else None
        try:
            values = {key: field_value(field) for key, field in fields.items()}
            config = Config.from_mapping({**asdict(Config.load()), **values})
        except (OSError, TypeError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        if action == "store":
            key, accepted = QInputDialog.getText(
                self,
                "Save API key",
                "API key for the configured provider endpoint:",
                QLineEdit.EchoMode.Password,
            )
            if not accepted:
                return

            def function() -> None:
                store_key(config, key)
        else:
            if (
                QMessageBox.question(
                    self, "Delete API key", "Delete the key saved for this provider endpoint?"
                )
                != QMessageBox.StandardButton.Yes
            ):
                return

            def function() -> None:
                delete_key(config)

        def received(response: Response) -> None:
            if "error" in response:
                self.status.setText(str(response["error"]))
                return
            if action == "store":
                if (
                    field_value(fields["ai_endpoint"]) != config.ai_endpoint
                    or field_value(fields["ai_provider"]) != config.ai_provider
                    or (profile and self.prompt_profiles.selection != selection)
                ):
                    self.status.setText(
                        "API key saved for earlier endpoint; current editor unchanged"
                    )
                    return
                source = cast("QComboBox", fields["ai_credential_source"])
                source.setCurrentIndex(source.findData("secret-service"))
                self.status.setText(
                    "API key saved in desktop keyring. Save profile to use it."
                    if profile
                    else "API key saved in desktop keyring. Save settings to use it."
                )
            else:
                self.status.setText("API key deleted from desktop keyring")

        self.run_task(function, received)

    def provider_action(self, action):
        if not self.save():
            return
        from .provider_tools import list_models, verify

        config = Config.load()
        function = (lambda: verify(config)) if action == "verify" else (lambda: list_models(config))

        def received(response):
            if "error" in response:
                self.status.setText(response["error"])
            elif action == "verify":
                self.status.setText("Provider connection verified")
            else:
                from PySide6.QtWidgets import QInputDialog

                models = response["result"]
                if not models:
                    self.status.setText("Provider returned no models")
                    return
                model, accepted = QInputDialog.getItem(
                    self, "AI models", "Choose a model", models, editable=False
                )
                if accepted:
                    self.fields["ai_model"].setText(model)
                    self.status.setText("Model selected; Save settings to apply")

        self.run_task(function, received)
        self.status.setText("Checking provider…")

    def preview_sound(self, stop):
        if self.save():
            from .desktop import play_sound

            play_sound(Config.load(), stop=stop, preview=True)

    def start_hotkeys(self):
        if not self.save():
            return
        import subprocess

        with (state_dir() / "hotkeys.log").open("a") as output:
            Path(output.name).chmod(0o600)
            subprocess.Popen(  # noqa: S603
                python_command("mouthkeys", "hotkeys"),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
        self.status.setText("Shortcut listener requested; portal setup may ask for desktop consent")

    def capture_shortcut(self):
        from .shortcut_capture import ShortcutCapture

        ShortcutCapture(
            self, lambda key, chord: cast("QLineEdit", self.fields[key]).setText(chord)
        ).exec()

    def download_model(self) -> None:
        self.model_downloads.download_model()

    @property
    def busy(self) -> bool:
        return (
            bool(self.jobs)
            or self.model_downloads.running
            or self.file_page.running
            or self.backup_page.running
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self.busy:
            self.status.setText("Wait for the current operation before closing")
            event.ignore()
            return
        event.accept()  # Closing the controller leaves the shortcut daemon running.


def launch():
    first_launch = not config_path().exists()
    ensure_daemon()
    ensure_overlay()
    from .hotkeys import ensure_hotkeys

    ensure_hotkeys()
    app = QApplication(sys.argv)
    app.setApplicationName("MouthKeys")
    window = Window()
    if first_launch:
        window.nav.setCurrentRow(9)
    from .tray import create_tray

    tray = create_tray(window)
    app.setQuitOnLastWindowClosed(tray is None)
    if tray is None:
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.show()
    return app.exec()
