"Optional Qt desktop controller. The daemon owns audio and never opens windows."

from collections.abc import Callable, Mapping
from pathlib import Path
from sys import float_info
from typing import cast

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__, desktop_entry, models_ui, shortcut_capture
from .assistant_ui import AssistantPage
from .backup_ui import BackupPage
from .config import Config
from .desktop import play_sound
from .dictionary_ui import DictionaryPage
from .file_transcription_ui import FileTranscriptionPage
from .help_ui import HelpPage
from .history_ui import HistoryPage
from .hotkeys import spawn_hotkeys
from .learning_ui import LearningPage
from .microphone_ui import MicrophonePage
from .model_downloads_ui import ModelDownloads
from .onboarding import SetupPage
from .profiles import load_profiles
from .prompt_profiles_ui import PromptProfilesPage
from .provider_settings_ui import KeyringAction, ProviderSettings
from .qt_control import Request, Response, ResponseHandler, Task
from .regional_ui import RegionalFillerBanner
from .settings_fields import field_value
from .settings_form import SettingsForm
from .stats_view import StatsPage
from .window_theme import window_stylesheet


def _scroll_page(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    return scroll


class Window(QWidget):
    def __init__(self) -> None:
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
        self._add_recording_controls(layout)
        tabs = QStackedWidget()
        self.nav.currentRowChanged.connect(tabs.setCurrentIndex)
        self.nav.currentTextChanged.connect(self.page_title.setText)
        layout.addWidget(tabs)
        self.history_page = HistoryPage(self.send)
        tabs.addWidget(self.history_page)
        tabs.addWidget(self._build_settings())
        self._add_auxiliary_pages(tabs)
        self.nav.setCurrentRow(0)
        self.apply_theme()
        QApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(700)
        self.poll()

    def _add_recording_controls(self, layout: QVBoxLayout) -> None:
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

    def _build_settings(self) -> QScrollArea:
        settings = QWidget()
        form = QFormLayout(settings)
        self.app_version = QLabel(__version__)
        self.app_version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("App version", self.app_version)
        self.service_version = QLabel("Connecting…")
        self.service_version.setWordWrap(True)
        self.service_version.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Recording service version", self.service_version)
        options = self.settings_options = SettingsForm(form, Config.load(), load_profiles())
        self.fields = options.fields
        self.settings_category = options.category
        self.providers = ProviderSettings(self, self.fields, self.run_task, self.status.setText)
        self.regional_banner = RegionalFillerBanner(
            self, self.fields, on_change=self.refresh_regional_offers
        )
        options.add_grouped_row(self.regional_banner, "Text formatting")
        self.models_info = QLabel(
            "Download only on request. Install optional engine extras before switching engines."
        )
        self.models_info.setWordWrap(True)
        options.add_grouped_row(self.models_info, "Speech")
        catalog_button = QPushButton("Inspect model catalog / installed state")
        catalog_button.clicked.connect(self.inspect_models)
        options.add_grouped_row(catalog_button, "Speech")
        save = QPushButton("Save settings")
        save.clicked.connect(self.save)
        form.addRow(save)
        self.providers.add_buttons(options, self.save)
        for label, action, category in (
            (
                "Preview recording sound",
                lambda: self.preview_sound(stop=False),
                "Microphone & media",
            ),
            ("Preview stop sound", lambda: self.preview_sound(stop=True), "Microphone & media"),
            ("Start configured shortcut listener", self.start_hotkeys, "Desktop"),
            ("Capture keyboard shortcut", self.capture_shortcut, "Desktop"),
            (
                "Reset filler words to upstream defaults",
                lambda: self.regional_banner.reset_defaults(self.status.setText),
                "Text formatting",
            ),
            (
                "Regional filler suggestion",
                lambda: self.regional_banner.show_suggestion(self.status.setText),
                "Text formatting",
            ),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            options.add_grouped_row(button, category)
        self.model_downloads = ModelDownloads(
            self.save, self.selected_model, self.status.setText, lambda: self.send("warmup")
        )
        options.add_grouped_row(self.model_downloads, "Speech")
        self.settings_category.setCurrentText("Speech")
        return _scroll_page(settings)

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
            self.settings_options.refresh_profiles,
            self._profile_keyring,
        )
        tabs.addWidget(_scroll_page(self.prompt_profiles))
        self.assistant_page = AssistantPage(self, self.send, self.jobs)
        tabs.addWidget(self.assistant_page)
        self.setup_page = SetupPage(
            self.fields,
            self.send,
            open_speech=self._open_speech_settings,
            open_microphone=lambda: self.nav.setCurrentRow(11),
            refresh_other_banner=self.regional_banner.refresh,
        )
        tabs.addWidget(self.setup_page)
        self.learning_page = LearningPage(self.send)
        self.dictionary_page.vocabulary_changed.connect(self.learning_page.refresh_words)
        tabs.addWidget(_scroll_page(self.learning_page))
        self.microphone_page = MicrophonePage(self)
        tabs.addWidget(self.microphone_page)

    def _open_speech_settings(self) -> None:
        self.nav.setCurrentRow(1)
        self.settings_category.setCurrentText("Speech")

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

    def inspect_models(self) -> None:
        try:
            models_ui.ModelBrowser(self).exec()
        # Report to the Settings UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.status.setText(str(exc))

    def operation_result(self, response: Response) -> None:
        self.status.setText(str(response.get("error", response.get("result", "Done"))))

    def refresh_regional_offers(self) -> None:
        self.regional_banner.refresh()
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

    def run_task(self, function: Callable[[], object], callback: ResponseHandler) -> None:
        self._start_task(Task(function, self), callback)

    def _start_task(self, task: Task, callback: ResponseHandler) -> None:
        self.jobs.add(task)
        task.done.connect(callback)
        task.finished.connect(lambda: self.jobs.discard(task))
        task.finished.connect(task.deleteLater)
        task.start()

    def apply_theme(self, *_: object) -> None:
        self.setStyleSheet(
            window_stylesheet(
                Config.load().theme,
                system_light=QApplication.styleHints().colorScheme() == Qt.ColorScheme.Light,
            )
        )

    def send(
        self,
        action: str,
        data: Mapping[str, object] | None = None,
        callback: ResponseHandler | None = None,
    ) -> None:
        def done(response: Response) -> None:
            if "error" in response:
                self.status.setText(str(response["error"]))
            if callback:
                callback(response)

        self._start_task(Request(action, data, self), done)

    def poll(self) -> None:
        if self.polling:
            return
        self.polling = True
        self.send("status", callback=self._update_status)

    def _update_status(self, response: Response) -> None:
        self.polling = False
        result = response.get("result")
        if not isinstance(result, dict) or not all(
            isinstance(result.get(key), str) for key in ("phase", "message", "version")
        ):
            self.loaded_model = None
            self.service_version.setText("Unavailable")
            return

        model = result.get("loaded_model")
        self.loaded_model = model if isinstance(model, dict) else None
        phase, message, service_version = result["phase"], result["message"], result["version"]
        if service_version != __version__:
            service_version += " — restart MouthKeys after updating"
        self.service_version.setText(service_version)
        seconds = result.get("seconds")
        if not isinstance(seconds, int | float) or not 0 <= seconds <= float_info.max:
            seconds = 0
        elapsed = f"  {seconds:.1f}s" if phase == "recording" else ""
        self.status.setText(f"{phase.upper()}{elapsed} · {message}")
        self.toggle.setEnabled(
            phase in ("starting", "recording", "switching", "capture_failed")
            or result.get("can_record", phase == "idle") is True
        )
        self.assistant_page.update_status(result)
        self.setup_page.update_status(result)
        self.learning_page.update_status(result)
        self.microphone_page.update_status(result)
        self.prompt_profiles.update_status(result)
        self.dictionary_page.update_training(result)
        self.history_page.refresh_history()

    def save(self) -> bool:
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
        # Report to the Settings UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.status.setText(str(exc))
            return False
        return True

    def _profile_keyring(self, action: KeyringAction) -> None:
        page = self.prompt_profiles
        self.providers.keyring(action, page.fields, profile_selection=lambda: page.selection)

    def preview_sound(self, *, stop: bool) -> None:
        if self.save():
            play_sound(Config.load(), stop=stop, preview=True)

    def start_hotkeys(self) -> None:
        if not self.save():
            return
        try:
            spawn_hotkeys()
        except OSError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Shortcut listener requested; portal setup may ask for desktop consent")

    def capture_shortcut(self) -> None:
        shortcut_capture.ShortcutCapture(
            self, lambda key, chord: cast("QLineEdit", self.fields[key]).setText(chord)
        ).exec()

    def download_model(self) -> None:
        self.model_downloads.download_model()

    @property
    def busy(self) -> bool:
        return bool(self.jobs) or any(
            page.running for page in (self.model_downloads, self.file_page, self.backup_page)
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        if self.busy:
            self.status.setText("Wait for the current operation before closing")
            event.ignore()
            return
        event.accept()  # Closing the controller leaves the shortcut daemon running.
