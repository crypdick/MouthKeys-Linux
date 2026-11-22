"Optional Qt desktop controller. The daemon owns audio and never opens windows."

from collections.abc import Callable, Mapping
from pathlib import Path
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

from . import __version__, desktop_entry
from .backup_ui import BackupPage
from .config import Config, state_dir
from .dictionary_ui import DictionaryPage
from .file_transcription_ui import FileTranscriptionPage
from .help_ui import HelpPage
from .history_ui import HistoryPage
from .model_downloads_ui import ModelDownloads
from .processes import python_command
from .profiles import load_profiles
from .prompt_profiles_ui import PromptProfilesPage
from .provider_settings_ui import KeyringAction, ProviderSettings
from .qt_control import Request, Response, ResponseHandler, Task
from .regional_ui import RegionalFillerBanner
from .settings_fields import field_value
from .settings_form import SettingsForm
from .stats_view import StatsPage
from .window_theme import window_stylesheet


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
            ("Preview recording sound", lambda: self.preview_sound(False), "Microphone & media"),
            ("Preview stop sound", lambda: self.preview_sound(True), "Microphone & media"),
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
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)
        return settings_scroll

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

    def inspect_models(self):
        from .models_ui import ModelBrowser

        try:
            ModelBrowser(self).exec()
        # Report to the Settings UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.status.setText(str(exc))

    def operation_result(self, response: Response) -> None:
        self.status.setText(str(response.get("error", response.get("result", "Done"))))

    def refresh_regional_offers(self) -> None:
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

    def run_task(self, function: Callable[[], object], callback: ResponseHandler) -> None:
        task = Task(function, self)
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

    def poll(self) -> None:
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

    def _profile_keyring(self, action: KeyringAction) -> None:
        page = self.prompt_profiles
        self.providers.keyring(action, page.fields, profile_selection=lambda: page.selection)

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
