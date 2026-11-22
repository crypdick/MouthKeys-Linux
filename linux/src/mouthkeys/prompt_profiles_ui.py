"""Settings page for prompt profiles, draft tests, and application overrides."""

import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from typing import cast

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .ai_provider_settings import PROVIDER_ENDPOINTS
from .config import CHOICES, RANGES, Config
from .postprocess import process
from .profiles import PROFILE_FIELDS, PromptProfile, load_profiles, save_profiles
from .provider_settings_ui import KeyringAction
from .qt_control import Response, ResponseHandler, Send
from .routing import PromptRoute, load_routes, save_routes
from .settings_fields import SettingsField, field_value


def _set_field(field: SettingsField, value: str | int) -> None:
    if isinstance(field, QTextEdit):
        field.setPlainText(str(value))
    elif isinstance(field, QLineEdit):
        field.setText(str(value))
    elif isinstance(field, QSpinBox):
        field.setValue(int(value))
    else:
        field.setCurrentIndex(field.findData(value))


def _add_buttons(layout: QVBoxLayout, buttons: tuple[tuple[str, Callable[[], None]], ...]) -> None:
    row = QHBoxLayout()
    for label, handler in buttons:
        button = QPushButton(label)
        button.clicked.connect(handler)
        row.addWidget(button)
    layout.addLayout(row)


class PromptProfilesPage(QWidget):
    def __init__(
        self,
        send: Send,
        run_task: Callable[[Callable[[], object], ResponseHandler], None],
        show_status: Callable[[str], None],
        profiles_changed: Callable[[list[PromptProfile]], None],
        keyring_action: Callable[[KeyringAction], None],
    ) -> None:
        super().__init__()
        self.send = send
        self.run_task = run_task
        self.show_status = show_status
        self.profiles_changed = profiles_changed
        self.recording: tuple[str, Config] | None = None

        layout = QVBoxLayout(self)
        self._build_editor(layout, keyring_action)
        self._build_preview(layout)
        self._build_routes(layout)
        self.refresh_profiles()

    def _build_editor(
        self, layout: QVBoxLayout, keyring_action: Callable[[KeyringAction], None]
    ) -> None:
        note = QLabel(
            "Save named cleanup prompts and provider settings. Choose active profile in "
            "App Settings. Enable AI cleanup separately. Secrets stay in environment variables "
            "or your desktop keyring, never profile files."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.picker = QComboBox()
        layout.addWidget(self.picker)
        form = QFormLayout()
        self.profile_name = QLineEdit()
        form.addRow("Name", self.profile_name)
        self.shortcut = QLineEdit()
        self.shortcut.setPlaceholderText("Optional chord, e.g. super+alt+1")
        form.addRow("Shortcut", self.shortcut)
        self.fields: dict[str, SettingsField] = {}
        for name in PROFILE_FIELDS:
            field: SettingsField
            if name == "ai_prompt":
                field = QTextEdit()
                field.setMinimumHeight(110)
                field.setMaximumHeight(150)
            elif name in RANGES:
                field = QSpinBox()
                field.setRange(*RANGES[name])
            elif name in CHOICES:
                field = QComboBox()
                for value in CHOICES[name]:
                    field.addItem(value, value)
            else:
                field = QLineEdit()
            form.addRow(name.removeprefix("ai_").replace("_", " ").capitalize(), field)
            self.fields[name] = field
        layout.addLayout(form)
        _add_buttons(
            layout,
            (
                ("Use provider endpoint", self.use_provider_endpoint),
                ("Save profile", self.save_profile),
                ("Delete profile", self.delete_profile),
                ("Save keyring API key", partial(keyring_action, "store")),
                ("Delete keyring API key", partial(keyring_action, "delete")),
            ),
        )
        self.picker.currentIndexChanged.connect(self.select_profile)

    def _build_preview(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel("Prompt test — no desktop insertion"))
        self.preview_input = QTextEdit()
        self.preview_input.setPlaceholderText("Type test text, or record a prompt tryout")
        self.preview_output = QTextEdit()
        self.preview_output.setReadOnly(True)
        for field in (self.preview_input, self.preview_output):
            field.setMinimumHeight(80)
            field.setMaximumHeight(120)
            layout.addWidget(field)
        _add_buttons(
            layout,
            (
                ("Test draft prompt", self.test_prompt),
                ("Record test text", self.record_prompt),
                ("Stop test recording", lambda: self.send("stop", None, None)),
            ),
        )

    def _build_routes(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel("Application prompt overrides (exact application identity)"))
        self.routes = QTableWidget(0, 3)
        self.routes.setHorizontalHeaderLabels(["Application", "Prompt", "Mode"])
        self.routes.horizontalHeader().setStretchLastSection(True)
        self.routes.setMinimumHeight(160)
        layout.addWidget(self.routes)
        _add_buttons(
            layout,
            (
                ("Add application", self.add_route),
                ("Remove application", self.remove_route),
                ("Save app overrides", self.save_routes),
            ),
        )
        profiles = load_profiles()
        for route in load_routes(profiles):
            self._insert_route(route, profiles)

    @property
    def selection(self) -> str:
        return cast("str", self.picker.currentData())

    def values(self) -> dict[str, str | int | None]:
        return {name: field_value(field) for name, field in self.fields.items()}

    def refresh_profiles(self, selected: str = "") -> None:
        profiles = load_profiles()
        with QSignalBlocker(self.picker):
            self.picker.clear()
            self.picker.addItem("New profile", "")
            for profile in profiles:
                self.picker.addItem(profile["name"], profile["id"])
            self.picker.setCurrentIndex(max(0, self.picker.findData(selected)))
        for row in range(self.routes.rowCount()):
            picker = cast("QComboBox", self.routes.cellWidget(row, 1))
            self._fill_route_choices(picker, profiles, picker.currentData())
        self.profiles_changed(profiles)
        self._fill_profile_form(profiles)

    def select_profile(self, _index: int) -> None:
        self._fill_profile_form(load_profiles())

    def _fill_profile_form(self, profiles: list[PromptProfile]) -> None:
        selected = self.selection
        profile = next((p for p in profiles if p["id"] == selected), None)
        self.profile_name.setText(profile["name"] if profile else "")
        self.shortcut.setText(profile["shortcut"] if profile else "")
        values = cast("dict[str, str | int]", dict(profile)) if profile else asdict(Config.load())
        for name, field in self.fields.items():
            _set_field(field, values[name])

    def use_provider_endpoint(self) -> None:
        provider = cast("QComboBox", self.fields["ai_provider"]).currentText()
        _set_field(self.fields["ai_endpoint"], PROVIDER_ENDPOINTS[provider])

    def save_profile(self) -> None:
        try:
            selected = self.selection or str(uuid.uuid4())
            profile = {
                "id": selected,
                "name": self.profile_name.text(),
                "shortcut": self.shortcut.text(),
                **self.values(),
            }
            remaining = [p for p in load_profiles() if p["id"] != selected]
            save_profiles([*remaining, profile])
            self.refresh_profiles(selected)
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))
            return
        self.show_status("Profile saved; select it in App Settings to activate")

    def delete_profile(self) -> None:
        selected = self.selection
        if not selected:
            return
        try:
            if any(route["profile"] == selected for route in load_routes()):
                self.show_status(
                    "Remove application overrides using this profile before deleting it"
                )
                return
            config = Config.load()
            if selected in (config.ai_profile, config.edit_profile, config.command_profile):
                self.show_status("Choose and save another active profile before deleting this one")
                return
            if (
                QMessageBox.question(self, "Delete prompt", "Delete selected prompt profile?")
                == QMessageBox.StandardButton.Yes
            ):
                save_profiles([p for p in load_profiles() if p["id"] != selected])
                self.refresh_profiles()
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))

    def add_route(self) -> None:
        self._insert_route({"app": "", "profile": "default"}, load_profiles())

    def _insert_route(self, route: PromptRoute, profiles: list[PromptProfile]) -> None:
        row = self.routes.rowCount()
        self.routes.insertRow(row)
        self.routes.setItem(row, 0, QTableWidgetItem(route["app"]))
        picker = QComboBox()
        self._fill_route_choices(picker, profiles, route["profile"])
        self.routes.setCellWidget(row, 1, picker)
        mode = QComboBox()
        for name in ("dictation", "edit", "command"):
            mode.addItem(name, name)
        mode.setCurrentIndex(mode.findData(route.get("mode", "dictation")))
        self.routes.setCellWidget(row, 2, mode)

    @staticmethod
    def _fill_route_choices(
        picker: QComboBox, profiles: list[PromptProfile], selected: str | None
    ) -> None:
        picker.clear()
        picker.addItem("Default prompt", "default")
        picker.addItem("Off — no AI cleanup", "off")
        for profile in profiles:
            picker.addItem(profile["name"], profile["id"])
        picker.setCurrentIndex(picker.findData(selected))

    def remove_route(self) -> None:
        if (row := self.routes.currentRow()) >= 0:
            self.routes.removeRow(row)

    def save_routes(self) -> None:
        try:
            save_routes(
                [
                    {
                        "app": cast("QTableWidgetItem", self.routes.item(row, 0)).text(),
                        "profile": cast("QComboBox", self.routes.cellWidget(row, 1)).currentData(),
                        "mode": cast("QComboBox", self.routes.cellWidget(row, 2)).currentData(),
                    }
                    for row in range(self.routes.rowCount())
                ]
            )
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))
            return
        self.show_status("Application prompt overrides saved")

    def draft_config(self) -> Config:
        return Config.from_mapping(
            {**asdict(Config.load()), **self.values(), "ai_profile": "", "ai_enabled": "on"}
        )

    def record_prompt(self) -> None:
        try:
            config = self.draft_config()
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))
            return
        self.send("start", {"mode": "training"}, partial(self.recording_started, config))

    def recording_started(self, config: Config, response: Response) -> None:
        result = response.get("result")
        if isinstance(result, dict) and isinstance(identity := result.get("id"), str):
            self.recording = (identity, config)
            self.preview_output.setPlainText("Recording test; Stop when finished")

    def update_status(self, status: Mapping[str, object]) -> None:
        if self.recording is None or status.get("id") != self.recording[0]:
            return
        proposal = status.get("assistant_proposal")
        if (
            isinstance(proposal, dict)
            and proposal.get("mode") == "training"
            and isinstance(text := proposal.get("text"), str)
        ):
            _, config = self.recording
            self.recording = None
            self.preview_input.setPlainText(text)
            self.test_prompt(config=config)
        elif status.get("phase") == "idle":
            self.recording = None

    def test_prompt(self, *, config: Config | None = None) -> None:
        try:
            config = config or self.draft_config()
            text = self.preview_input.toPlainText()
            if not text.strip():
                raise ValueError("Provide test text first")
        except (OSError, TypeError, ValueError) as exc:
            self.preview_output.setPlainText(str(exc))
            return
        self.preview_output.setPlainText("Testing draft prompt…")
        self.run_task(partial(process, text, config), self.show_preview)

    def show_preview(self, response: Response) -> None:
        self.preview_output.setPlainText(str(response.get("error", response.get("result", ""))))
