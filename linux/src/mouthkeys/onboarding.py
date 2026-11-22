"""Guided local setup and an explicit, non-delivering dictation tryout."""

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import Config
from .desktop_input import doctor
from .qt_control import Response, Send
from .regional_ui import RegionalFillerBanner
from .settings_fields import SettingsField
from .setup_languages import LANGUAGE_ALIASES, ModelRoute, languages, routes


def _note(layout: QVBoxLayout, text: str) -> None:
    label = QLabel(text)
    label.setWordWrap(True)
    layout.addWidget(label)


def _message(value: object, fallback: str) -> str:
    return value if isinstance(value, str) else fallback


class SetupPage(QWidget):
    def __init__(
        self,
        fields: Mapping[str, SettingsField],
        send: Send,
        *,
        open_speech: Callable[[], None],
        open_microphone: Callable[[], None],
        refresh_other_banner: Callable[[], None],
    ) -> None:
        super().__init__()
        self.fields = fields
        self.send = send
        self.tryout_id: str | None = None
        self.tryout_generation = 0
        layout = QVBoxLayout(self)
        _note(
            layout,
            "1. Check desktop dependencies. 2. Select and explicitly download a speech model. "
            "3. Bind your desktop shortcut. 4. Try dictation here. "
            "Local dictation needs no account; cloud cleanup is optional.",
        )
        self._route_controls(layout, Config.load())
        self.regional_banner = RegionalFillerBanner(self, fields, on_change=refresh_other_banner)
        layout.addWidget(self.regional_banner)
        for label, action in (
            ("Check setup", self.check_setup),
            ("Choose speech model", open_speech),
            ("Choose microphone and priorities", open_microphone),
            ("Record tryout", self.record),
            ("Stop tryout", lambda: send("stop", None, None)),
            ("Cancel tryout", lambda: send("cancel", None, None)),
            ("Finish setup", self.finish),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            layout.addWidget(button)
        _note(
            layout,
            "Bind mouthkeys toggle in i3 or desktop keyboard settings. "
            "Separate press/release bindings can invoke mouthkeys start and mouthkeys stop. "
            "Cancel: mouthkeys cancel. Last result: mouthkeys paste.",
        )
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText(
            "Setup check and tryout text appear here. Tryout never pastes into another app. "
            "Finish setup may be used without a tryout; real editor/terminal paste "
            "still needs validation."
        )
        layout.addWidget(self.result)

    def _route_controls(self, layout: QVBoxLayout, config: Config) -> None:
        self.language_search = QLineEdit()
        self.language_search.setPlaceholderText("Find language by name, code, or alias")
        self.language_search.textChanged.connect(self.filter_languages)
        layout.addWidget(self.language_search)
        self.language = QComboBox()
        self.language.setAccessibleName("Setup dictation language")
        self.language_model = QStandardItemModel(self.language)
        self.language.setModel(self.language_model)
        for code, name in sorted(languages(), key=lambda item: item[1]):
            self.language.addItem(f"{name} ({code})", code)
        language = config.language.split("-", 1)[0]
        language = {"auto": "en", "nb": "no"}.get(language, language)
        self.language.setCurrentIndex(max(0, self.language.findData(language)))
        layout.addWidget(self.language)
        self.route = QComboBox()
        self.route.setAccessibleName("Compatible speech model route")
        layout.addWidget(self.route)
        self.refresh_routes()
        for index in range(self.route.count()):
            route = cast("ModelRoute", self.route.itemData(index))
            if route["engine"] == config.engine and route["model"] == config.model:
                self.route.setCurrentIndex(index)
                break
        self.language.currentIndexChanged.connect(lambda _index: self.refresh_routes())
        self.route.currentIndexChanged.connect(lambda _index: self.invalidate_tryout())

        apply_route = QPushButton("Apply language + model route")
        apply_route.clicked.connect(self.apply_route)
        layout.addWidget(apply_route)
        _note(
            layout,
            "Compatible choices, not quality rankings. Apply saves only this route; it does not "
            "download or load a model. Optional engines may need their Linux extras installed. "
            "Use Choose speech model for an explicit download.",
        )

    def check_setup(self) -> None:
        self.result.setPlainText(json.dumps(doctor(Config.load()), indent=2))

    def invalidate_tryout(self) -> None:
        self.tryout_generation += 1
        self.tryout_id = None

    def filter_languages(self, query: str) -> None:
        query = query.strip().casefold()
        matches = []
        for index in range(self.language.count()):
            code = self.language.itemData(index)
            searchable = " ".join([self.language.itemText(index), *LANGUAGE_ALIASES.get(code, [])])
            enabled = query in searchable.casefold()
            self.language_model.item(index).setEnabled(enabled)
            if enabled:
                matches.append(index)
        if matches and self.language.currentIndex() not in matches:
            self.language.setCurrentIndex(matches[0])

    def refresh_routes(self) -> None:
        self.invalidate_tryout()
        with QSignalBlocker(self.route):
            self.route.clear()
            language = self.language.currentData()
            if isinstance(language, str):
                for route in routes(language):
                    self.route.addItem(route["label"], route)

    def apply_route(self) -> None:
        route = cast("ModelRoute | None", self.route.currentData())
        if route is None:
            return
        try:
            updated = replace(
                Config.load(),
                engine=route["engine"],
                model=route["model"],
                language=route["language"],
                model_revision=route["model_revision"],
                compute_type="default",
            )
            updated.save()
        except (OSError, ValueError) as exc:
            self.result.setPlainText(str(exc))
            return
        self.invalidate_tryout()
        for key in ("engine", "model", "language", "model_revision", "compute_type"):
            field = self.fields[key]
            value = getattr(updated, key)
            if isinstance(field, QComboBox):
                if field.isEditable():
                    field.setEditText(value)
                else:
                    field.setCurrentIndex(field.findData(value))
            else:
                cast("QLineEdit", field).setText(value)
        self.result.setPlainText(
            "Language and model route saved. Choose speech model to download explicitly, "
            "then record a new tryout."
        )

    def record(self) -> None:
        generation = self.tryout_generation

        def started(response: Response) -> None:
            if generation != self.tryout_generation:
                return
            result = response.get("result")
            if not (
                isinstance(result, dict)
                and isinstance(recording_id := result.get("id"), str)
                and recording_id
            ):
                self.result.setPlainText(_message(response.get("error"), "Could not start tryout"))
                return
            if result.get("phase") == "recording":
                self.tryout_id = recording_id
                self.result.setPlainText("Listening. Stop when finished; no desktop delivery.")
            else:
                self.result.setPlainText(_message(result.get("message"), "Could not start tryout"))

        self.send("start", {"mode": "training"}, started)

    def update_status(self, status: Mapping[str, object]) -> None:
        if self.tryout_id is None or status.get("id") != self.tryout_id:
            return
        proposal = status.get("assistant_proposal")
        if isinstance(proposal, dict) and proposal.get("mode") == "training":
            text = proposal.get("text")
            if isinstance(text, str):
                self.result.setPlainText(text)
                self.regional_banner.refresh(dictation=text)
                self.tryout_id = None
                return
        if status.get("phase") == "idle":
            self.result.setPlainText(_message(status.get("message"), "Tryout finished"))
            self.tryout_id = None

    def finish(self) -> None:
        try:
            replace(Config.load(), onboarding_complete="on").save()
        except (OSError, ValueError) as exc:
            self.result.setPlainText(str(exc))
            return
        field = cast("QComboBox", self.fields["onboarding_complete"])
        field.setCurrentIndex(field.findData("on"))
        self.result.setPlainText(
            "Setup preferences saved. Start from your editor using the shortcut and validate "
            "paste, focus, cancellation, and microphone readiness on your desktop."
        )
