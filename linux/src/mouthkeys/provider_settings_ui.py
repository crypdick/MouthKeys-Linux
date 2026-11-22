"""Run explicit provider and keyring actions from the Settings editors."""

from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import partial
from typing import Literal, cast

from PySide6.QtWidgets import QComboBox, QInputDialog, QLineEdit, QMessageBox, QPushButton, QWidget

from . import credentials, provider_tools
from .config import Config
from .qt_control import Response, ResponseHandler
from .settings_fields import SettingsField, field_value
from .settings_form import SettingsForm

KeyringAction = Literal["store", "delete"]
ProviderAction = Literal["verify", "models"]


class ProviderSettings:
    def __init__(
        self,
        parent: QWidget,
        fields: Mapping[str, SettingsField],
        run_task: Callable[[Callable[[], object], ResponseHandler], None],
        show_status: Callable[[str], None],
    ) -> None:
        self.parent = parent
        self.fields = fields
        self.run_task = run_task
        self.show_status = show_status

    def add_buttons(self, form: SettingsForm, save: Callable[[], bool]) -> None:
        for label, action in (
            ("Verify AI connection", lambda: self.request("verify", save)),
            ("Discover AI models", lambda: self.request("models", save)),
            ("Save API key in desktop keyring", lambda: self.keyring("store", self.fields)),
            ("Delete API key from desktop keyring", lambda: self.keyring("delete", self.fields)),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            form.add_grouped_row(button, "AI cleanup")

    def request(self, action: ProviderAction, save: Callable[[], bool]) -> None:
        if not save():
            return
        config = Config.load()
        operation = provider_tools.verify if action == "verify" else provider_tools.list_models

        def received(response: Response) -> None:
            if "error" in response:
                self.show_status(str(response["error"]))
            elif action == "verify":
                self.show_status("Provider connection verified")
            else:
                self._pick_model(cast("list[str]", response["result"]))

        self.run_task(partial(operation, config), received)
        self.show_status("Checking provider…")

    def _pick_model(self, models: list[str]) -> None:
        if not models:
            self.show_status("Provider returned no models")
            return
        model, accepted = QInputDialog.getItem(
            self.parent, "AI models", "Choose a model", models, editable=False
        )
        if accepted:
            cast("QLineEdit", self.fields["ai_model"]).setText(model)
            self.show_status("Model selected; Save settings to apply")

    def keyring(
        self,
        action: KeyringAction,
        fields: Mapping[str, SettingsField],
        *,
        profile_selection: Callable[[], str] | None = None,
    ) -> None:
        try:
            values = {key: field_value(field) for key, field in fields.items()}
            config = Config.from_mapping({**asdict(Config.load()), **values})
        except (OSError, TypeError, ValueError) as exc:
            self.show_status(str(exc))
            return
        if action == "delete":
            self._delete_key(config)
            return
        selected_profile = profile_selection() if profile_selection is not None else None
        key, accepted = QInputDialog.getText(
            self.parent,
            "Save API key",
            "API key for the configured provider endpoint:",
            QLineEdit.EchoMode.Password,
        )
        if not accepted:
            return

        def received(response: Response) -> None:
            if "error" in response:
                self.show_status(str(response["error"]))
                return
            if (
                field_value(fields["ai_endpoint"]) != config.ai_endpoint
                or field_value(fields["ai_provider"]) != config.ai_provider
                or (profile_selection is not None and profile_selection() != selected_profile)
            ):
                self.show_status("API key saved for earlier endpoint; current editor unchanged")
                return
            source = cast("QComboBox", fields["ai_credential_source"])
            source.setCurrentIndex(source.findData("secret-service"))
            editor = "profile" if profile_selection is not None else "settings"
            self.show_status(f"API key saved in desktop keyring. Save {editor} to use it.")

        self.run_task(partial(credentials.store_key, config, key), received)

    def _delete_key(self, config: Config) -> None:
        if (
            QMessageBox.question(
                self.parent, "Delete API key", "Delete the key saved for this provider endpoint?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        def received(response: Response) -> None:
            self.show_status(str(response.get("error", "API key deleted from desktop keyring")))

        self.run_task(partial(credentials.delete_key, config), received)
