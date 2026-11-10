"""Guided local setup and an explicit, non-delivering dictation tryout."""

from dataclasses import replace

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
from .regional_ui import RegionalFillerBanner
from .setup_languages import LANGUAGE_ALIASES, languages, routes


class SetupPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.tryout_id = None
        self.tryout_generation = 0
        layout = QVBoxLayout(self)
        note = QLabel(
            "1. Check desktop dependencies. 2. Select and explicitly download a speech model. "
            "3. Bind your desktop shortcut. 4. Try dictation here. "
            "Local dictation needs no account; cloud cleanup is optional."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.language_search = QLineEdit()
        self.language_search.setPlaceholderText("Find language by name, code, or alias")
        self.language_search.textChanged.connect(self.filter_languages)
        layout.addWidget(self.language_search)
        self.language = QComboBox()
        self.language.setAccessibleName("Setup dictation language")
        for code, name in sorted(languages(), key=lambda item: item[1]):
            self.language.addItem(f"{name} ({code})", code)
        config = Config.load()
        language = config.language.split("-", 1)[0]
        if language == "nb":
            language = "no"
        self.language.setCurrentIndex(
            max(0, self.language.findData(language if language != "auto" else "en"))
        )
        self.language.currentIndexChanged.connect(self.refresh_routes)
        layout.addWidget(self.language)
        self.route = QComboBox()
        self.route.setAccessibleName("Compatible speech model route")
        layout.addWidget(self.route)
        self.refresh_routes()
        for index in range(self.route.count()):
            route = self.route.itemData(index)
            if route["engine"] == config.engine and route["model"] == config.model:
                self.route.setCurrentIndex(index)
                break
        self.route.currentIndexChanged.connect(self.invalidate_tryout)
        apply_route = QPushButton("Apply language + model route")
        apply_route.clicked.connect(self.apply_route)
        layout.addWidget(apply_route)
        route_note = QLabel(
            "Compatible choices, not quality rankings. Apply saves only this route; it does not "
            "download or load a model. Optional engines may need their Linux extras installed. "
            "Use Choose speech model for an explicit download."
        )
        route_note.setWordWrap(True)
        layout.addWidget(route_note)
        self.regional_banner = RegionalFillerBanner(window, on_change=self.refresh_other_banner)
        layout.addWidget(self.regional_banner)
        for label, action in (
            ("Check setup", self.check_setup),
            ("Choose speech model", self.open_settings),
            ("Choose microphone and priorities", lambda: window.nav.setCurrentRow(11)),
            ("Record tryout", self.record),
            ("Stop tryout", lambda: window.send("stop")),
            ("Cancel tryout", lambda: window.send("cancel")),
            ("Finish setup", self.finish),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            layout.addWidget(button)
        shortcut = QLabel(
            "Bind mouthkeys toggle in i3 or desktop keyboard settings. "
            "Separate press/release bindings can invoke mouthkeys start and mouthkeys stop. "
            "Cancel: mouthkeys cancel. Last result: mouthkeys paste."
        )
        shortcut.setWordWrap(True)
        layout.addWidget(shortcut)
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText(
            "Setup check and tryout text appear here. Tryout never pastes into another app. "
            "Finish setup may be used without a tryout; real editor/terminal paste "
            "still needs validation."
        )
        layout.addWidget(self.result)

    def check_setup(self):
        import json

        from .delivery import doctor

        self.result.setPlainText(json.dumps(doctor(Config.load()), indent=2))

    def open_settings(self):
        self.window.nav.setCurrentRow(1)
        self.window.settings_category.setCurrentText("Speech")

    def invalidate_tryout(self, *args):
        self.tryout_generation += 1
        self.tryout_id = None

    def refresh_other_banner(self):
        banner = getattr(self.window, "regional_banner", None)
        if banner is not None:
            banner.refresh()

    def filter_languages(self, query):
        query = query.strip().casefold()
        for index in range(self.language.count()):
            code = self.language.itemData(index)
            searchable = " ".join([self.language.itemText(index), *LANGUAGE_ALIASES.get(code, [])])
            self.language.model().item(index).setEnabled(query in searchable.casefold())
        if query and not self.language.model().item(self.language.currentIndex()).isEnabled():
            for index in range(self.language.count()):
                if self.language.model().item(index).isEnabled():
                    self.language.setCurrentIndex(index)
                    break

    def refresh_routes(self, *args):
        self.invalidate_tryout()
        self.route.blockSignals(True)
        try:
            self.route.clear()
            for route in routes(self.language.currentData()):
                self.route.addItem(route["label"], route)
        finally:
            self.route.blockSignals(False)

    def apply_route(self):
        route = self.route.currentData()
        if not route:
            return
        self.invalidate_tryout()
        values = {key: route[key] for key in ("engine", "model", "language", "model_revision")}
        # Compute types are backend-specific; use the chosen backend's default.
        values["compute_type"] = "default"
        replace(Config.load(), **values).save()
        for key, value in values.items():
            field = self.window.fields[key]
            if isinstance(field, QComboBox):
                if field.isEditable():
                    field.setEditText(value)
                else:
                    field.setCurrentIndex(field.findData(value))
            else:
                field.setText(value)
        self.result.setPlainText(
            "Language and model route saved. Choose speech model to download explicitly, "
            "then record a new tryout."
        )

    def record(self):
        generation = self.tryout_generation

        def started(response):
            if "result" in response and generation == self.tryout_generation:
                self.tryout_id = response["result"]["id"]
                self.result.setPlainText("Listening. Stop when finished; no desktop delivery.")

        self.window.send("start", {"mode": "training"}, callback=started)

    def update_status(self, status):
        if self.tryout_id and status.get("id") == self.tryout_id:
            proposal = status.get("assistant_proposal")
            if proposal and proposal.get("mode") == "training":
                self.result.setPlainText(proposal["text"])
                self.regional_banner.refresh(dictation=proposal["text"])
                self.tryout_id = None
            elif status.get("phase") == "idle":
                self.result.setPlainText(status.get("message", "Tryout finished"))
                self.tryout_id = None

    def finish(self):
        replace(Config.load(), onboarding_complete="on").save()
        field = self.window.fields["onboarding_complete"]
        field.setCurrentIndex(field.findData("on"))
        self.result.setPlainText(
            "Setup preferences saved. Start from your editor using the shortcut and validate "
            "paste, focus, cancellation, and microphone readiness on your desktop."
        )
