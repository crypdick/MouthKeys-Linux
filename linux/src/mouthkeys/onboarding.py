"""Guided local setup and an explicit, non-delivering dictation tryout."""

from dataclasses import replace

from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from .config import Config


class SetupPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.tryout_id = None
        layout = QVBoxLayout(self)
        note = QLabel(
            "1. Check desktop dependencies. 2. Select and explicitly download a speech model. "
            "3. Bind your desktop shortcut. 4. Try dictation here. "
            "Local dictation needs no account; cloud cleanup is optional."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
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

    def record(self):
        def started(response):
            if "result" in response:
                self.tryout_id = response["result"]["id"]
                self.result.setPlainText("Listening. Stop when finished; no desktop delivery.")

        self.window.send("start", {"mode": "training"}, callback=started)

    def update_status(self, status):
        if self.tryout_id and status.get("id") == self.tryout_id:
            proposal = status.get("assistant_proposal")
            if proposal and proposal.get("mode") == "training":
                self.result.setPlainText(proposal["text"])
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
