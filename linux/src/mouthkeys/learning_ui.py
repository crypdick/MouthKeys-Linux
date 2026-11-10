"""Review real correction observations, edit vocabulary, and enroll pronunciation takes."""

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LearningPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.pending = []
        self.enrollment_id = None
        layout = QVBoxLayout(self)
        note = QLabel(
            "Automatic learning is opt-in in App Settings. Only actual corrections in a verified "
            "editable field are observed; surrounding field text is not saved. "
            "Review suggestions here."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.proposals = QTableWidget(0, 2)
        self.proposals.setHorizontalHeaderLabels(["Heard", "Corrected"])
        self.proposals.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.proposals)
        actions = QHBoxLayout()
        for label, action in (
            ("Accept correction", lambda: self.review(True)),
            ("Dismiss correction", lambda: self.review(False)),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            actions.addWidget(button)
        layout.addLayout(actions)
        form = QFormLayout()
        self.label = QLineEdit()
        self.label.setPlaceholderText("Correct isolated phrase (1–5 words)")
        form.addRow("Pronunciation", self.label)
        layout.addLayout(form)
        training = QHBoxLayout()
        for label, action in (
            ("Record pronunciation take", self.enroll),
            ("Stop take", lambda: window.send("stop")),
            ("Cancel", lambda: window.send("cancel")),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            training.addWidget(button)
        layout.addLayout(training)
        self.profiles = QComboBox()
        layout.addWidget(self.profiles)
        delete = QPushButton("Delete selected pronunciation")
        delete.clicked.connect(self.delete_pronunciation)
        layout.addWidget(delete)
        self.feedback = QLabel(
            "Enroll at least two consistent takes. Matching uses acoustic templates, "
            "not a calibrated ASR probability. Matches use actual word-timed spans "
            "where available; "
            "engines without word timing only match an isolated short phrase."
        )
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.refresh_profiles()
        layout.addWidget(
            QLabel("Recognition vocabulary — weight biases native context where supported")
        )
        self.vocabulary = QTableWidget(0, 3)
        self.vocabulary.setHorizontalHeaderLabels(["Word or phrase", "Weight (1–5)", "Aliases"])
        self.vocabulary.horizontalHeader().setStretchLastSection(True)
        self.vocabulary.setMinimumHeight(150)
        layout.addWidget(self.vocabulary)
        buttons = QHBoxLayout()
        for label, action in (
            ("Add word", self.add_word),
            ("Remove word", self.remove_word),
            ("Save vocabulary", self.save_words),
            ("Refresh vocabulary", self.refresh_words),
        ):
            button = QPushButton(label)
            button.clicked.connect(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.refresh_words()

    def refresh_words(self):
        from .vocabulary import VocabularyStore

        self.base_entries = VocabularyStore.entries()
        self.vocabulary.setRowCount(0)
        for entry in self.base_entries:
            self.add_word(entry=entry)

    def add_word(self, _checked=False, entry=None):
        entry = entry or {"text": "", "weight": 1, "aliases": []}
        row = self.vocabulary.rowCount()
        self.vocabulary.insertRow(row)
        self.vocabulary.setItem(row, 0, QTableWidgetItem(entry["text"]))
        weight = QSpinBox()
        weight.setRange(1, 5)
        weight.setValue(entry["weight"])
        self.vocabulary.setCellWidget(row, 1, weight)
        self.vocabulary.setItem(row, 2, QTableWidgetItem(", ".join(entry["aliases"])))

    def remove_word(self):
        row = self.vocabulary.currentRow()
        if row >= 0:
            self.vocabulary.removeRow(row)

    def save_words(self):
        from .vocabulary import VocabularyStore

        try:
            if VocabularyStore.entries() != self.base_entries:
                raise ValueError(
                    "Vocabulary changed (possibly learned a correction). Refresh before saving."
                )
            entries = [
                {
                    "text": self.vocabulary.item(row, 0).text(),
                    "weight": self.vocabulary.cellWidget(row, 1).value(),
                    "aliases": [
                        word.strip()
                        for word in self.vocabulary.item(row, 2).text().split(",")
                        if word.strip()
                    ],
                }
                for row in range(self.vocabulary.rowCount())
            ]
            VocabularyStore.save_entries(entries)
            self.refresh_words()
            self.feedback.setText(
                "Vocabulary saved. Enable vocabulary boosting in App Settings to use it."
            )
        except Exception as exc:
            self.feedback.setText(str(exc))

    def refresh_profiles(self):
        from .pronunciation import PronunciationStore

        self.profiles.clear()
        for profile in PronunciationStore.load():
            count = len(profile["samples"])
            self.profiles.addItem(
                f"{profile['label']} · {count} takes · "
                f"{'ready' if count >= 2 else 'needs another take'}",
                profile["label"],
            )

    def enroll(self):
        def started(response):
            if "result" in response:
                self.enrollment_id = response["result"]["id"]
                self.feedback.setText("Listening. Say only the isolated phrase, then Stop take.")

        self.window.send(
            "start", {"mode": "pronunciation", "label": self.label.text()}, callback=started
        )

    def update_status(self, status):
        proposals = status.get("learning_pending", [])
        if proposals != self.pending:
            self.pending = proposals
            self.proposals.setRowCount(len(proposals))
            for row, item in enumerate(proposals):
                self.proposals.setItem(row, 0, QTableWidgetItem(item["heard"]))
                self.proposals.setItem(row, 1, QTableWidgetItem(item["corrected"]))
        if self.enrollment_id and status.get("id") == self.enrollment_id:
            proposal = status.get("assistant_proposal")
            if proposal and proposal.get("mode") == "pronunciation":
                self.refresh_profiles()
                self.feedback.setText("Take saved. Two consistent takes required before matching.")
                self.enrollment_id = None
            elif status.get("phase") == "idle":
                self.feedback.setText(status.get("message", "Training finished"))
                self.enrollment_id = None

    def review(self, accept):
        row = self.proposals.currentRow()
        if 0 <= row < len(self.pending):
            self.window.send(
                "learning_accept" if accept else "learning_dismiss", {"id": self.pending[row]["id"]}
            )

    def delete_pronunciation(self):
        from PySide6.QtWidgets import QMessageBox

        from .pronunciation import PronunciationStore

        label = self.profiles.currentData()
        if (
            label
            and QMessageBox.question(self, "Delete pronunciation", f"Delete all takes for {label}?")
            == QMessageBox.StandardButton.Yes
        ):
            PronunciationStore.delete(label)
            self.refresh_profiles()
