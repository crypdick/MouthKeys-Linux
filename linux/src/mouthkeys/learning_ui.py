"""Review real correction observations, edit vocabulary, and enroll pronunciation takes."""

from collections.abc import Mapping
from typing import NamedTuple

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .pronunciation import PronunciationStore
from .qt_control import Response, Send
from .vocabulary import VocabularyEntry, VocabularyStore


class ReviewRow(NamedTuple):
    id: str
    heard: str
    corrected: str


def _review_rows(value: object) -> list[ReviewRow]:
    rows = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            identity, heard, corrected = item.get("id"), item.get("heard"), item.get("corrected")
            if isinstance(identity, str) and isinstance(heard, str) and isinstance(corrected, str):
                rows.append(ReviewRow(identity, heard, corrected))
    return rows


class LearningPage(QWidget):
    def __init__(self, send: Send) -> None:
        super().__init__()
        self._send = send
        self.pending: list[ReviewRow] = []
        self.enrollment_id: str | None = None
        self._start_pending = False
        self._pending_take_action: str | None = None
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
            ("Accept correction", lambda: self.review(accept=True)),
            ("Dismiss correction", lambda: self.review(accept=False)),
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
            ("Stop take", lambda: self._take_action("stop")),
            ("Cancel", lambda: self._take_action("cancel")),
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
        self.vocabulary.setHorizontalHeaderLabels(
            ["Word or phrase", "Weight (blank = default)", "Aliases"]
        )
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

    def refresh_words(self) -> None:
        self.base_entries = VocabularyStore.entries()
        self.vocabulary.setRowCount(0)
        for entry in self.base_entries:
            self.add_word(entry=entry)

    def add_word(self, *, entry: VocabularyEntry | None = None) -> None:
        entry = entry or {"text": "", "weight": 1, "aliases": []}
        row = self.vocabulary.rowCount()
        self.vocabulary.insertRow(row)
        self.vocabulary.setItem(row, 0, QTableWidgetItem(entry["text"]))
        weight = QLineEdit("" if entry["weight"] is None else str(entry["weight"]))
        weight.setPlaceholderText("Default")
        weight.setToolTip("Optional positive floating-point weight. Up to 256 vocabulary terms.")
        self.vocabulary.setCellWidget(row, 1, weight)
        self.vocabulary.setItem(row, 2, QTableWidgetItem(", ".join(entry["aliases"])))

    def remove_word(self) -> None:
        row = self.vocabulary.currentRow()
        if row >= 0:
            self.vocabulary.removeRow(row)

    def _read_word(self, row: int) -> VocabularyEntry:
        text = self.vocabulary.item(row, 0)
        weight = self.vocabulary.cellWidget(row, 1)
        aliases = self.vocabulary.item(row, 2)
        if text is None or aliases is None or not isinstance(weight, QLineEdit):
            raise ValueError("Vocabulary row is incomplete")
        number = weight.text().strip()
        return {
            "text": text.text(),
            "weight": float(number) if number else None,
            "aliases": [alias for value in aliases.text().split(",") if (alias := value.strip())],
        }

    def save_words(self) -> None:
        try:
            entries = [self._read_word(row) for row in range(self.vocabulary.rowCount())]
            VocabularyStore.save_entries(entries, expected=self.base_entries)
            self.refresh_words()
            self.feedback.setText(
                "Vocabulary saved. Enable vocabulary boosting in App Settings to use it."
            )
        except (OSError, ValueError) as exc:
            self.feedback.setText(str(exc))

    def refresh_profiles(self) -> None:
        self.profiles.clear()
        for profile in PronunciationStore.load():
            count = len(profile["samples"])
            self.profiles.addItem(
                f"{profile['label']} · {count} takes · "
                f"{'ready' if count >= 2 else 'needs another take'}",
                profile["label"],
            )

    def enroll(self) -> None:
        if self._start_pending:
            return
        self._start_pending = True
        self._send(
            "start", {"mode": "pronunciation", "label": self.label.text()}, self._training_started
        )

    def _training_started(self, response: Response) -> None:
        self._start_pending = False
        action, self._pending_take_action = self._pending_take_action, None
        result = response.get("result")
        identity = result.get("id") if isinstance(result, dict) else None
        if not isinstance(identity, str) or not identity:
            self.feedback.setText(str(response.get("error", "Cannot start pronunciation take")))
            return
        self.enrollment_id = identity
        self.feedback.setText("Listening. Say only the isolated phrase, then Stop take.")
        if action is not None:
            self._take_action(action)

    def _take_action(self, action: str) -> None:
        if self._start_pending:
            self._pending_take_action = action
        elif self.enrollment_id is not None:
            self._send(action, {"id": self.enrollment_id}, None)

    def _current_proposal(self) -> ReviewRow | None:
        row = self.proposals.currentRow()
        return self.pending[row] if 0 <= row < len(self.pending) else None

    def _update_reviews(self, proposals: list[ReviewRow]) -> None:
        if proposals == self.pending:
            return
        selected = self._current_proposal()
        self.pending = proposals
        self.proposals.setRowCount(len(proposals))
        selection = -1
        for row, item in enumerate(proposals):
            self.proposals.setItem(row, 0, QTableWidgetItem(item.heard))
            self.proposals.setItem(row, 1, QTableWidgetItem(item.corrected))
            if selected is not None and selected.id == item.id:
                selection = row
        self.proposals.setCurrentCell(selection, 0 if selection >= 0 else -1)

    def update_status(self, status: Mapping[str, object]) -> None:
        self._update_reviews(_review_rows(status.get("learning_pending")))
        if self.enrollment_id is None or status.get("id") != self.enrollment_id:
            return
        proposal = status.get("assistant_proposal")
        if isinstance(proposal, dict) and proposal.get("mode") == "pronunciation":
            self.refresh_profiles()
            self.feedback.setText("Take saved. Two consistent takes required before matching.")
            self.enrollment_id = None
        elif status.get("phase") == "idle":
            self.feedback.setText(str(status.get("message", "Training finished")))
            self.enrollment_id = None

    def review(self, *, accept: bool) -> None:
        if proposal := self._current_proposal():
            self._send(
                "learning_accept" if accept else "learning_dismiss", {"id": proposal.id}, None
            )

    def delete_pronunciation(self) -> None:
        label: object = self.profiles.currentData()
        if (
            isinstance(label, str)
            and label
            and QMessageBox.question(self, "Delete pronunciation", f"Delete all takes for {label}?")
            == QMessageBox.StandardButton.Yes
        ):
            PronunciationStore.delete(label)
            self.refresh_profiles()
