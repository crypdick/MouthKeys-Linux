"""Focused Qt shortcut editor; actual global ownership belongs to configured backend."""

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QKeySequenceEdit,
    QLabel,
    QVBoxLayout,
)


class ShortcutCapture(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Capture shortcut")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Press a keyboard chord here. Mouse buttons and multiple primary chords "
            "can also be entered directly in App Settings. Desktop/i3 bindings remain "
            "unchanged until an integrated backend is explicitly enabled."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.target = QComboBox()
        for key, label in (
            ("toggle_shortcut", "Primary dictation"),
            ("cancel_shortcut", "Cancel"),
            ("paste_shortcut_key", "Paste last"),
            ("reprocess_shortcut_key", "Reprocess"),
            ("command_shortcut_key", "Command Mode"),
            ("edit_shortcut_key", "Edit Mode"),
        ):
            self.target.addItem(label, key)
        layout.addWidget(self.target)
        self.sequence = QKeySequenceEdit()
        self.sequence.setMaximumSequenceLength(1)
        layout.addWidget(self.sequence)
        self.feedback = QLabel(
            "Captured shortcut is applied to the editor; Save settings to persist."
        )
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        from .hotkeys import parse_chord

        chord = (
            self.sequence.keySequence().toString(QKeySequence.SequenceFormat.PortableText).lower()
        )
        try:
            parse_chord(chord)
        except ValueError as exc:
            self.feedback.setText(str(exc))
            return
        self.window.fields[self.target.currentData()].setText(chord)
        self.accept()
