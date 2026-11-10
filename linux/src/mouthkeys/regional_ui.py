"""Inline regional offer; never opens a modal or takes desktop focus."""

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .config import Config
from .regional import region_offer


class RegionalFillerBanner(QWidget):
    def __init__(self, window, on_change=None):
        super().__init__(window)
        self.window = window
        self.on_change = on_change
        self.offer = None
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.keep = QPushButton()
        self.remove = QPushButton()
        for button, keep in ((self.keep, True), (self.remove, False)):
            button.setAutoDefault(False)
            button.clicked.connect(lambda checked=False, keep=keep: self.answer(keep))
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self, dictation=None):
        self.offer = region_offer(Config.load(), dictation=dictation)
        if self.offer:
            word = self.offer["word"]
            self.message.setText(self.offer["message"])
            self.keep.setText(f'Keep "{word}"')
            self.remove.setText(f'Keep removing "{word}"')
        self.setVisible(self.offer is not None)

    def answer(self, keep):
        if not self.offer:
            return
        current = Config.load()
        # A different surface may already have answered while this banner was open.
        if current.regional_filler_answered == "on":
            self.refresh()
            return
        words = current.filler_words
        if keep:
            words = ",".join(
                word for word in words.split(",") if word.strip().casefold() != self.offer["word"]
            )
        replace(current, filler_words=words, regional_filler_answered="on").save()
        fields = self.window.fields
        fields["filler_words"].setText(words)
        answered = fields["regional_filler_answered"]
        answered.setCurrentIndex(answered.findData("on"))
        self.refresh()
        if self.on_change:
            self.on_change()
