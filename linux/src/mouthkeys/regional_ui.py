"""Regional filler settings, shared by the passive banner and explicit Settings prompt."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import Config
from .regional import UPSTREAM_FILLERS, RegionalOffer, region_offer
from .settings_fields import SettingsField


class RegionalFillerBanner(QWidget):
    def __init__(
        self,
        parent: QWidget,
        fields: Mapping[str, SettingsField],
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.fillers = cast("QLineEdit", fields["filler_words"])
        self.answered = cast("QComboBox", fields["regional_filler_answered"])
        self.on_change = on_change
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
            button.clicked.connect(lambda _checked=False, keep=keep: self.answer(keep=keep))
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self, dictation: str | None = None) -> None:
        self.offer = region_offer(Config.load(), dictation=dictation)
        if self.offer:
            word = self.offer["word"]
            self.message.setText(self.offer["message"])
            self.keep.setText(f'Keep "{word}"')
            self.remove.setText(f'Keep removing "{word}"')
        self.setVisible(self.offer is not None)

    def reset_defaults(self, show_status: Callable[[str], None]) -> None:
        self.fillers.setText(UPSTREAM_FILLERS)
        show_status("Upstream filler list restored in editor; Save settings to apply")

    def show_suggestion(self, show_status: Callable[[str], None]) -> None:
        offer = region_offer(Config.load())
        if offer is None:
            show_status("No unanswered regional filler suggestion applies to current settings")
            return
        choice = QMessageBox.question(self.window(), "Regional filler", offer["message"])
        self._save_answer(offer, keep=choice == QMessageBox.StandardButton.Yes)

    def answer(self, *, keep: bool) -> None:
        if self.offer is not None:
            self._save_answer(self.offer, keep=keep)

    def _save_answer(self, offer: RegionalOffer, *, keep: bool) -> None:
        current = Config.load()
        # The prompt may have been open while another editor saved settings or answered it.
        if current.regional_filler_answered == "on":
            self.refresh()
            return
        words = current.filler_words
        if keep:
            words = ",".join(
                word for word in words.split(",") if word.strip().casefold() != offer["word"]
            )
        replace(current, filler_words=words, regional_filler_answered="on").save()
        self.fillers.setText(words)
        self.answered.setCurrentIndex(self.answered.findData("on"))
        self.refresh()
        if self.on_change:
            self.on_change()
