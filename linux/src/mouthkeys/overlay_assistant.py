"""Explicitly opened, focusable assistant surface for the passive recording panel."""

from collections.abc import Callable, Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from .assistant_ui import AssistantPage
from .overlay_rendering import PanelColors
from .qt_control import Request


class OverlayAssistant(QWidget):
    def __init__(self, colors: PanelColors, refresh: Callable[[], None]) -> None:
        # Separate window: the recording panel never changes its no-focus flags.
        super().__init__(None, Qt.WindowType.Tool)
        self.setWindowTitle("MouthKeys · Assistant")
        self.resize(760, 820)
        self.refresh = refresh
        self.jobs: set[Request] = set()
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.assistant_page = AssistantPage(self)
        scroll.setWidget(self.assistant_page)
        layout.addWidget(scroll)
        self.apply_theme(colors)

    def apply_theme(self, colors: PanelColors) -> None:
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, colors.surface),
            (QPalette.ColorRole.Base, colors.surface),
            (QPalette.ColorRole.Button, colors.surface),
            (QPalette.ColorRole.WindowText, colors.ink),
            (QPalette.ColorRole.Text, colors.ink),
            (QPalette.ColorRole.ButtonText, colors.ink),
            (QPalette.ColorRole.Highlight, colors.ink),
            (QPalette.ColorRole.HighlightedText, colors.surface),
        ):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)

    def send(
        self,
        action: str,
        data: Mapping[str, object] | None = None,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        job = Request(action, data, parent=self)
        self.jobs.add(job)

        def received(response: dict[str, object]) -> None:
            if callback:
                callback(response)
            elif "error" in response:
                self.assistant_page.feedback.setText(str(response["error"]))
            self.refresh()

        job.done.connect(received)
        job.finished.connect(lambda: self.jobs.discard(job))
        job.finished.connect(job.deleteLater)
        job.start()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        # Closing collapses the surface without destroying provider/command workers.
        self.hide()
        event.ignore()
