"""Explicitly opened, focusable assistant surface for the passive recording panel."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from .assistant_ui import AssistantPage
from .gui import Request


class OverlayAssistant(QWidget):
    def __init__(self, panel):
        # Separate window: the recording panel never changes its no-focus flags.
        super().__init__(None, Qt.WindowType.Tool)
        self.setWindowTitle("MouthKeys · Assistant")
        self.resize(760, 820)
        self.panel = panel
        self.jobs = set()
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.assistant_page = AssistantPage(self)
        scroll.setWidget(self.assistant_page)
        layout.addWidget(scroll)
        self.apply_theme()

    def apply_theme(self):
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, self.panel.surface),
            (QPalette.ColorRole.Base, self.panel.surface),
            (QPalette.ColorRole.Button, self.panel.surface),
            (QPalette.ColorRole.WindowText, self.panel.ink),
            (QPalette.ColorRole.Text, self.panel.ink),
            (QPalette.ColorRole.ButtonText, self.panel.ink),
            (QPalette.ColorRole.Highlight, self.panel.ink),
            (QPalette.ColorRole.HighlightedText, self.panel.surface),
        ):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)

    def send(self, action, data=None, callback=None):
        job = Request(action, data or {}, parent=self)
        self.jobs.add(job)

        def received(response):
            if callback:
                callback(response)
            elif "error" in response:
                self.assistant_page.feedback.setText(response["error"])
            self.panel.poll()

        job.done.connect(received)
        job.finished.connect(lambda: self.jobs.discard(job))
        job.finished.connect(job.deleteLater)
        job.start()

    def closeEvent(self, event):
        # Closing collapses the surface without destroying provider/command workers.
        self.hide()
        event.ignore()
