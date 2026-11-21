"""Local diagnostics and support links, opened only on request."""

import platform
from os import SEEK_END
from urllib.parse import urlencode

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from . import __version__
from .config import state_dir

LOG_TAIL_BYTES = 100_000


class HelpPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        note = QLabel(
            "Bind mouthkeys toggle / start / stop / cancel / paste"
            " in i3 or desktop keyboard settings. Hold-to-talk can use "
            "separate key-press start and key-release stop bindings. Optional "
            "speech engines need their package extras and explicit model "
            "downloads. Open README for compositor-specific setup."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        for label, handler in (
            ("Show diagnostic log", self.show_log),
            ("Report issue", self.report_issue),
            ("Open Linux guide", self.open_guide),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            layout.addWidget(button)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def show_log(self) -> None:
        try:
            with (state_dir() / "daemon.log").open("rb") as stream:
                size = stream.seek(0, SEEK_END)
                stream.seek(max(0, size - LOG_TAIL_BYTES))
                text = stream.read(LOG_TAIL_BYTES).decode(errors="replace")
        except FileNotFoundError:
            text = "No diagnostic log yet"
        except OSError:
            text = "Could not read the diagnostic log"
        self.log_text.setPlainText(text)

    def report_issue(self) -> None:
        url = "https://github.com/crypdick/MouthKeys-Linux/issues/new?" + urlencode(
            {"body": f"MouthKeys {__version__}\nLinux {platform.release()}\n\nDescribe issue:\n"}
        )
        QDesktopServices.openUrl(QUrl(url))

    def open_guide(self) -> None:
        QDesktopServices.openUrl(
            QUrl("https://github.com/crypdick/MouthKeys-Linux/blob/main/linux/README.md")
        )
