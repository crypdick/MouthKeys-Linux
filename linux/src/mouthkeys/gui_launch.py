"""Start the optional Settings application and its configured desktop integrations."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .config import config_path
from .control import ensure_daemon, ensure_overlay
from .gui import Window
from .hotkeys import ensure_hotkeys
from .tray import create_tray


def launch() -> int:
    first_launch = not config_path().exists()
    ensure_daemon()
    ensure_overlay()
    ensure_hotkeys()
    app = QApplication(sys.argv)
    app.setApplicationName("MouthKeys")
    window = Window()
    if first_launch:
        window.nav.setCurrentRow(9)
    tray = create_tray(window)
    app.setQuitOnLastWindowClosed(tray is None)
    if tray is None:
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.show()
    return app.exec()
