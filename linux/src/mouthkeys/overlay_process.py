"""Singleton process and wake socket for the passive panel."""

import fcntl
import socket
import sys
from contextlib import suppress

from PySide6.QtCore import QSocketNotifier
from PySide6.QtWidgets import QApplication

from .config import runtime_dir
from .overlay import Overlay


def launch() -> int:
    # A separate lifetime means closing Settings never removes shortcut feedback.
    root = runtime_dir()
    with (root / "overlay.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        app = QApplication(sys.argv)
        app.setApplicationName("MouthKeys")
        app.setQuitOnLastWindowClosed(False)
        panel = Overlay()
        path = root / "overlay-wake.sock"
        path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as wake:
            try:
                wake.bind(str(path))
                path.chmod(0o600)
                wake.setblocking(False)  # noqa: FBT003 - socket positional flag
                notifier = QSocketNotifier(wake.fileno(), QSocketNotifier.Type.Read)

                def awakened(*_: object) -> None:
                    # Empty the queue before polling, so repeated wakes don't queue Qt work.
                    with suppress(BlockingIOError):
                        while wake.recv(64):
                            pass
                    panel.timer.setInterval(200)
                    panel.poll()

                notifier.activated.connect(awakened)
                try:
                    panel.poll()
                    return app.exec()
                finally:
                    notifier.setEnabled(False)
            finally:
                path.unlink(missing_ok=True)
