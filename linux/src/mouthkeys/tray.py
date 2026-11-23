"""Qt system tray, using the same controller commands as desktop shortcuts."""

import socket
import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSocketNotifier, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon, QWidget

from .config import Config
from .control import ensure_daemon, ensure_overlay
from .desktop import source_names
from .qt_control import Request, Response, ResponseHandler
from .tray_status import TrayAlerts, TrayStatus, status_text


class Tray(QSystemTrayIcon):
    def __init__(self) -> None:
        super().__init__(QIcon(str(Path(__file__).parent / "assets" / "icon.png")))
        self.window: QWidget | None = None
        self.jobs: set[Request] = set()
        self.pending = False
        self.quitting = False
        self.alerts = TrayAlerts()
        menu = QMenu()
        self.status_action = menu.addAction("MouthKeys — Ready")
        self.status_action.setEnabled(False)
        menu.addSeparator()
        self.toggle_action = menu.addAction("Record / Stop", lambda: self.send("toggle"))
        menu.addAction("Cancel recording", lambda: self.send("cancel"))
        menu.addAction("Copy last transcript", lambda: self.send("copy"))
        self.microphones = menu.addMenu("Microphone")
        self.microphones.aboutToShow.connect(self.refresh_microphones)
        menu.addSeparator()
        menu.addAction("History / App Settings", self.open_window)
        menu.addAction("Quit MouthKeys", self.request_quit)
        self.setContextMenu(menu)
        self.menu = menu
        self.activated.connect(self.activate)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1000)
        self.setToolTip("MouthKeys")

    def request_quit(self) -> None:
        if getattr(self.window, "busy", False):
            self.setToolTip("Wait for current settings operation before quitting")
            return
        self.finish_quitting()

    def finish_quitting(self) -> None:
        self.quitting = True
        self.timer.stop()
        if not self.jobs and (app := QApplication.instance()) is not None:
            app.quit()

    def activate(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_window()

    def open_window(self) -> None:
        if self.window is None:
            from .gui import Window  # noqa: PLC0415 - open the settings window only on request

            self.window = Window()
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def refresh_microphones(self) -> None:
        self.microphones.clear()
        config = Config.load()
        try:
            names = ["default", *source_names()]
        # The microphone menu remains usable when enumeration fails.
        except Exception:  # allow: exception-handling  # noqa: BLE001
            names = ["default"]
        excluded = {s.strip() for s in config.microphone_never.split(",")}
        for name in names:
            if name in excluded:
                continue
            action = self.microphones.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == config.audio_source)
            action.triggered.connect(
                lambda _checked=False, name=name: replace(Config.load(), audio_source=name).save()
            )

    def send(self, action: str, callback: ResponseHandler | None = None) -> None:
        job = Request(action, parent=self)
        self.jobs.add(job)

        def finished(response: Response) -> None:
            if isinstance(error := response.get("error"), str):
                self.setToolTip("MouthKeys — " + error)
            if callback:
                callback(response)

        job.done.connect(finished)

        def settled() -> None:
            self.jobs.discard(job)
            job.deleteLater()
            if self.quitting and not self.jobs and (app := QApplication.instance()) is not None:
                app.quit()

        job.finished.connect(settled)
        job.start()

    def poll(self) -> None:
        if not self.pending:
            self.pending = True
            self.send("status", self.notify_status)

    def notify_status(self, response: Response) -> None:
        self.pending = False
        status = TrayStatus.parse(response.get("result"))
        if status is None:
            message = status_text(response, "error", "Unavailable")
            label, enabled = "Record", False
        else:
            message, label, enabled = status.message, status.record_label, status.can_record
        text = "MouthKeys — " + message
        self.status_action.setText(text)
        self.setToolTip(text)
        self.toggle_action.setText(label)
        self.toggle_action.setEnabled(enabled)
        if status is not None:
            for title, detail in self.alerts.observe(Config.load(), status):
                self.showMessage(title, detail)


def create_tray() -> Tray | None:
    if Config.load().tray != "on" or not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = Tray()
    tray.show()
    return tray


def launch(service_fd: int | None = None) -> int:
    if service_fd is None:
        ensure_daemon()
        return 0
    with socket.socket(fileno=service_fd) as service:
        app = QApplication(sys.argv)
        app.setApplicationName("MouthKeys")
        app.setQuitOnLastWindowClosed(False)
        tray = create_tray()
        if tray is None:
            return 0
        disconnect = QSocketNotifier(service.fileno(), QSocketNotifier.Type.Read)

        def service_stopped(_descriptor: object, _event: object) -> None:
            disconnect.setEnabled(False)
            tray.finish_quitting()

        disconnect.activated.connect(service_stopped)
        ensure_overlay()
        service.sendall(b"1")
        return app.exec()
