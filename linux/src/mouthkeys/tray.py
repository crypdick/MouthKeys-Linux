"""Qt system tray, using the same controller commands as desktop shortcuts."""

import fcntl
import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .config import Config, runtime_dir
from .desktop import source_names
from .gui import Request


class Tray(QSystemTrayIcon):
    def __init__(self, window=None):
        super().__init__(QIcon(str(Path(__file__).parent / "assets" / "icon.png")), window)
        self.window = window
        self.jobs = set()
        self.pending = False
        self.quitting = False
        self.last_microphone = None
        self.last_notice = None
        self.completed_id = None
        self.observed_status = False
        menu = QMenu()
        self.status_action = menu.addAction("MouthKeys — Ready")
        self.status_action.setEnabled(False)
        menu.addSeparator()
        self.toggle_action = menu.addAction("Record / Stop", lambda: self.send("toggle"))
        menu.addAction("Cancel recording", lambda: self.send("cancel"))
        menu.addAction("Copy last transcript", lambda: self.send("copy"))
        menu.addAction("Reprocess last recording", lambda: self.send("reprocess"))
        self.microphones = menu.addMenu("Microphone")
        self.microphones.aboutToShow.connect(self.refresh_microphones)
        menu.addSeparator()
        menu.addAction("History / App Settings", self.open_window)
        menu.addAction("Stop recording service", lambda: self.send("shutdown"))
        menu.addAction("Quit tray", self.request_quit)
        self.setContextMenu(menu)
        self.menu = menu
        self.activated.connect(self.activate)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1000)
        self.setToolTip("MouthKeys")

    def request_quit(self):
        if self.window is not None and (
            self.window.jobs
            or self.window.download is not None
            or getattr(self.window, "file_process", None) is not None
        ):
            self.setToolTip("Wait for current settings operation before quitting")
            return
        self.quitting = True
        self.timer.stop()
        if not self.jobs:
            QApplication.instance().quit()

    def activate(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.open_window()

    def open_window(self):
        if self.window is None:
            from .gui import Window

            self.window = Window()
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def refresh_microphones(self):
        self.microphones.clear()
        config = Config.load()
        try:
            names = ["default", *source_names()]
        except Exception:
            names = ["default"]
        excluded = {s.strip() for s in config.microphone_never.split(",")}
        for name in names:
            if name in excluded:
                continue
            action = self.microphones.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == config.audio_source)
            action.triggered.connect(
                lambda checked=False, name=name: replace(Config.load(), audio_source=name).save()
            )

    def send(self, action, callback=None):
        job = Request(action, parent=self)
        self.jobs.add(job)

        def finished(response):
            if "error" in response:
                self.setToolTip("MouthKeys — " + response["error"])
            if callback:
                callback(response)

        job.done.connect(finished)

        def settled():
            self.jobs.discard(job)
            job.deleteLater()
            if self.quitting and not self.jobs:
                QApplication.instance().quit()

        job.finished.connect(settled)
        job.start()

    def poll(self):
        if self.pending:
            return
        self.pending = True

        def received(response):
            self.pending = False
            result = response.get("result", {})
            phase = result.get("phase", "disconnected")
            text = "MouthKeys — " + result.get("message", response.get("error", "Unavailable"))
            self.status_action.setText(text)
            self.setToolTip(text)
            self.toggle_action.setText("Stop recording" if phase == "recording" else "Record")
            self.toggle_action.setEnabled(
                result.get("can_record", phase == "idle") or phase == "recording"
            )
            self.notify_status(result)

        self.send("status", received)

    def notify_status(self, result):
        config = Config.load()
        microphone = result.get("microphone")
        if (
            getattr(config, "show_microphone_alerts", "off") == "on"
            and self.observed_status
            and microphone
            and self.last_microphone
            and microphone != self.last_microphone
        ):
            self.showMessage("Microphone changed", microphone)
        notice = result.get("notice") or {}
        if (
            getattr(config, "show_ai_alerts", "off") == "on"
            and self.observed_status
            and notice.get("kind") == "ai"
            and notice.get("serial") != self.last_notice
        ):
            self.showMessage("AI processing failed", notice.get("message", ""))
        if (
            getattr(config, "show_microphone_alerts", "off") == "on"
            and notice.get("serial") != self.last_notice
            and any(word in notice.get("message", "").lower() for word in ("microphone", "source"))
        ):
            self.showMessage("Microphone unavailable", notice["message"])
        item_id = result.get("id")
        if result.get("phase") == "idle" and item_id != self.completed_id:
            message = result.get("message", "")
            if (
                getattr(config, "show_paste_alerts", "off") == "on"
                and self.observed_status
                and item_id
                and result.get("preview")
            ):
                if message.startswith(("Insertion requested", "Paste requested")):
                    self.showMessage(
                        "Paste requested",
                        "Destination acceptance cannot be verified. "
                        "Transcript remains available in History.",
                    )
                elif message.startswith("Copied") and config.delivery != "clipboard":
                    self.showMessage(
                        "Manual paste needed",
                        "Automatic insertion was not requested. Transcript copied to clipboard.",
                    )
            self.completed_id = item_id
        self.last_microphone = microphone
        self.last_notice = notice.get("serial")
        self.observed_status = True


def create_tray(window=None):
    if Config.load().tray != "on" or not QSystemTrayIcon.isSystemTrayAvailable():
        return None
    tray = Tray(window)
    tray.show()
    return tray


def launch():
    from .service import ensure_daemon, ensure_overlay

    with (runtime_dir() / "tray.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        ensure_daemon()
        ensure_overlay()
        app = QApplication(sys.argv)
        app.setApplicationName("MouthKeys")
        app.setQuitOnLastWindowClosed(False)
        tray = create_tray()
        if tray is None:
            raise RuntimeError("System tray unavailable or disabled; run mouthkeys gui")
        return app.exec()
