"""Datasheet Mono recording panel; independent of the settings window."""

import fcntl
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, QSocketNotifier, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFont, QIcon, QPainter, QPen, QPixmap, QTextLayout
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QWidget

from .config import Config, runtime_dir
from .desktop import source_names
from .gui import Request
from .processes import python_command
from .trace import Trace


def button_icon(action, color):
    """Small vector marks, independent of installed symbol fonts."""
    pixmap = QPixmap(20, 20)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(color), 1.5))
    if action == "copy":
        painter.drawRect(4, 3, 10, 11)
        painter.drawRect(7, 6, 10, 11)
    elif action == "history":
        for y in (5, 10, 15):
            painter.drawLine(4, y, 16, y)
    elif action == "cancel":
        painter.drawLine(5, 5, 15, 15)
        painter.drawLine(5, 15, 15, 5)
    else:
        painter.drawArc(4, 4, 12, 12, 35 * 16, 290 * 16)
        painter.drawLine(16, 3, 16, 8)
        painter.drawLine(16, 8, 11, 8)
    painter.end()
    return QIcon(pixmap)


class Overlay(QWidget):
    """340 × 130 pill and 30 px rails, matching the native round-six geometry."""

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(424, 146)
        self.settings = Config.load()
        self.panel_scale = 1.0
        self.data = {}
        self.trace = Trace()
        self.meter_frame = None
        self.text_cache = None
        self.jobs = set()
        self.pending = False
        self.active = False
        self.dismiss_at = None
        self.positioned = False
        self.drag = None
        self.seconds = 0
        self.notice = None
        self.notice_serial = None
        self.failures = 0
        self.buttons = {}
        for action, tip, x, y in (
            ("history", "Open history", 6, 29),
            ("copy", "Copy transcript", 6, 83),
            ("cancel", "Cancel / dismiss", 388, 29),
            ("reprocess", "Reprocess retained audio", 388, 83),
        ):
            button = QPushButton(self)
            button.setIconSize(QSize(20, 20))
            button.setGeometry(x, y, 30, 30)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda checked=False, action=action: self.action(action))
            self.buttons[action] = button
        self.mic_button = QPushButton(self)
        self.mic_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mic_button.setToolTip("Choose microphone for next recording")
        self.mic_button.setAccessibleName("Choose microphone")
        self.mic_button.clicked.connect(self.choose_microphone)
        self.apply_settings(self.settings)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(200)
        self.animation = QTimer(self)
        self.animation.setTimerType(Qt.TimerType.PreciseTimer)
        self.animation.timeout.connect(self.animate)
        self.apply_theme()
        QApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)

    def apply_theme(self, *_):
        theme = self.settings.theme
        light = theme == "light" or (
            theme == "system" and QApplication.styleHints().colorScheme() == Qt.ColorScheme.Light
        )
        self.surface = "#ffffff" if light else "#111214"
        self.ink = "#111214" if light else "#ebebec"
        self.edge = "#111214" if light else "#2c2e33"
        self.dim = "#55565a" if light else "#949597"
        chip = "#f2f2f4" if light else "#1a1b1f"
        self.setStyleSheet(
            f"QPushButton {{ background: {chip}; color: {self.ink}; border: none;"
            "font-size: 20px; border-radius: 0; }"
            f"QPushButton:hover {{ border: 1px solid {self.ink}; }}"
            f"QPushButton:pressed {{ background: {self.ink}; color: {self.surface}; }}"
            f"QPushButton:disabled {{ color: {self.dim}; }}"
        )
        for action, button in self.buttons.items():
            button.setIcon(button_icon(action, self.ink))
        self.update()

    def apply_settings(self, config):
        previous = self.settings
        self.settings = config
        self.panel_scale = config.overlay_size / 100
        self.setFixedSize(round(424 * self.panel_scale), round(146 * self.panel_scale))
        for action, x, y in (
            ("history", 6, 29),
            ("copy", 6, 83),
            ("cancel", 388, 29),
            ("reprocess", 388, 83),
        ):
            self.buttons[action].setGeometry(
                round(x * self.panel_scale),
                round(y * self.panel_scale),
                round(30 * self.panel_scale),
                round(30 * self.panel_scale),
            )
        self.mic_button.setGeometry(
            round(185 * self.panel_scale),
            round(110 * self.panel_scale),
            round(108 * self.panel_scale),
            round(20 * self.panel_scale),
        )
        if previous.theme != config.theme:
            self.apply_theme()
        if (previous.overlay_position, previous.overlay_bottom_offset, previous.overlay_size) != (
            config.overlay_position,
            config.overlay_bottom_offset,
            config.overlay_size,
        ):
            self.positioned = False

    def choose_microphone(self):
        menu = QMenu(self)
        try:
            names = ["default", *source_names()]
        except Exception:
            names = ["default"]
        config = Config.load()
        excluded = {s.strip() for s in config.microphone_never.split(",")}
        for name in names:
            if name not in excluded:
                action = menu.addAction(name)
                action.triggered.connect(
                    lambda checked=False, name=name: replace(
                        Config.load(), audio_source=name
                    ).save()
                )
        menu.exec(self.mic_button.mapToGlobal(self.mic_button.rect().bottomLeft()))

    def send(self, action, callback=None, item_id=None):
        data = (
            {"id": item_id}
            if item_id
            else (
                {"id": self.data["id"]}
                if action in ("copy", "reprocess") and self.data.get("id")
                else {}
            )
        )
        job = Request(action, data, parent=self)
        self.jobs.add(job)
        if callback:
            job.done.connect(callback)
        else:

            def feedback(response):
                if "error" in response:
                    self.notice = (response["error"], time.monotonic() + 10)
                    self.dismiss_at = self.notice[1]
                    self.show()
                    self.update()

            job.done.connect(feedback)
        job.finished.connect(lambda: self.jobs.discard(job))
        job.finished.connect(job.deleteLater)
        job.start()

    def action(self, action):
        if action == "history":
            self.send("history", self.show_history)
        elif action == "cancel":
            if self.active:
                self.send("cancel")
            self.hide()
            self.dismiss_at = 0
        else:
            self.send(action)

    def show_history(self, response):
        if "error" in response:
            self.notice = (response["error"], time.monotonic() + 10)
            self.update()
            return
        menu = QMenu(self)
        rows = response.get("result", [])[:12]
        if not rows:
            menu.addAction("No history yet").setEnabled(False)
        for row in rows:
            text = row.get("text") or row.get("error") or row.get("status", "Recording")
            submenu = menu.addMenu(" ".join(text.split())[:65])
            copy = submenu.addAction("Copy transcript")
            copy.setEnabled(bool(row.get("text")))
            copy.triggered.connect(
                lambda checked=False, item_id=row["id"]: self.send("copy", item_id=item_id)
            )
            retry = submenu.addAction("Reprocess audio")
            retry.setEnabled(not self.active)
            retry.triggered.connect(
                lambda checked=False, item_id=row["id"]: self.send("reprocess", item_id=item_id)
            )
        menu.addSeparator()
        menu.addAction(
            "Open full history / settings",
            lambda: subprocess.Popen(python_command("mouthkeys", "gui"), start_new_session=True),
        )
        self.history_menu = menu
        origin = self.mapToGlobal(QPoint(round(42 * self.panel_scale), 0))
        menu.popup(origin - QPoint(0, menu.sizeHint().height()))

    def poll(self):
        if self.pending:
            return
        self.pending = True

        def received(response):
            self.pending = False
            if "result" in response:
                self.failures = 0
                self.timer.setInterval(50 if response["result"]["phase"] == "recording" else 200)
                self.set_status(response["result"])
            else:
                self.hide()  # Shutdown/disconnect must not leave a stale recording panel.
                self.active = False
                self.notice_serial = None
                self.animation.stop()
                self.failures += 1
                self.timer.setInterval(min(5000, 200 * 2 ** min(self.failures, 5)))

        self.send("status", received)

    def place(self):
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        y = (
            area.top() + self.settings.overlay_bottom_offset
            if self.settings.overlay_position == "top"
            else area.bottom() - self.height() - self.settings.overlay_bottom_offset
        )
        self.move(area.center().x() - self.width() // 2, y)
        self.positioned = True

    def set_status(self, data):
        config = Config.load()
        self.apply_settings(config)
        if config.recording_panel == "off":
            self.hide()
            self.active = False
            self.notice_serial = None
            return
        active = active_phase(data["phase"])
        new_recording = data.get("id") != self.data.get("id")
        if active and (not self.active or new_recording):
            self.dismiss_at = None
            self.notice = None
            if not self.positioned:
                self.place()
            self.show()  # Never activate or raise the destination app.
        if not active and self.active:
            message = data.get("message", "")
            success = message.startswith(("Copied —", "Paste requested —"))
            self.dismiss_at = time.monotonic() + (0.6 if success else 10)
            if message == "Cancelled":
                self.dismiss_at = 0
        notice = data.get("notice")
        if not notice:
            self.notice = None
            self.notice_serial = None
        if notice and notice["serial"] != self.notice_serial:
            self.notice_serial = notice["serial"]
            if time.time() - notice["at"] < 10:
                self.notice = (notice["message"], time.monotonic() + 10)
                self.dismiss_at = self.notice[1]
                if not self.positioned:
                    self.place()
                self.show()
        if self.dismiss_at is not None and time.monotonic() >= self.dismiss_at:
            self.hide()
        now = time.monotonic()
        if data["phase"] == "recording":
            if new_recording or self.data.get("phase") != "recording":
                self.trace.begin(now)
                self.meter_frame = None
            meter = data.get("meter", {})
            if meter.get("frame") is not None and meter["frame"] != self.meter_frame:
                self.trace.ingest(
                    meter["level"] * config.visualizer_sensitivity / 100, min(now, meter["at"])
                )
                self.meter_frame = meter["frame"]
        else:
            self.trace.stop(now)
        if active and not self.animation.isActive():
            self.animation.start(16)
        self.active = active
        self.data = data
        self.mic_button.setText(data.get("microphone", "default"))
        self.mic_button.setEnabled(not active)
        if data["phase"] == "recording":
            self.seconds = data.get("seconds", 0)
        self.buttons["copy"].setEnabled(not active and bool(data.get("preview")))
        self.buttons["reprocess"].setEnabled(not active)
        self.update()

    def animate(self):
        if not self.isVisible():
            self.animation.stop()
            return
        now = time.monotonic()
        self.trace.advance(now)
        self.update(
            QRect(
                round(53 * self.panel_scale),
                round(68 * self.panel_scale),
                round(252 * self.panel_scale),
                round(40 * self.panel_scale),
            )
        )
        if (
            self.data.get("phase") == "idle"
            and self.trace.stopped_at is not None
            and now - self.trace.stopped_at >= 0.06
        ):
            self.animation.stop()

    def paintEvent(self, event):
        p = QPainter(self)
        p.scale(self.panel_scale, self.panel_scale)
        p.fillRect(42, 8, 340, 130, QColor(0, 0, 0, 80))
        p.fillRect(42, 6, 340, 130, QColor(self.surface))
        p.setPen(QColor(self.edge))
        if self.settings.overlay_border == "on":
            p.drawRect(42, 6, 339, 129)
        phase = self.data.get("phase", "idle")
        text = self.data.get("preview", "")
        text = text[-self.settings.preview_chars :] if self.settings.live_preview == "on" else ""
        if phase == "idle":
            text = self.data.get("message", "Ready")
        elif phase != "recording":
            text = self.data.get("message", "Working…")
        elif not text:
            text = "Listening…"
        if self.notice and time.monotonic() < self.notice[1]:
            text = self.notice[0]
        # Lay out prose into three 16 px lines, keeping the newest words in view.
        font = QFont("sans-serif")
        font.setPixelSize(13)
        if self.text_cache is None or self.text_cache[0] != text:
            layout = QTextLayout(text, font)
            layout.beginLayout()
            lines = []
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(316)
                lines.append(text[line.textStart() : line.textStart() + line.textLength()])
            layout.endLayout()
            self.text_cache = (text, lines[-3:])
        p.setFont(font)
        p.setPen(QColor(self.dim if phase == "transcribing" else self.ink))
        for i, part in enumerate(self.text_cache[1]):
            p.drawText(54, 28 + i * 16, part)
        p.setPen(QColor(self.edge))
        p.drawLine(54, 87, 304, 87)
        p.setPen(Qt.PenStyle.NoPen)
        p.save()
        p.setClipRect(QRectF(54, 70, 250, 34))
        scale = self.devicePixelRatioF()
        for i, height in enumerate(self.trace.shown):
            x = round((54 + i * 4 - self.trace.fraction * 4) * scale) / scale
            top = round((87 - height / 2) * scale) / scale
            bottom = round((87 + height / 2) * scale) / scale
            p.fillRect(QRectF(x, top, 2, bottom - top), QColor(self.ink))
        p.restore()
        if phase == "transcribing":
            x = 54 + (int(time.monotonic() / (1.05 / 55)) % 55) * 4
            p.fillRect(x, 85, 24, 4, QColor("#ff4f1f"))
        p.setPen(QColor("#ff4f1f"))
        if phase == "recording":
            p.fillRect(316, 84, 6, 6, QColor("#ff4f1f"))
        else:
            p.drawRect(316, 84, 5, 5)
        mono = QFont("monospace")
        mono.setPixelSize(14)
        p.setFont(mono)
        p.setPen(QColor(self.ink))
        seconds = int(self.seconds)
        p.drawText(326, 92, f"{seconds // 60}:{seconds % 60:02d}")
        mono.setPixelSize(10)
        p.setFont(mono)
        p.setPen(QColor(self.dim))
        words, wpm = live_counters(self.data.get("preview", ""), self.seconds)
        p.drawText(54, 124, f"{words} W / {wpm} WPM" if active_phase(phase) else "MOUTHKEYS")
        p.drawText(302, 124, "WORDS")
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            self.move(event.globalPosition().toPoint() - self.drag)

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseDoubleClickEvent(self, event):
        self.place()


def active_phase(phase):
    return phase in ("recording", "stopping", "queued", "transcribing", "delivering", "loading")


def live_counters(text, seconds):
    words = len(text.split())
    return words, round(words * 60 / seconds) if seconds > 0 else 0


def launch():
    # A separate lifetime means closing Settings never removes shortcut feedback.
    with (runtime_dir() / "overlay.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        app = QApplication(sys.argv)
        app.setApplicationName("MouthKeys")
        app.setQuitOnLastWindowClosed(False)
        panel = Overlay()
        # Wake a resident panel immediately when a shortcut restarts the daemon.
        path = runtime_dir() / "overlay-wake.sock"
        path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as wake:
            wake.bind(str(path))
            os.chmod(path, 0o600)
            wake.setblocking(False)
            notifier = QSocketNotifier(wake.fileno(), QSocketNotifier.Type.Read)

            def awakened(*_):
                try:
                    while wake.recv(64):
                        pass
                except BlockingIOError:
                    pass
                panel.timer.setInterval(200)
                panel.poll()

            notifier.activated.connect(awakened)
            panel.poll()
            try:
                return app.exec()
            finally:
                notifier.setEnabled(False)
                path.unlink(missing_ok=True)
