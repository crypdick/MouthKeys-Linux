"""Datasheet Mono recording panel; independent of the settings window."""

import subprocess
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QWidget

from .config import Config
from .desktop import source_names
from .lark_monitor import BatteryMonitor, microphone_labels
from .overlay_placement import OverlayPlacement
from .overlay_rendering import (
    DARK,
    LIGHT,
    PanelFrame,
    PanelNotice,
    PanelRenderer,
    PanelStatus,
    active_phase,
    button_icon,
    display_text,
)
from .processes import python_command
from .qt_control import Request, Response
from .trace import Trace

if TYPE_CHECKING:
    from .overlay_assistant import OverlayAssistant


class Overlay(QWidget):
    """340 × 130 pill and 30 px rails, matching the native round-six geometry."""

    def __init__(self) -> None:
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
        self.settings = Config.load()
        self.battery = BatteryMonitor()
        self.data: PanelStatus = {}
        self.trace = Trace()
        self.meter_frame: int | None = None
        self.renderer = PanelRenderer()
        self.jobs: set[Request] = set()
        self.pending = False
        self.active = False
        self.dismiss_at: float | None = None
        self.placement = OverlayPlacement(self, self._layout_panel)
        self.seconds: float = 0
        self.notice: tuple[str, float] | None = None
        self.notice_serial: int | None = None
        self.failures = 0
        self.assistant_window: OverlayAssistant | None = None
        self.buttons = {}
        for action, tip in (
            ("history", "Open history"),
            ("copy", "Copy transcript"),
            ("cancel", "Cancel / dismiss"),
        ):
            button = QPushButton(self)
            button.setIconSize(QSize(20, 20))
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _checked=False, action=action: self.action(action))
            self.buttons[action] = button
        self.mic_button = QPushButton(self)
        self.mic_button.setObjectName("microphone")
        self.mic_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mic_button.setToolTip("Choose microphone for this session, switching active recording")
        self.mic_button.setAccessibleName("Choose microphone")
        self.mic_button.clicked.connect(self.choose_microphone)
        self.assistant_button = QPushButton("Review AI / chats", self)
        self.assistant_button.setObjectName("assistant")
        self.assistant_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.assistant_button.setToolTip(
            "Open assistant answer, conversations and reviewed actions"
        )
        self.assistant_button.setAccessibleName("Open assistant review and conversations")
        self.assistant_button.clicked.connect(self.open_assistant)
        self.assistant_button.hide()
        self.placement.place(self.settings)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(200)
        self.animation = QTimer(self)
        self.animation.setTimerType(Qt.TimerType.PreciseTimer)
        self.animation.timeout.connect(self.animate)
        self.apply_theme()
        QApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)

    def apply_theme(self, *_: object) -> None:
        theme = self.settings.theme
        light = theme == "light" or (
            theme == "system" and QApplication.styleHints().colorScheme() == Qt.ColorScheme.Light
        )
        colors = self.renderer.colors = LIGHT if light else DARK
        self.setStyleSheet(
            f"QPushButton {{ background: {colors.chip}; color: {colors.ink}; border: none;"
            "font-size: 20px; border-radius: 0; }"
            f"QPushButton:hover {{ border: 1px solid {colors.ink}; }}"
            f"QPushButton:pressed {{ background: {colors.ink}; color: {colors.surface}; }}"
            f"QPushButton:disabled {{ color: {colors.dim}; }}"
            f"QPushButton#microphone, QPushButton#assistant {{ "
            f"font-family: monospace; font-size: 10px; "
            f"background: {colors.surface}; color: {colors.dim}; padding: 0; }}"
        )
        for action, button in self.buttons.items():
            button.setIcon(button_icon(action, colors.ink))
        if self.assistant_window is not None:
            self.assistant_window.apply_theme(colors)
        self.update()

    def _layout_panel(self, bounds: QSize) -> None:
        compact = self.settings.overlay_style == "compact"
        height = 106 if compact else 146
        self.panel_scale = min(
            self.settings.overlay_size / 100, bounds.width() / 424, bounds.height() / height
        )
        self.setFixedSize(round(424 * self.panel_scale), round(height * self.panel_scale))
        for action, x, y in (
            ("history", 6, 14 if compact else 29),
            ("copy", 6, 60 if compact else 83),
            ("cancel", 388, 14 if compact else 29),
        ):
            self.buttons[action].setGeometry(
                round(x * self.panel_scale),
                round(y * self.panel_scale),
                round(30 * self.panel_scale),
                round(30 * self.panel_scale),
            )
        self.mic_button.setGeometry(
            round(185 * self.panel_scale),
            round((74 if compact else 110) * self.panel_scale),
            round(108 * self.panel_scale),
            round(20 * self.panel_scale),
        )
        self.assistant_button.setGeometry(
            round(54 * self.panel_scale),
            round((74 if compact else 110) * self.panel_scale),
            round(125 * self.panel_scale),
            round(20 * self.panel_scale),
        )

    def apply_settings(self, config: Config) -> None:
        previous = self.settings
        self.settings = config
        if previous.theme != config.theme:
            self.apply_theme()
        if (
            previous.overlay_position,
            previous.overlay_bottom_offset,
            previous.overlay_size,
            previous.overlay_style,
        ) != (
            config.overlay_position,
            config.overlay_bottom_offset,
            config.overlay_size,
            config.overlay_style,
        ):
            self.placement.place(config)

    def choose_microphone(self) -> None:
        menu = QMenu(self)
        try:
            names = ["default", *source_names()]
        # The microphone menu remains usable when enumeration fails.
        except Exception:  # allow: exception-handling  # noqa: BLE001
            names = ["default"]
        config = Config.load()
        excluded = {s.strip() for s in config.microphone_never.split(",")}
        for name in names:
            if name not in excluded:
                action = menu.addAction(name)
                action.triggered.connect(
                    lambda _checked=False, name=name: self.send(
                        "select_microphone", data={"source": name, "persist": False}
                    )
                )
        menu.exec(self.mic_button.mapToGlobal(self.mic_button.rect().bottomLeft()))

    def send(
        self,
        action: str,
        callback: Callable[[dict[str, object]], None] | None = None,
        item_id: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        if data is None:
            selected = item_id or (self.data.get("id") if action == "copy" else None)
            data = {"id": selected} if selected else {}
        job = Request(action, data, parent=self)
        self.jobs.add(job)
        if callback:
            job.done.connect(callback)
        else:

            def feedback(response: dict[str, object]) -> None:
                error = response.get("error")
                if isinstance(error, str):
                    self.notice = (error, time.monotonic() + 10)
                    self.dismiss_at = self.notice[1]
                    self.placement.show(self.settings)
                    self.update()

            job.done.connect(feedback)
        job.finished.connect(lambda: self.jobs.discard(job))
        job.finished.connect(job.deleteLater)
        job.start()

    def action(self, action: str) -> None:
        if action == "history":
            self.send("history", self.show_history)
        elif action == "cancel":
            if self.active:
                self.send("cancel")
            self.hide()
            self.dismiss_at = 0
        else:
            self.send(action)

    def show_history(self, response: Response) -> None:
        if "error" in response:
            self.notice = (str(response["error"]), time.monotonic() + 10)
            self.update()
            return
        menu = QMenu(self)
        rows = cast("list[dict[str, str]]", response.get("result", []))[:12]
        if not rows:
            menu.addAction("No history yet").setEnabled(False)
        for row in rows:
            text = row.get("text") or row.get("error") or row.get("status", "Recording")
            submenu = menu.addMenu(" ".join(text.split())[:65])
            copy = submenu.addAction("Copy transcript")
            copy.setEnabled(bool(row.get("text")))
            copy.triggered.connect(
                lambda _checked=False, item_id=row["id"]: self.send("copy", item_id=item_id)
            )
        menu.addSeparator()
        menu.addAction("Open assistant / recent chats", self.open_assistant)
        menu.addAction(
            "Open full history / settings",
            lambda: subprocess.Popen(python_command("mouthkeys", "gui"), start_new_session=True),  # noqa: S603
        )
        self.history_menu = menu
        origin = self.mapToGlobal(QPoint(round(42 * self.panel_scale), 0))
        menu.popup(origin - QPoint(0, menu.sizeHint().height()))

    def open_assistant(self) -> None:
        # Only this explicit click may open an interactive/focusable surface.
        if self.assistant_window is None:
            from .overlay_assistant import OverlayAssistant  # noqa: PLC0415 - explicit open only

            self.assistant_window = OverlayAssistant(self.renderer.colors, self.poll)
        self.assistant_window.assistant_page.update_status(self.data)
        self.assistant_window.show()
        self.assistant_window.raise_()
        self.assistant_window.activateWindow()

    def poll(self) -> None:
        if self.pending:
            return
        self.pending = True

        def received(response: Response) -> None:
            self.pending = False
            if "result" in response:
                self.failures = 0
                status = cast("PanelStatus", response["result"])
                self.timer.setInterval(50 if status["phase"] == "recording" else 200)
                self.set_status(status)
            else:
                self.hide()  # Shutdown/disconnect must not leave a stale recording panel.
                self.active = False
                self.notice_serial = None
                self.animation.stop()
                self.failures += 1
                self.timer.setInterval(min(5000, 200 * 2 ** min(self.failures, 5)))

        self.send("status", received)

    def set_status(self, data: PanelStatus) -> None:
        config = Config.load()
        self.apply_settings(config)
        if self.assistant_window is not None:
            self.assistant_window.assistant_page.update_status(data)
        proposal = data.get("assistant_proposal")
        assistant = bool(proposal) or data.get("mode") in ("command", "edit")
        self.assistant_button.setVisible(assistant)
        if config.recording_panel == "off":
            self.data = data
            self.hide()
            self.active = False
            self.notice_serial = None
            return
        active = active_phase(data["phase"])
        new_recording = data.get("id") != self.data.get("id")
        self._update_visibility(config, data, active=active, new_recording=new_recording)
        self._update_trace(config, data, new_recording=new_recording)
        if active and not self.animation.isActive():
            self.animation.start(16)
        self.active = active
        self.data = data
        self._update_microphone(config, data.get("microphone", "default"), active=active)
        if data["phase"] == "recording":
            self.seconds = data.get("seconds", 0)
        self.buttons["copy"].setEnabled(not active and bool(data.get("preview")))
        self.update()

    def _update_visibility(
        self, config: Config, data: PanelStatus, *, active: bool, new_recording: bool
    ) -> None:
        if active and (not self.active or new_recording):
            self.dismiss_at = None
            self.notice = None
            self.placement.show(config, new_recording=new_recording)
        if not active and self.active:
            message = data.get("message", "")
            success = message.startswith(("Copied —", "Paste requested —", "Insertion requested —"))
            self.dismiss_at = time.monotonic() + (0.6 if success else 10)
            if message == "Cancelled":
                self.dismiss_at = 0
        proposal = data.get("assistant_proposal")
        if proposal and (new_recording or proposal != self.data.get("assistant_proposal")):
            # Retain the answer until explicit dismissal; never open or focus its review window.
            self.dismiss_at = None
            self.placement.show(config)
        self._update_notice(config, data.get("notice"))
        if self.dismiss_at is not None and time.monotonic() >= self.dismiss_at:
            self.hide()
        self.placement.recover(config)

    def _update_notice(self, config: Config, notice: PanelNotice | None) -> None:
        if not notice:
            self.notice = None
            self.notice_serial = None
        if notice and notice["serial"] != self.notice_serial:
            self.notice_serial = notice["serial"]
            if time.time() - notice["at"] < 10:
                message = notice["message"]
                if notice.get("kind") == "paste" and config.overlay_style == "compact":
                    message = "Auto-paste skipped. Text copied."
                self.notice = (message, time.monotonic() + 10)
                self.dismiss_at = self.notice[1]
                self.placement.show(config)

    def _update_trace(self, config: Config, data: PanelStatus, *, new_recording: bool) -> None:
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

    def _update_microphone(self, config: Config, microphone: str, *, active: bool) -> None:
        text, hint = microphone_labels(microphone, self.battery.poll(config, microphone))
        self.mic_button.setToolTip(hint)
        self.mic_button.setEnabled(not active)
        self.mic_button.setText(
            self.mic_button.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, self.mic_button.width() - 6
            )
        )

    def animate(self) -> None:
        if not self.isVisible():
            self.animation.stop()
            return
        now = time.monotonic()
        self.trace.advance(now)
        self.update(
            QRect(
                round(53 * self.panel_scale),
                round((31 if self.settings.overlay_style == "compact" else 68) * self.panel_scale),
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

    def display_text(self) -> str:
        return display_text(self.data, self.settings, self.notice)

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        frame = PanelFrame(
            phase=self.data.get("phase", "idle"),
            text=self.display_text(),
            preview=self.data.get("preview", ""),
            seconds=self.seconds,
            trace=self.trace,
            compact=self.settings.overlay_style == "compact",
            border=self.settings.overlay_border == "on",
            assistant_visible=self.assistant_button.isVisible(),
        )
        with QPainter(self) as painter:
            painter.scale(self.panel_scale, self.panel_scale)
            self.renderer.draw(painter, frame, self.devicePixelRatioF())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.placement.begin_drag(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.placement.drag_to(event.globalPosition().toPoint())

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.placement.end_drag()

    def mouseDoubleClickEvent(self, _event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self.placement.place(self.settings)
