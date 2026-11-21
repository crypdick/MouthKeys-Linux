"""Keep the passive dictation panel inside a screen's work area."""

from collections.abc import Callable

from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QWidget

from .config import Config


def _work_area(point: QPoint) -> QRect:
    screen = QApplication.screenAt(point) or QApplication.primaryScreen()
    return screen.availableGeometry()


class OverlayPlacement:
    def __init__(self, panel: QWidget, layout: Callable[[QSize], None]) -> None:
        self.panel = panel
        self.layout = layout
        self.drag_offset: QPoint | None = None
        self.dragged = False

    def _fits(self) -> bool:
        frame = self.panel.frameGeometry()
        return any(screen.availableGeometry().contains(frame) for screen in QApplication.screens())

    def _fit_size(self, area: QRect) -> None:
        frame_margin = self.panel.frameSize() - self.panel.size()
        self.layout((area.size() - frame_margin).expandedTo(QSize(1, 1)))

    def _move_within(self, area: QRect, point: QPoint) -> None:
        size = self.panel.frameSize()
        self.panel.move(
            max(area.left(), min(point.x(), area.left() + area.width() - size.width())),
            max(area.top(), min(point.y(), area.top() + area.height() - size.height())),
        )

    def place(self, config: Config) -> None:
        self.dragged = False
        area = _work_area(QCursor.pos())
        self._fit_size(area)
        size = self.panel.frameSize()
        x = area.left() + (area.width() - size.width()) // 2
        offset = config.overlay_bottom_offset
        y = (
            area.top() + offset
            if config.overlay_position == "top"
            else area.top() + area.height() - size.height() - offset
        )
        self._move_within(area, QPoint(x, y))

    def show(self, config: Config, *, new_recording: bool = False) -> None:
        if (new_recording or not self.panel.isVisible()) and (not self.dragged or not self._fits()):
            self.place(config)
        self.panel.show()  # Keep WA_ShowWithoutActivating; never raise or activate.

    def recover(self, config: Config) -> None:
        if not self.panel.isVisible() or self.drag_offset is not None:
            return
        # Native Wayland may not report global top-level positions.
        if not QApplication.platformName().startswith("wayland") and not self._fits():
            self.place(config)

    def begin_drag(self, point: QPoint) -> None:
        self.drag_offset = point - self.panel.pos()

    def drag_to(self, point: QPoint) -> None:
        if self.drag_offset is not None:
            area = _work_area(point)
            self._fit_size(area)
            self._move_within(area, point - self.drag_offset)
            self.dragged = True

    def end_drag(self) -> None:
        self.drag_offset = None
