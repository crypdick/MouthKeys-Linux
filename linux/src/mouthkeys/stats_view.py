"""Local activity chart with accessible values; no Qt Charts dependency."""

from datetime import date, timedelta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


class ActivityChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setAccessibleName("Daily dictated words")
        self.values = {}
        self.days = 7
        self.series = []
        self.set_values({})

    def set_values(self, values, days=None):
        self.values = values
        if days is not None:
            self.days = days
        today = date.today()
        dates = [today - timedelta(days=self.days - 1 - index) for index in range(self.days)]
        self.series = [(str(day), values.get(str(day), 0)) for day in dates]
        self.setAccessibleDescription(
            "; ".join(f"{day}: {count} words" for day, count in self.series)
        )
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        ink = self.palette().windowText().color()
        painter.setPen(ink)
        left, right, top, bottom = 40, self.width() - 10, 22, self.height() - 28
        painter.drawLine(left, bottom, right, bottom)
        peak = max(1, max(count for _, count in self.series))
        painter.drawText(0, top + 4, str(peak))
        width = max(1, (right - left) / self.days)
        for index, (_, count) in enumerate(self.series):
            height = (bottom - top) * count / peak
            painter.fillRect(
                int(left + index * width + 2),
                int(bottom - height),
                max(1, int(width - 4)),
                int(height),
                ink,
            )
        painter.drawText(left, self.height() - 5, self.series[0][0][5:])
        painter.drawText(right - 40, self.height() - 5, self.series[-1][0][5:])
        if not any(count for _, count in self.series):
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No activity yet")
        painter.end()

    def mouseMoveEvent(self, event):
        width = max(1, (self.width() - 50) / self.days)
        index = int((event.position().x() - 40) / width)
        if 0 <= index < len(self.series):
            day, count = self.series[index]
            self.setToolTip(f"{day}: {count:,} words")
