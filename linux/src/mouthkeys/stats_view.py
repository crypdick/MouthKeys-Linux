"""Local activity chart with accessible values; no Qt Charts dependency."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from math import floor

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QComboBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

from .qt_control import Response, Send


class ActivityChart(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setAccessibleName("Daily dictated words")
        self.values: Mapping[str, int] = {}
        self.days = 7
        self.series: list[tuple[str, int]] = []
        self.set_values({})

    def set_values(self, values: Mapping[str, int], days: int | None = None) -> None:
        self.values = values
        if days is not None:
            self.days = days
        today = datetime.now().astimezone().date()
        dates = [today - timedelta(days=self.days - 1 - index) for index in range(self.days)]
        self.series = [(str(day), values.get(str(day), 0)) for day in dates]
        self.setAccessibleDescription(
            "; ".join(f"{day}: {count} words" for day, count in self.series)
        )
        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        ink = self.palette().windowText().color()
        painter.setPen(ink)
        left, right, top, bottom = 40, self.width() - 10, 22, self.height() - 28
        painter.drawLine(left, bottom, right, bottom)
        peak = max(count for _, count in self.series)
        peak = max(1, peak)
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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        width = max(1, (self.width() - 50) / self.days)
        index = floor((event.position().x() - 40) / width)
        if 0 <= index < len(self.series):
            day, count = self.series[index]
            self.setToolTip(f"{day}: {count:,} words")
        else:
            self.setToolTip("")


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _number(value: object) -> int | float:
    return value if type(value) is int or type(value) is float else 0


def _milestone_lines(value: object) -> list[str]:
    lines = []
    for label, milestones in _mapping(value).items():
        if not isinstance(milestones, list):
            continue
        goals = []
        for item in milestones:
            goal = _mapping(item)
            target = goal.get("target")
            if type(target) is int:
                goals.append(f"{target:,}{' achieved' if goal.get('achieved') is True else ''}")
        if goals:
            lines.append(f"{label} milestones: " + " · ".join(goals))
    return lines


def _stats_lines(result: Mapping[str, object], daily_words: Mapping[str, int]) -> list[str]:
    words = _number(result.get("words"))
    dictations = _number(result.get("transcriptions"))
    streak = _number(result.get("current_streak"))
    minutes = _number(result.get("time_saved_seconds")) / 60
    lines = [
        f"{words:,} words  ·  {dictations:,} dictations",
        f"{streak} day streak  ·  {minutes:.1f} minutes saved",
        "",
    ]
    for name, value in _mapping(result.get("windows")).items():
        tally = _mapping(value)
        lines.append(
            f"{name}: {_number(tally.get('words')):,} words / "
            f"{_number(tally.get('transcriptions')):,} dictations"
        )
    lines.extend(
        [
            "",
            f"Longest dictation: {_number(result.get('longest_words')):,} words",
            f"Most words in one day: {_number(result.get('most_words_in_day')):,}",
            f"Average length: {_number(result.get('average_words'))} words",
            "",
            "Daily words:",
        ]
    )
    lines.extend(f"{day}: {words:,}" for day, words in daily_words.items())
    lines.extend(
        [
            "",
            f"Top apps: {result.get('top_apps', [])}",
            f"Peak hours: {result.get('peak_hour', [])}",
            f"Best streak: {_number(result.get('best_streak'))} days",
        ]
    )
    lines.extend(_milestone_lines(result.get("milestones")))
    return lines


class StatsPage(QWidget):
    def __init__(self, send: Send) -> None:
        super().__init__()
        self.send = send
        layout = QVBoxLayout(self)
        refresh = QPushButton("Refresh stats")
        refresh.clicked.connect(self.refresh_stats)
        layout.addWidget(refresh)
        self.chart_range = QComboBox()
        self.chart_range.addItem("Last 7 days", 7)
        self.chart_range.addItem("Last 30 days", 30)
        layout.addWidget(self.chart_range)
        self.activity_chart = ActivityChart()
        self.chart_range.currentIndexChanged.connect(self.update_chart_range)
        layout.addWidget(self.activity_chart)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlaceholderText(
            "Refresh to show local dictation activity. Typing speed and "
            "weekend streak rules are configurable in App Settings."
        )
        layout.addWidget(self.stats_text)

    def refresh_stats(self) -> None:
        self.send("stats", None, self.show_stats)

    def update_chart_range(self, _index: int) -> None:
        self.activity_chart.set_values(self.activity_chart.values, self.chart_range.currentData())

    def show_stats(self, response: Response) -> None:
        result = response.get("result")
        if not isinstance(result, dict):
            return
        daily_words = {
            day: count
            for day, count in _mapping(result.get("daily_words")).items()
            if type(count) is int and count >= 0
        }
        self.activity_chart.set_values(daily_words, self.chart_range.currentData())
        self.stats_text.setPlainText("\n".join(_stats_lines(result, daily_words)))
