from datetime import datetime

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent

from mouthkeys import stats_view
from mouthkeys.stats_view import ActivityChart


def test_hidden_chart_renders_empty_and_active_days_and_describes_each_bar(monkeypatch):
    class Clock:
        @staticmethod
        def now():
            return datetime.fromisoformat("2026-04-03T12:00:00")

    monkeypatch.setattr(stats_view, "datetime", Clock)
    chart = ActivityChart()
    chart.resize(260, 150)
    assert chart.accessibleName() == "Daily dictated words"
    assert chart.accessibleDescription().endswith("2026-04-03: 0 words")
    image = QImage(chart.size(), QImage.Format.Format_ARGB32)
    chart.render(image)

    chart.set_values({"2026-04-03": 1234}, days=3)
    expected = "2026-04-01: 0 words; 2026-04-02: 0 words; 2026-04-03: 1234 words"
    assert chart.accessibleDescription() == expected
    chart.render(image)
    assert image.pixelColor(210, 80) == chart.palette().windowText().color()
    assert not chart.isVisible()

    def hover(x):
        position = QPointF(x, 80)
        chart.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove,
                position,
                position,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    hover(0)
    assert not chart.toolTip()
    hover(210)
    assert chart.toolTip() == "2026-04-03: 1,234 words"
    hover(39)
    assert not chart.toolTip()
