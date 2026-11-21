from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter

from mouthkeys.config import Config
from mouthkeys.overlay import Overlay
from mouthkeys.overlay_rendering import LIGHT, PanelFrame, PanelRenderer, button_icon, display_text
from mouthkeys.trace import Trace


@pytest.mark.parametrize(
    ("data", "overrides", "notice", "expected"),
    [
        ({}, {}, None, "Ready"),
        ({"phase": "recording"}, {}, None, "Listening…"),
        (
            {"phase": "recording", "preview": "Recognized"},
            {"live_preview": "off"},
            None,
            "Listening…",
        ),
        ({"phase": "recording", "preview": "old new"}, {"preview_chars": 3}, None, "new"),
        ({"phase": "loading", "preview": "stale"}, {}, None, "Working…"),
        (
            {
                "phase": "transcribing",
                "mode": "edit",
                "preview": "other",
                "assistant_stream": {"mode": "edit", "text": "Edited"},
            },
            {},
            None,
            "Edited",
        ),
        (
            {"phase": "transcribing", "mode": "edit", "assistant_stream": {"mode": "command"}},
            {},
            None,
            "Working…",
        ),
        ({"assistant_proposal": {"text": "New answer"}}, {"preview_chars": 6}, None, "answer"),
        ({"message": "Ready"}, {}, ("Saved", -1), "Ready"),
        ({"message": "Ready"}, {}, ("Saved", float("inf")), "Saved"),
    ],
)
def test_panel_text_respects_phase_preview_and_temporary_notices(data, overrides, notice, expected):
    # Short character limits here exercise rendering; persistence enforces the user-facing range.
    assert display_text(data, replace(Config(), **overrides), notice) == expected


def draw_frame(renderer, frame):
    image = QImage(424, 146, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    with QPainter(image) as painter:
        renderer.draw(painter, frame, 1)
    return image


def test_renderer_paints_waveform_and_uses_the_right_geometry_per_style():
    renderer = PanelRenderer()
    renderer.colors = LIGHT
    trace = Trace()
    trace.shown[0] = 22
    frame = PanelFrame(
        phase="recording",
        text="Several words " * 50,
        preview="Several words",
        seconds=60,
        trace=trace,
        compact=False,
        border=True,
        assistant_visible=False,
    )
    image = draw_frame(renderer, frame)
    assert image.pixelColor(317, 86) == QColor("#ff4f1f")
    assert image.pixelColor(54, 78) == QColor(LIGHT.ink)
    assert image == draw_frame(renderer, frame)

    compact = draw_frame(
        renderer,
        replace(frame, phase="transcribing", compact=True, border=False, assistant_visible=True),
    )
    assert compact.pixelColor(54, 42) == QColor(LIGHT.ink)
    assert compact.pixelColor(44, 110).alpha() == 0
    assert image != compact


def test_hidden_panel_widget_paints_and_icons_do_not_depend_on_font():
    panel = Overlay()
    panel.timer.stop()
    image = QImage(panel.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    panel.render(image)
    assert image.pixelColor(60, 70).alpha() == 255
    assert not panel.isVisible()
    assert not button_icon("retry", "#ffffff").isNull()
