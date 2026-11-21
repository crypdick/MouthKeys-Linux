"""Paint the passive panel in logical coordinates; independent of window lifetime."""

import time
from dataclasses import dataclass
from typing import TypedDict

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QTextLayout

from .config import Config
from .trace import Trace


class AssistantText(TypedDict, total=False):
    text: str
    mode: str


class PanelStatus(TypedDict, total=False):
    id: str | None
    phase: str
    mode: str
    preview: str
    message: str
    assistant_stream: AssistantText | None
    assistant_proposal: AssistantText | None


@dataclass(frozen=True)
class PanelColors:
    surface: str
    ink: str
    edge: str
    dim: str
    chip: str


LIGHT = PanelColors("#ffffff", "#111214", "#111214", "#55565a", "#f2f2f4")
DARK = PanelColors("#111214", "#ebebec", "#2c2e33", "#949597", "#1a1b1f")


@dataclass(frozen=True)
class PanelFrame:
    phase: str
    text: str
    preview: str
    seconds: float
    trace: Trace
    compact: bool
    border: bool
    assistant_visible: bool


def button_icon(action: str, color: str) -> QIcon:
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


def display_text(data: PanelStatus, settings: Config, notice: tuple[str, float] | None) -> str:
    phase = data.get("phase", "idle")
    stream = data.get("assistant_stream")
    text = (
        stream.get("text", "")
        if stream and stream.get("mode") == "edit"
        else data.get("preview", "")
    )
    text = text[-settings.preview_chars :] if settings.live_preview == "on" else ""
    if phase == "idle":
        proposal = data.get("assistant_proposal")
        text = (
            proposal["text"][-settings.preview_chars :]
            if proposal
            else data.get("message", "Ready")
        )
    elif phase != "recording":
        text = text if data.get("mode") == "edit" and text else data.get("message", "Working…")
    elif not text:
        text = "Listening…"
    if notice and time.monotonic() < notice[1]:
        text = notice[0]
    return text


def active_phase(phase: str) -> bool:
    return phase in (
        "starting",
        "recording",
        "switching",
        "capture_failed",
        "stopping",
        "queued",
        "transcribing",
        "delivering",
        "loading",
    )


def live_counters(text: str, seconds: float) -> tuple[int, int]:
    words = len(text.split())
    return words, round(words * 60 / seconds) if seconds > 0 else 0


class PanelRenderer:
    def __init__(self) -> None:
        self.colors = DARK
        self.text_cache: tuple[str, bool, list[str]] | None = None

    def draw(self, p: QPainter, frame: PanelFrame, scale: float) -> None:
        compact = frame.compact
        panel_height = 90 if compact else 130
        baseline = 50 if compact else 87
        p.fillRect(42, 8, 340, panel_height, QColor(0, 0, 0, 80))
        p.fillRect(42, 6, 340, panel_height, QColor(self.colors.surface))
        p.setPen(QColor(self.colors.edge))
        if frame.border:
            p.drawRect(42, 6, 339, panel_height - 1)
        phase = frame.phase
        self._draw_preview(p, frame)
        p.setPen(QColor(self.colors.edge))
        p.drawLine(54, baseline, 304, baseline)
        p.setPen(Qt.PenStyle.NoPen)
        p.save()
        p.setClipRect(QRectF(54, baseline - 17, 250, 34))
        for i, height in enumerate(frame.trace.shown):
            x = round((54 + i * 4 - frame.trace.fraction * 4) * scale) / scale
            top = round((baseline - height / 2) * scale) / scale
            bottom = round((baseline + height / 2) * scale) / scale
            p.fillRect(QRectF(x, top, 2, bottom - top), QColor(self.colors.ink))
        p.restore()
        if phase == "transcribing":
            x = 54 + (int(time.monotonic() / (1.05 / 55)) % 55) * 4
            p.fillRect(x, baseline - 2, 24, 4, QColor("#ff4f1f"))
        p.setPen(QColor("#ff4f1f"))
        if phase == "recording":
            p.fillRect(316, baseline - 3, 6, 6, QColor("#ff4f1f"))
        else:
            p.drawRect(316, baseline - 3, 5, 5)
        mono = QFont("monospace")
        mono.setPixelSize(14)
        p.setFont(mono)
        p.setPen(QColor(self.colors.ink))
        seconds = int(frame.seconds)
        p.drawText(326, baseline + 5, f"{seconds // 60}:{seconds % 60:02d}")
        mono.setPixelSize(10)
        p.setFont(mono)
        p.setPen(QColor(self.colors.dim))
        words, wpm = live_counters(frame.preview, frame.seconds)
        if not frame.assistant_visible:
            p.drawText(
                54,
                88 if compact else 124,
                f"{words} W / {wpm} WPM" if active_phase(phase) else "MOUTHKEYS",
            )
        p.drawText(302, 88 if compact else 124, "WORDS")

    def _draw_preview(self, p: QPainter, frame: PanelFrame) -> None:
        text = frame.text
        # Lay out prose into three 16 px lines, keeping the newest words in view.
        font = QFont("sans-serif")
        font.setPixelSize(13)
        if self.text_cache is None or self.text_cache[:2] != (text, frame.compact):
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
            self.text_cache = (text, frame.compact, lines[-1:] if frame.compact else lines[-3:])
        p.setFont(font)
        p.setPen(QColor(self.colors.dim if frame.phase == "transcribing" else self.colors.ink))
        for i, part in enumerate(self.text_cache[2]):
            p.drawText(54, 28 + i * 16, part)
