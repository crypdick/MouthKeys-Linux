"""Search, preview, and manage locally retained dictation transcripts."""

from dataclasses import dataclass
from datetime import UTC, datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .qt_control import Response, Send

PAGE_SIZE = 100
MAX_RESULTS = 10_000


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class Transcript:
    item_id: str
    created: datetime
    status: str
    text: str
    error: str
    duration: float
    app: str
    window: str
    raw_text: str

    @classmethod
    def parse(cls, value: object) -> "Transcript | None":
        if not isinstance(value, dict):
            return None
        item_id, stamp = value.get("id"), value.get("created")
        if not isinstance(item_id, str) or not item_id or not isinstance(stamp, (int, float)):
            return None
        try:
            created = datetime.fromtimestamp(stamp, UTC).astimezone()
        except (OSError, OverflowError, ValueError):
            return None
        duration = value.get("duration")
        return cls(
            item_id,
            created,
            _text(value, "status"),
            _text(value, "text"),
            _text(value, "error"),
            duration if isinstance(duration, (int, float)) else 0,
            _text(value, "app"),
            _text(value, "window"),
            _text(value, "raw_text"),
        )

    def label(self) -> str:
        snippet = (self.text or self.error)[:65]
        return f"{self.created:%m-%d %H:%M}  /  {self.status.upper()}\n{snippet}"

    def detail(self) -> str:
        text = self.text + (f"\n\n{self.error}" if self.error else "")
        if self.duration:
            text += (
                f"\n\n{self.duration:.1f}s · {len(self.text.split())} words · "
                f"{len(self.text)} characters"
            )
        if self.app or self.window:
            text += f"\n{self.app} · {self.window}"
        if self.raw_text and self.raw_text != self.text:
            text += f"\n\nOriginal transcription:\n{self.raw_text}"
        return text


class HistoryPage(QWidget):
    def __init__(self, send: Send) -> None:
        super().__init__()
        self.send = send
        self.rows: list[Transcript] = []
        self.limit = PAGE_SIZE
        self.pending: tuple[str, int] | None = None
        self.serial = 0
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search transcripts, applications, and windows")
        self.search.textChanged.connect(lambda _: self.refresh_history())
        layout.addWidget(self.search)
        self.transcripts = QListWidget()
        self.transcripts.setWordWrap(True)
        self.transcripts.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transcripts.currentRowChanged.connect(self.show_transcript)
        split = QHBoxLayout()
        split.addWidget(self.transcripts, 1)
        layout.addLayout(split, 1)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        split.addWidget(self.preview, 2)
        self.preview.setPlaceholderText(
            "No History Yet\n\nYour transcriptions will appear here. Start dictating to begin."
        )
        row = QHBoxLayout()
        for label, handler in (
            ("Copy", self.copy_selected),
            ("Clear history", self.clear_history),
            ("Delete selected", self.delete_selected),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            row.addWidget(button)
        layout.addLayout(row)
        extras = QHBoxLayout()
        more = QPushButton("Load more")
        more.clicked.connect(self.load_more)
        extras.addWidget(more)
        layout.addLayout(extras)

    def refresh_history(self) -> None:
        query = (self.search.text(), self.limit)
        if self.pending == query:
            return
        self.pending = query
        self.serial += 1
        serial = self.serial

        def update(response: Response) -> None:
            if self.serial == serial:
                self.pending = None
                if query == (self.search.text(), self.limit):
                    self.apply_history(response)

        self.send("history", {"query": query[0], "limit": query[1]}, update)

    def load_more(self) -> None:
        self.limit = min(MAX_RESULTS, self.limit + PAGE_SIZE)
        self.refresh_history()

    def apply_history(self, response: Response) -> None:
        result = response.get("result")
        if not isinstance(result, list):
            return
        rows = [row for value in result if (row := Transcript.parse(value)) is not None]
        if rows == self.rows:
            return
        selected = self.selected_id()
        self.rows = rows
        self.transcripts.clear()
        index = 0
        for position, row in enumerate(rows):
            self.transcripts.addItem(row.label())
            if row.item_id == selected:
                index = position
        self.transcripts.setCurrentRow(index if rows else -1)

    def show_transcript(self, index: int) -> None:
        if 0 <= index < len(self.rows):
            self.preview.setPlainText(self.rows[index].detail())
        else:
            self.preview.clear()

    def selected_id(self) -> str | None:
        index = self.transcripts.currentRow()
        return self.rows[index].item_id if 0 <= index < len(self.rows) else None

    def copy_selected(self) -> None:
        if item_id := self.selected_id():
            self.send("copy", {"id": item_id}, None)

    def delete_selected(self) -> None:
        item_id = self.selected_id()
        if (
            item_id
            and QMessageBox.question(self, "Delete transcript", "Delete selected transcript?")
            == QMessageBox.StandardButton.Yes
        ):
            self.send("delete", {"id": item_id}, lambda _: self.refresh_history())

    def clear_history(self) -> None:
        if (
            QMessageBox.question(self, "Clear history", "Delete all transcripts?")
            == QMessageBox.StandardButton.Yes
        ):
            self.send("clear", None, None)
