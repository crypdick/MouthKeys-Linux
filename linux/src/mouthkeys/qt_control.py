"""Run Qt tasks and control requests without blocking the event loop."""

from collections.abc import Callable, Mapping
from threading import Event

from PySide6.QtCore import QObject, QThread, Signal

from .control import request

Response = dict[str, object]
ResponseHandler = Callable[[Response], None]
Send = Callable[[str, Mapping[str, object] | None, ResponseHandler | None], None]


class Task(QThread):
    done = Signal(object)

    def __init__(
        self,
        function: Callable[[], object],
        parent: QObject | None,
        cancel_event: Event | None = None,
    ) -> None:
        super().__init__(parent)
        self.function = function
        self.cancel_event = cancel_event or Event()

    def run(self) -> None:
        try:
            self.done.emit({"result": self.function()})
        # Report to the requesting UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.done.emit({"error": str(exc)})


class Request(Task):
    def __init__(
        self, action: str, data: Mapping[str, object] | None = None, parent: QObject | None = None
    ) -> None:
        fields = data or {}
        super().__init__(lambda: request(action, **fields), parent)
