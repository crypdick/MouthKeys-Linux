"""Send control-socket requests from Qt interfaces without blocking the event loop."""

from collections.abc import Mapping

from PySide6.QtCore import QObject, QThread, Signal

from .control import request


class Request(QThread):
    done = Signal(object)

    def __init__(
        self, action: str, data: Mapping[str, object] | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self.action, self.data = action, data or {}

    def run(self) -> None:
        try:
            self.done.emit({"result": request(self.action, **self.data)})
        # Report to the requesting UI without duplicating private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self.done.emit({"error": str(exc)})
