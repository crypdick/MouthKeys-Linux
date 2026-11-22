"""Qt subprocess that keeps restored clipboard selections alive without a window."""

import json
import sys
from collections.abc import Mapping
from contextlib import suppress
from typing import Protocol, runtime_checkable

from PySide6 import QtCore, QtGui

MAX_BYTES = 32 * 1024 * 1024


@runtime_checkable
class _Clipboard(Protocol):
    def ownsClipboard(self) -> bool: ...  # noqa: N802 - Qt interface
    def clear(self) -> None: ...
    def setMimeData(self, mime: QtCore.QMimeData) -> None: ...  # noqa: N802 - Qt interface


def temporary_mime(text: str, *, exclude_history: bool) -> QtCore.QMimeData:
    mime = QtCore.QMimeData()
    mime.setText(text)
    if exclude_history:
        # Advisory standard Linux secret hint; participating managers honor it.
        mime.setData("x-kde-passwordManagerHint", b"secret")
    return mime


def capture_formats(mime: QtCore.QMimeData | None) -> dict[str, bytes]:
    saved = {}
    total = 0
    if mime is not None:
        for name in mime.formats():
            value = bytes(mime.data(name).data())
            total += len(value)
            if total > MAX_BYTES:
                raise ValueError("Clipboard snapshot exceeds 32 MiB")
            saved[name] = value
    return saved


def restore_formats(clipboard: _Clipboard, saved: Mapping[str, bytes]) -> bool:
    if not clipboard.ownsClipboard():
        return False
    if not saved:
        clipboard.clear()
        return False
    mime = QtCore.QMimeData()
    for name, value in saved.items():
        mime.setData(name, QtCore.QByteArray(value))
    clipboard.setMimeData(mime)
    return True


def _read_request() -> tuple[str, int, bool] | None:
    line = sys.stdin.buffer.readline(MAX_BYTES + 1)
    if len(line) > MAX_BYTES:
        return None
    try:
        data = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    text = data.get("text")
    delay_ms = data.get("delay_ms", 500)
    exclude_history = data.get("exclude_history", False)
    if (
        not isinstance(text, str)
        or type(delay_ms) is not int
        or not 100 <= delay_ms <= 5000
        or not isinstance(exclude_history, bool)
    ):
        return None
    return text, delay_ms, exclude_history


def main() -> int:
    request = _read_request()
    if request is None:
        return 1
    text, delay_ms, exclude_history = request
    app = QtGui.QGuiApplication(["MouthKeys clipboard"])
    clipboard = app.clipboard()
    try:
        saved = capture_formats(clipboard.mimeData())
    except ValueError:
        return 1  # Never silently replace an incomplete multi-format snapshot.
    clipboard.setMimeData(temporary_mime(text, exclude_history=exclude_history))
    app.processEvents()
    if not clipboard.ownsClipboard():
        return 1
    print("ready", flush=True)  # noqa: T201
    finished = False
    restored = False
    restore_requested = False
    acknowledged = False

    def acknowledge(result: str) -> None:
        nonlocal acknowledged
        if restore_requested and not acknowledged:
            acknowledged = True
            with suppress(BrokenPipeError):
                print(result, flush=True)  # noqa: T201

    def restore() -> None:
        nonlocal restored
        restored = True
        success = restore_formats(clipboard, saved)
        acknowledge("restored" if success else "skipped")
        if not success:
            app.quit()

    def receive(*_: object) -> None:
        nonlocal finished, restore_requested
        action = sys.stdin.buffer.readline(16)
        notifier.setEnabled(False)
        finished = True
        if action == b"restore\n":
            restore_requested = True
            if clipboard.ownsClipboard():
                QtCore.QTimer.singleShot(delay_ms, restore)
            else:
                acknowledge("skipped")
                app.quit()
        elif not clipboard.ownsClipboard():
            app.quit()

    def changed() -> None:
        if (finished or restored) and not clipboard.ownsClipboard():
            acknowledge("skipped")
            app.quit()

    notifier = QtCore.QSocketNotifier(sys.stdin.fileno(), QtCore.QSocketNotifier.Type.Read)
    notifier.activated.connect(receive)
    clipboard.changed.connect(changed)
    # Parent crash must not leave an unused helper; keep current clipboard until copied over.
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
