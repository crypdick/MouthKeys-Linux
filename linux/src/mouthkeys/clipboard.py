"""Isolated Qt clipboard owner for opt-in, multi-format paste restoration.

Separate GUI process keeps Qt out of the daemon and owns restored selections until
another application copies. No clipboard data is written to disk or diagnostic logs.
"""

import json
import select
import subprocess
import sys

from .processes import python_command

MAX_BYTES = 32 * 1024 * 1024


def temporary_mime(text, exclude_history):
    from PySide6.QtCore import QMimeData

    mime = QMimeData()
    mime.setText(text)
    if exclude_history:
        # Advisory standard Linux secret hint; participating managers honor it.
        mime.setData("x-kde-passwordManagerHint", b"secret")
    return mime


def capture_formats(mime):
    saved = {}
    total = 0
    if mime is not None:
        for name in mime.formats():
            value = bytes(mime.data(name))
            total += len(value)
            if total > MAX_BYTES:
                raise ValueError("Clipboard snapshot exceeds 32 MiB")
            saved[name] = value
    return saved


def restore_formats(clipboard, saved):
    from PySide6.QtCore import QByteArray, QMimeData

    if not clipboard.ownsClipboard():
        return False
    if not saved:
        clipboard.clear()
        return False
    mime = QMimeData()
    for name, value in saved.items():
        mime.setData(name, QByteArray(value))
    clipboard.setMimeData(mime)
    return True


class ClipboardPaste:
    def __init__(self, text, delay_ms=500, exclude_history=False):
        self.delay_ms = delay_ms
        self.process = subprocess.Popen(
            python_command("mouthkeys.clipboard"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.process.stdin.write(
                json.dumps(
                    {"text": text, "delay_ms": delay_ms, "exclude_history": bool(exclude_history)}
                ).encode()
                + b"\n"
            )
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], 3)
            if not ready or self.process.stdout.readline(128) != b"ready\n":
                raise RuntimeError("Clipboard snapshot unavailable; using ordinary clipboard")
        except Exception:
            self.process.kill()
            self.process.wait(timeout=2)
            raise

    def finish(self, restore):
        try:
            self.process.stdin.write(b"restore\n" if restore else b"keep\n")
            self.process.stdin.flush()
            if restore:
                ready, _, _ = select.select([self.process.stdout], [], [], self.delay_ms / 1000 + 2)
                return bool(
                    ready and self.process.stdout.readline(128) in (b"restored\n", b"skipped\n")
                )
            return True
        except (OSError, ValueError):
            return False
        finally:
            self.process.stdin.close()
            self.process.stdout.close()


def main():
    from PySide6.QtCore import QSocketNotifier, QTimer
    from PySide6.QtGui import QGuiApplication

    line = sys.stdin.buffer.readline(MAX_BYTES + 1)
    if len(line) > MAX_BYTES:
        return 1
    data = json.loads(line)
    delay_ms = data.get("delay_ms", 500)
    if type(delay_ms) is not int or not 100 <= delay_ms <= 5000:
        return 1
    app = QGuiApplication(["MouthKeys clipboard"])
    clipboard = app.clipboard()
    try:
        saved = capture_formats(clipboard.mimeData())
    except ValueError:
        return 1  # Never silently replace an incomplete multi-format snapshot.
    clipboard.setMimeData(temporary_mime(data["text"], data.get("exclude_history", False)))
    app.processEvents()
    if not clipboard.ownsClipboard():
        return 1
    print("ready", flush=True)
    finished = False
    restored = False
    restore_requested = False
    acknowledged = False

    def acknowledge(result):
        nonlocal acknowledged
        if restore_requested and not acknowledged:
            acknowledged = True
            try:
                print(result, flush=True)
            except BrokenPipeError:
                pass

    def restore():
        nonlocal restored
        restored = True
        success = restore_formats(clipboard, saved)
        acknowledge("restored" if success else "skipped")
        if not success:
            app.quit()

    def receive(*_):
        nonlocal finished, restore_requested
        action = sys.stdin.buffer.readline(16)
        notifier.setEnabled(False)
        finished = True
        if action == b"restore\n":
            restore_requested = True
            if clipboard.ownsClipboard():
                QTimer.singleShot(delay_ms, restore)
            else:
                acknowledge("skipped")
                app.quit()
        elif not clipboard.ownsClipboard():
            app.quit()

    def changed():
        if (finished or restored) and not clipboard.ownsClipboard():
            acknowledge("skipped")
            app.quit()

    notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Type.Read)
    notifier.activated.connect(receive)
    clipboard.changed.connect(changed)
    # Parent crash must not leave an unused helper; keep current clipboard until copied over.
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
