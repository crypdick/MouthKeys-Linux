"""Short-lived Qt clipboard owner for opt-in, multi-format paste restoration.

Separate GUI process keeps Qt out of the daemon and owns restored selections until
another application copies. No clipboard data is written to disk or diagnostic logs.
"""

import json
import select
import subprocess
import sys

from .processes import python_command

MAX_BYTES = 32 * 1024 * 1024


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
    def __init__(self, text, delay_ms=500):
        self.process = subprocess.Popen(
            python_command("mouthkeys.clipboard"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            self.process.stdin.write(
                json.dumps({"text": text, "delay_ms": delay_ms}).encode() + b"\n"
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
        except (OSError, ValueError):
            pass
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
    clipboard.setText(data["text"])
    app.processEvents()
    if not clipboard.ownsClipboard():
        return 1
    print("ready", flush=True)
    finished = False
    restored = False

    def restore():
        nonlocal restored
        restored = True
        if not restore_formats(clipboard, saved):
            app.quit()

    def receive(*_):
        nonlocal finished
        action = sys.stdin.buffer.readline(16)
        notifier.setEnabled(False)
        finished = True
        if action == b"restore\n":
            QTimer.singleShot(delay_ms, restore)
        elif not clipboard.ownsClipboard():
            app.quit()

    def changed():
        if (finished or restored) and not clipboard.ownsClipboard():
            app.quit()

    notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Type.Read)
    notifier.activated.connect(receive)
    clipboard.changed.connect(changed)
    # Parent crash must not leave an unused helper; keep current clipboard until copied over.
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
