"""Daemon-side client for the isolated clipboard owner; imports no Qt."""

import json
import select
import subprocess
from contextlib import suppress
from typing import Protocol, runtime_checkable

from .processes import python_command


@runtime_checkable
class _ReplyPipe(Protocol):
    def fileno(self) -> int: ...
    def readline(self, size: int, /) -> bytes: ...


def _reply(pipe: _ReplyPipe, timeout: float) -> bytes:
    ready, _, _ = select.select([pipe], [], [], timeout)
    # The owner writes each short, newline-terminated protocol token atomically.
    return pipe.readline(128) if ready else b""


class ClipboardPaste:
    def __init__(self, text: str, *, delay_ms: int = 500, exclude_history: bool = False) -> None:
        self.delay_ms = delay_ms
        self.process = subprocess.Popen(  # noqa: S603
            python_command("mouthkeys.clipboard_owner"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdin, stdout = self.process.stdin, self.process.stdout
        try:
            if stdin is None or stdout is None:
                raise RuntimeError("Clipboard helper did not open both protocol pipes")
            stdin.write(
                json.dumps(
                    {"text": text, "delay_ms": delay_ms, "exclude_history": exclude_history}
                ).encode()
                + b"\n"
            )
            stdin.flush()
            if _reply(stdout, 3) != b"ready\n":
                raise RuntimeError("Clipboard snapshot unavailable; using ordinary clipboard")
        except Exception:
            try:
                self.process.kill()
                self.process.wait(timeout=2)
            finally:
                for pipe in (stdin, stdout):
                    if pipe is not None:
                        pipe.close()
            raise
        self._stdin, self._stdout = stdin, stdout

    def finish(self, restore: bool) -> bool:  # noqa: FBT001 - delivery callback takes a bool
        try:
            self._stdin.write(b"restore\n" if restore else b"keep\n")
            self._stdin.flush()
            if restore:
                return _reply(self._stdout, self.delay_ms / 1000 + 2) in (
                    b"restored\n",
                    b"skipped\n",
                )
        except (OSError, ValueError):
            return False
        finally:
            with suppress(OSError):
                self._stdin.close()
            self._stdout.close()
        return True
