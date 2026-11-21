"""Supervise capture with an stdin lifeline, without thread-unsafe preexec_fn hooks.

The daemon keeps the write end open. Normal stop, crashes, and SIGKILL all close it,
so FFmpeg receives SIGINT and finalizes its WAV. This process opens no microphone
itself; its executable argument also permits hardware-free lifecycle tests.
"""

import select
import signal
import subprocess
import sys
from collections.abc import Sequence


def supervise(args: Sequence[str]) -> int:
    process = subprocess.Popen(args, stdin=subprocess.DEVNULL)  # noqa: S603
    try:
        while process.poll() is None:
            readable, _, _ = select.select([sys.stdin.buffer], [], [], 0.1)
            if readable and not sys.stdin.buffer.read(1):
                break
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            returncode = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=1)
    return returncode if returncode >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(supervise(sys.argv[1:]))
