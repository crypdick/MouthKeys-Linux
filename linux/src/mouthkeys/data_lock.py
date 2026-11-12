"""Serialize dictionary/vocabulary transactions across GUI, CLI and service."""

import fcntl
import os
import threading
import time
from contextlib import contextmanager

from .config import config_path, private_dir

_LOCK = threading.RLock()
_LOCAL = threading.local()
LOCK_TIMEOUT = 3.0


@contextmanager
def dictionary_lock():
    # Reentrant in one thread; acquiring another flock descriptor would deadlock.
    if not _LOCK.acquire(timeout=LOCK_TIMEOUT):
        raise TimeoutError("Dictionary is busy; retry after the current operation")
    try:
        if getattr(_LOCAL, "held", False):
            yield
            return
        path = private_dir(config_path().parent) / ".dictionary-transfer.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            deadline = time.monotonic() + LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Dictionary is busy in another process; retry shortly"
                        ) from None
                    time.sleep(0.02)
            _LOCAL.held = True
            try:
                yield
            finally:
                _LOCAL.held = False
        finally:
            os.close(descriptor)
    finally:
        _LOCK.release()
