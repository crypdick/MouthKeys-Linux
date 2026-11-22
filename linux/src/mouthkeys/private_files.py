"""Atomically replace a file with owner-only permissions."""

import os
import tempfile
from pathlib import Path


def write_private(path: str | Path, data: bytes) -> str:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        # Remove uncommitted data even when interrupted, then propagate the failure.
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return str(path)
