"""Optional AT-SPI field identity, cursor context, and guarded editing.

GI bindings are optional system integration. Isolated helper bounds unresponsive
application calls, excludes password fields, and never writes surrounding text to disk.
Read-only snapshots support exact text-before/after paste verification; explicit
insert/focus requests are separately gated by the delivery controller.
"""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

from .processes import python_command


def _request(action: str, target: object = None, text: str | None = None) -> object:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    args = python_command("mouthkeys.atspi_helper")
    if importlib.util.find_spec("gi") is None and Path("/usr/bin/python3").is_file():
        args = [
            "/usr/bin/python3",
            "-I",
            str(Path(__file__).with_name("atspi_helper.py").resolve()),
        ]
    try:
        result = subprocess.run(  # noqa: S603
            args,
            input=json.dumps(
                {"action": action, "target": target, "text": text}, ensure_ascii=False
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        if len(result.stdout) > 1024 * 1024:
            return None
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def read_snapshot(
    target: object = None, *, include_protected: bool = False
) -> dict[str, object] | None:
    """Read a field snapshot, optionally reporting protected focus without field data."""
    value = _request("snapshot", target)
    if not isinstance(value, dict):
        return None
    if include_protected and value.get("protected") is True:
        return {"protected": True}
    if value.get("protected") is not False:
        return None
    text = value.get("text")
    if not isinstance(text, str) or len(text) > 100000:
        return None
    return value


def focus_target(target: object) -> bool:
    return _request("focus", target) is True


def insert_text(target: object, text: str) -> dict[str, object] | None:
    """Insert/replace only an unchanged focused captured editable field, without keys."""
    value = _request("insert", target, text)
    return value if isinstance(value, dict) else None


def read_context() -> str | None:
    value = read_snapshot()
    if not value or value.get("foreground") is not True:
        return None
    context = value.get("context")
    return context if isinstance(context, str) and len(context) <= 200 else None
