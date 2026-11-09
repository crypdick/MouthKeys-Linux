"""Optional, read-only AT-SPI cursor context. Never used as proof of paste delivery.

GI bindings are optional system integration. Isolated helper bounds unresponsive
application calls and never reads password fields or writes surrounding text to disk.
"""

import importlib.util
import json
import subprocess
import time
from pathlib import Path


def focused_context(atspi):
    atspi.set_timeout(50, 50)
    deadline = time.monotonic() + 0.25
    stack = [(atspi.get_desktop(0), 0)]
    seen = 0
    while stack and seen < 400 and time.monotonic() < deadline:
        node, depth = stack.pop()
        seen += 1
        try:
            state = node.get_state_set()
            if node.get_role() == atspi.Role.PASSWORD_TEXT:
                continue
            if state.contains(atspi.StateType.FOCUSED):
                if state.contains(atspi.StateType.EDITABLE):
                    text = node.get_text_iface()
                    if text:
                        caret = text.get_caret_offset()
                        if caret >= 0:
                            return text.get_text(max(0, caret - 200), caret)
            if depth < 20:
                count = min(node.get_child_count(), 400 - seen)
                for index in reversed(range(count)):
                    child = node.get_child_at_index(index)
                    if child:
                        stack.append((child, depth + 1))
        except Exception:
            continue  # Apps may disappear or deny accessibility between any two calls.
    return None


def read_context():
    from .processes import python_command

    args = python_command("mouthkeys.accessibility")
    # Distro GI bindings match the distro Python, not uv's managed interpreter.
    # This helper has no package imports when run directly. -I excludes user code paths.
    if importlib.util.find_spec("gi") is None and Path("/usr/bin/python3").is_file():
        args = ["/usr/bin/python3", "-I", str(Path(__file__).resolve())]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        value = json.loads(result.stdout)
        return value if isinstance(value, str) and len(value) <= 200 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main():
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        value = focused_context(Atspi)
    except (ImportError, ValueError, RuntimeError):
        value = None
    print(json.dumps(value))


if __name__ == "__main__":
    main()
