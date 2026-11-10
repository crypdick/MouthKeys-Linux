"""Isolated AT-SPI D-Bus operations, using protocol object identity, not widget names.

Protocol: GNOME/at-spi2-core xml/{Accessible,Text,EditableText,Component}.xml.
No imports from MouthKeys: distro Python can supply GI independently of uv Python.
"""

import json
import re
import sys
import time

ACCESSIBLE = "org.a11y.atspi.Accessible"
TEXT = "org.a11y.atspi.Text"
EDITABLE = "org.a11y.atspi.EditableText"
COMPONENT = "org.a11y.atspi.Component"
ROOT = ("org.a11y.atspi.Registry", "/org/a11y/atspi/accessible/root")
LIMIT = 100000


def valid_ref(ref):
    return (
        isinstance(ref, (list, tuple))
        and len(ref) == 2
        and isinstance(ref[0], str)
        and re.fullmatch(r":\d+\.\d+", ref[0])
        and isinstance(ref[1], str)
        and ref[1].startswith("/org/a11y/atspi/accessible/")
    )


class Bus:
    def __init__(self):
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi, Gio, GLib

        self.Gio, self.GLib = Gio, GLib
        self.password_role = int(Atspi.Role.PASSWORD_TEXT)
        self.terminal_role = int(Atspi.Role.TERMINAL)
        self.focused_state = int(Atspi.StateType.FOCUSED)
        self.editable_state = int(Atspi.StateType.EDITABLE)
        self.active_state = int(Atspi.StateType.ACTIVE)
        self.window_roles = {int(Atspi.Role.FRAME), int(Atspi.Role.WINDOW), int(Atspi.Role.DIALOG)}
        session = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        result = session.call_sync(
            "org.a11y.Bus",
            "/org/a11y/bus",
            "org.a11y.Bus",
            "GetAddress",
            None,
            None,
            Gio.DBusCallFlags.NONE,
            300,
            None,
        )
        self.connection = Gio.DBusConnection.new_for_address_sync(
            result.unpack()[0],
            Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT
            | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
            None,
            None,
        )
        self.deadline = time.monotonic() + 1.25

    def call(self, ref, interface, method, signature=None, args=()):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Accessibility query timed out")
        parameters = self.GLib.Variant(signature, args) if signature else None
        result = self.connection.call_sync(
            ref[0],
            ref[1],
            interface,
            method,
            parameters,
            None,
            self.Gio.DBusCallFlags.NONE,
            80,
            None,
        )
        return result.unpack()

    def prop(self, ref, interface, name):
        return self.call(ref, "org.freedesktop.DBus.Properties", "Get", "(ss)", (interface, name))[
            0
        ]


def has_state(words, state):
    return len(words) > state // 32 and bool(words[state // 32] & (1 << (state % 32)))


def active_window(bus, ref):
    """Nearest exposed top-level window corroborates local child focus."""
    visited = set()
    current = ref
    for _ in range(24):
        current = tuple(current)
        if current in visited or not valid_ref(current):
            return None
        visited.add(current)
        role = bus.call(current, ACCESSIBLE, "GetRole")[0]
        if role in bus.window_roles:
            return has_state(bus.call(current, ACCESSIBLE, "GetState")[0], bus.active_state)
        current = bus.prop(current, ACCESSIBLE, "Parent")
    return None


def snapshot(bus, ref, *, require_focus=True, window=None):
    if not valid_ref(ref):
        return None
    role = bus.call(ref, ACCESSIBLE, "GetRole")[0]
    if role == bus.password_role:
        return None
    states = bus.call(ref, ACCESSIBLE, "GetState")[0]
    focused = has_state(states, bus.focused_state)
    editable = has_state(states, bus.editable_state)
    terminal = role == bus.terminal_role
    if (require_focus and not focused) or not (editable or terminal):
        return None
    try:
        foreground = active_window(bus, ref)
    except Exception:
        foreground = None
    count = bus.prop(ref, TEXT, "CharacterCount")
    caret = bus.prop(ref, TEXT, "CaretOffset")
    if type(count) is not int or count < 0 or not 0 <= caret <= count:
        return None  # Never truncate text and then pretend an exact comparison is possible.
    selections = bus.call(ref, TEXT, "GetNSelections")[0]
    if selections > 1:
        return None  # Disjoint edits cannot be represented by a single captured range.
    start, end = (
        bus.call(ref, TEXT, "GetSelection", "(i)", (0,)) if selections == 1 else (caret, caret)
    )
    if not 0 <= start <= end <= count:
        return None
    text_start, text_end = 0, count
    if count > LIMIT:
        text_start, text_end = max(0, start - 2000), min(count, end + 2000)
    if window and window.get("complete") is False:
        text_start = window["text_start"]
        text_end = min(count, text_start + len(window["text"]) + count - window["character_count"])
    if not 0 <= text_start <= text_end <= count or text_end - text_start > LIMIT:
        return None
    text = bus.call(ref, TEXT, "GetText", "(ii)", (text_start, text_end))[0]
    if not isinstance(text, str) or len(text) != text_end - text_start:
        return None
    app_name = ""
    try:
        app_name = bus.prop((ref[0], ROOT[1]), ACCESSIBLE, "Name")
    except Exception:
        pass
    # A snapshot spans several D-Bus calls. Reject edits/focus changes during it,
    # rather than returning an early caret/focus bit alongside later text.
    if (
        bus.prop(ref, TEXT, "CharacterCount") != count
        or bus.prop(ref, TEXT, "CaretOffset") != caret
        or bus.call(ref, TEXT, "GetNSelections")[0] != selections
        or (selections == 1 and bus.call(ref, TEXT, "GetSelection", "(i)", (0,)) != (start, end))
        or bus.call(ref, TEXT, "GetText", "(ii)", (text_start, text_end))[0] != text
        or bus.call(ref, ACCESSIBLE, "GetRole")[0] != role
    ):
        return None
    final_states = bus.call(ref, ACCESSIBLE, "GetState")[0]
    final_focus = has_state(final_states, bus.focused_state)
    if final_focus != focused or has_state(final_states, bus.editable_state) != editable:
        return None
    try:
        if active_window(bus, ref) != foreground:
            return None
    except Exception:
        if foreground is not None:
            return None
    return {
        "field_id": ref[0] + "|" + ref[1],
        "bus": ref[0],
        "path": ref[1],
        "text": text,
        "caret": caret,
        "selection_start": start,
        "selection_end": end,
        "context": text[max(0, start - text_start - 200) : max(0, start - text_start)],
        "text_start": text_start,
        "character_count": count,
        "complete": text_start == 0 and text_end == count,
        "focused": focused,
        "foreground": foreground,
        "editable": editable,
        "terminal": terminal,
        "protected": False,
        "app": app_name[:200] if isinstance(app_name, str) else "",
    }


def find_focused(bus):
    pending = [ROOT]
    visited = set()
    while pending and len(visited) < 500 and time.monotonic() < bus.deadline:
        ref = tuple(pending.pop())
        if ref in visited:
            continue
        visited.add(ref)
        try:
            role = bus.call(ref, ACCESSIBLE, "GetRole")[0]
            if role == bus.password_role:
                continue
            value = snapshot(bus, ref)
            if value and value.get("foreground") is True:
                return value
            children = bus.call(ref, ACCESSIBLE, "GetChildren")[0]
            pending.extend(reversed([child for child in children if valid_ref(child)]))
        except Exception:
            continue
    return None


def unchanged(before, after):
    return bool(
        before
        and after
        and before.get("field_id") == after.get("field_id")
        and before.get("text") == after.get("text")
        and before.get("text_start", 0) == after.get("text_start", 0)
        and before.get("character_count") == after.get("character_count")
        and before.get("selection_start") == after.get("selection_start")
        and before.get("selection_end") == after.get("selection_end")
        and after.get("focused")
        and after.get("foreground") is True
        and not after.get("protected")
    )


def expected_text(before, text):
    start = before["selection_start"] - before.get("text_start", 0)
    end = before["selection_end"] - before.get("text_start", 0)
    return before["text"][:start] + text + before["text"][end:]


def operate(bus, request):
    action = request.get("action", "snapshot")
    before = request.get("target")
    if action == "snapshot" and not before:
        return find_focused(bus)
    ref = (before.get("bus"), before.get("path")) if isinstance(before, dict) else None
    if not valid_ref(ref):
        return None
    if action == "snapshot":
        return snapshot(bus, ref, require_focus=False, window=before)
    if action == "focus":
        # Explicit return-to-start only. Require target still exists and unchanged first.
        current = snapshot(bus, ref, require_focus=False)
        if (
            not current
            or current["text"] != before.get("text")
            or current["selection_start"] != before.get("selection_start")
            or current["selection_end"] != before.get("selection_end")
        ):
            return False
        if not bus.call(ref, COMPONENT, "GrabFocus")[0]:
            return False
        return unchanged(before, snapshot(bus, ref))
    if action == "insert":
        text = request.get("text")
        if not isinstance(text, str) or len(text) > LIMIT:
            return None
        current = snapshot(bus, ref)
        if (
            not unchanged(before, current)
            or current["terminal"]
            or not current["editable"]
            or current["selection_start"] != current["selection_end"]
        ):
            return None
        expected = expected_text(before, text)
        if len(expected) > LIMIT:
            return None
        # InsertText preserves rich content outside the insertion point. Never use
        # whole-field SetTextContents to replace a selection: it erases formatting.
        if not bus.call(
            ref, EDITABLE, "InsertText", "(isi)", (before["caret"], text, len(text.encode("utf-8")))
        )[0]:
            return None
        try:
            bus.call(ref, TEXT, "SetCaretOffset", "(i)", (before["selection_start"] + len(text),))
        except Exception:
            pass  # An accepted insertion must not be retried if caret placement is unsupported.
        result = snapshot(bus, ref, window=before)
        return result if result and result["text"] == expected else None
    raise ValueError("Unknown accessibility action")


def main():
    try:
        line = sys.stdin.buffer.readline(1024 * 1024 + 1)
        if len(line) > 1024 * 1024:
            raise ValueError("Accessibility request too large")
        value = operate(Bus(), json.loads(line))
    except Exception:
        value = None
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
