"""Isolated AT-SPI D-Bus operations, using protocol object identity, not widget names.

Protocol: GNOME/at-spi2-core xml/{Accessible,Text,EditableText,Component}.xml.
No imports from MouthKeys: distro Python can supply GI independently of uv Python.
"""

import json
import re
import sys
import time
from collections.abc import Container, Mapping, Sequence
from contextlib import suppress
from typing import NamedTuple, Protocol, TypedDict, cast, runtime_checkable

ACCESSIBLE = "org.a11y.atspi.Accessible"
TEXT = "org.a11y.atspi.Text"
EDITABLE = "org.a11y.atspi.EditableText"
COMPONENT = "org.a11y.atspi.Component"
ROOT = ("org.a11y.atspi.Registry", "/org/a11y/atspi/accessible/root")
LIMIT = 100000


ObjectRef = tuple[str, str]


def _reference(value: object) -> ObjectRef | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        name, path = value
        if (
            isinstance(name, str)
            and re.fullmatch(r":\d+\.\d+", name)
            and isinstance(path, str)
            and path.startswith("/org/a11y/atspi/accessible/")
        ):
            return name, path
    return None


@runtime_checkable
class _Transport(Protocol):
    password_role: int
    terminal_role: int
    focused_state: int
    editable_state: int
    active_state: int
    deadline: float

    @property
    def window_roles(self) -> Container[int]: ...

    def call(
        self,
        ref: Sequence[str],
        interface: str,
        method: str,
        signature: str | None = None,
        args: tuple[object, ...] = (),
    ) -> tuple[object, ...]: ...

    def prop(self, ref: Sequence[str], interface: str, name: str) -> object: ...


class FieldSnapshot(TypedDict):
    field_id: str
    bus: str
    path: str
    text: str
    caret: int
    selection_start: int
    selection_end: int
    context: str
    text_start: int
    character_count: int
    complete: bool
    focused: bool
    foreground: bool | None
    editable: bool
    terminal: bool
    protected: bool
    app: str


class _Protected(TypedDict):
    protected: bool


class _Position(NamedTuple):
    characters: int
    caret: int
    start: int
    end: int
    selections: int


class Bus:
    def __init__(self) -> None:
        import gi  # noqa: PLC0415 - distribution GI bindings are optional

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi, Gio, GLib  # noqa: PLC0415 - distribution GI bindings are optional

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

    def call(
        self,
        ref: Sequence[str],
        interface: str,
        method: str,
        signature: str | None = None,
        args: tuple[object, ...] = (),
    ) -> tuple[object, ...]:
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
        values: tuple[object, ...] = result.unpack()
        return values

    def prop(self, ref: Sequence[str], interface: str, name: str) -> object:
        return self.call(ref, "org.freedesktop.DBus.Properties", "Get", "(ss)", (interface, name))[
            0
        ]


def _states(bus: _Transport, ref: ObjectRef) -> Sequence[int]:
    words = bus.call(ref, ACCESSIBLE, "GetState")[0]
    if not isinstance(words, (list, tuple)) or any(type(word) is not int for word in words):
        raise ValueError("Invalid accessibility state")
    return words


def has_state(words: Sequence[int], state: int) -> bool:
    return len(words) > state // 32 and bool(words[state // 32] & (1 << (state % 32)))


def active_window(bus: _Transport, ref: ObjectRef) -> bool | None:
    """Nearest exposed top-level window corroborates local child focus."""
    visited: set[ObjectRef] = set()
    current: ObjectRef | None = ref
    for _ in range(24):
        if current is None or current in visited:
            break
        visited.add(current)
        role = bus.call(current, ACCESSIBLE, "GetRole")[0]
        if type(role) is int and role in bus.window_roles:
            return has_state(_states(bus, current), bus.active_state)
        current = _reference(bus.prop(current, ACCESSIBLE, "Parent"))
    return None


def _foreground(bus: _Transport, ref: ObjectRef) -> bool | None:
    # A stale toolkit object cannot corroborate focus. Do not log bus payloads.
    with suppress(Exception):
        return active_window(bus, ref)
    return None


def _app_name(bus: _Transport, ref: ObjectRef) -> str:
    # An unavailable application name does not invalidate otherwise stable text.
    with suppress(Exception):
        name = bus.prop((ref[0], ROOT[1]), ACCESSIBLE, "Name")
        if isinstance(name, str):
            return name[:200]
    return ""


def _position(bus: _Transport, ref: ObjectRef) -> _Position | None:
    count = bus.prop(ref, TEXT, "CharacterCount")
    caret = bus.prop(ref, TEXT, "CaretOffset")
    selections = bus.call(ref, TEXT, "GetNSelections")[0]
    if type(count) is not int or type(caret) is not int or not 0 <= caret <= count:
        return None
    if type(selections) is not int or selections not in (0, 1):
        return None  # Disjoint edits cannot be represented by a single captured range.
    if not selections:
        return _Position(count, caret, caret, caret, selections)
    start, end = bus.call(ref, TEXT, "GetSelection", "(i)", (0,))
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= count:
        return None
    return _Position(count, caret, start, end, selections)


def _region(position: _Position, window: Mapping[str, object] | None) -> tuple[int, int] | None:
    count, start, end = position.characters, position.start, position.end
    first, last = (0, count) if count <= LIMIT else (max(0, start - 2000), min(count, end + 2000))
    if window is not None and window.get("complete") is False:
        offset, previous = window.get("text_start"), window.get("character_count")
        text = window.get("text")
        if type(offset) is not int or type(previous) is not int or not isinstance(text, str):
            return None
        first, last = offset, min(count, offset + len(text) + count - previous)
    if not 0 <= first <= last <= count or last - first > LIMIT:
        return None
    return first, last


def _read_text(bus: _Transport, ref: ObjectRef, region: tuple[int, int], role: int) -> str | None:
    # The toolkit can turn an editable field into a password field during capture.
    if bus.call(ref, ACCESSIBLE, "GetRole")[0] != role:
        return None
    text = bus.call(ref, TEXT, "GetText", "(ii)", region)[0]
    if not isinstance(text, str) or len(text) != region[1] - region[0]:
        return None
    return text


def _same_field_state(
    bus: _Transport, ref: ObjectRef, *, focused: bool, editable: bool, foreground: bool | None
) -> bool:
    states = _states(bus, ref)
    return (
        has_state(states, bus.focused_state) == focused
        and has_state(states, bus.editable_state) == editable
        and _foreground(bus, ref) == foreground
    )


def snapshot(
    bus: _Transport,
    ref: object,
    *,
    require_focus: bool = True,
    window: Mapping[str, object] | None = None,
) -> FieldSnapshot | None:
    address = _reference(ref)
    if address is None:
        return None
    role = bus.call(address, ACCESSIBLE, "GetRole")[0]
    if type(role) is not int or role == bus.password_role:
        return None
    states = _states(bus, address)
    focused = has_state(states, bus.focused_state)
    editable = has_state(states, bus.editable_state)
    terminal = role == bus.terminal_role
    if (require_focus and not focused) or not (editable or terminal):
        return None
    foreground = _foreground(bus, address)
    position = _position(bus, address)
    region = _region(position, window) if position is not None else None
    if position is None or region is None:
        return None
    text = _read_text(bus, address, region, role)
    app = _app_name(bus, address) if text is not None else ""
    if (
        text is None
        or _position(bus, address) != position
        or _read_text(bus, address, region, role) != text
        or not _same_field_state(
            bus, address, focused=focused, editable=editable, foreground=foreground
        )
    ):
        return None
    first, last = region
    return {
        "field_id": address[0] + "|" + address[1],
        "bus": address[0],
        "path": address[1],
        "text": text,
        "caret": position.caret,
        "selection_start": position.start,
        "selection_end": position.end,
        "context": text[max(0, position.start - first - 200) : max(0, position.start - first)],
        "text_start": first,
        "character_count": position.characters,
        "complete": first == 0 and last == position.characters,
        "focused": focused,
        "foreground": foreground,
        "editable": editable,
        "terminal": terminal,
        "protected": False,
        "app": app,
    }


def find_focused(bus: _Transport) -> FieldSnapshot | _Protected | None:
    pending = [ROOT]
    visited: set[ObjectRef] = set()
    while pending and len(visited) < 500 and time.monotonic() < bus.deadline:
        ref = pending.pop()
        if ref in visited:
            continue
        visited.add(ref)
        try:
            role = bus.call(ref, ACCESSIBLE, "GetRole")[0]
            if role == bus.password_role:
                if has_state(_states(bus, ref), bus.focused_state) and active_window(bus, ref):
                    # Prevent window-level fallback without reading any password data.
                    return {"protected": True}
                continue
            value = snapshot(bus, ref)
            if value and value["foreground"] is True:
                return value
            children = bus.call(ref, ACCESSIBLE, "GetChildren")[0]
            if isinstance(children, (list, tuple)):
                pending.extend(
                    child for item in reversed(children) if (child := _reference(item)) is not None
                )
        # Ignore stale toolkit objects and never log their payloads.
        except Exception:  # allow: exception-handling  # noqa: BLE001, S112
            continue
    return None


def unchanged(before: Mapping[str, object] | None, after: Mapping[str, object] | None) -> bool:
    return bool(
        before
        and after
        and before.get("text_start", 0) == after.get("text_start", 0)
        and all(
            before.get(field) == after.get(field)
            for field in (
                "field_id",
                "text",
                "character_count",
                "caret",
                "selection_start",
                "selection_end",
            )
        )
        and after.get("focused")
        and after.get("foreground") is True
        and not after.get("protected")
    )


def expected_text(before: Mapping[str, object], text: str) -> str:
    offset = cast("int", before.get("text_start", 0))
    start = cast("int", before["selection_start"]) - offset
    end = cast("int", before["selection_end"]) - offset
    original = cast("str", before["text"])
    return original[:start] + text + original[end:]


def _focus(bus: _Transport, ref: ObjectRef, before: Mapping[str, object]) -> bool:
    # Explicit return-to-start only. Require the target to still exist and be unchanged.
    current = snapshot(bus, ref, require_focus=False)
    if current is None or any(
        current[key] != before.get(key) for key in ("text", "selection_start", "selection_end")
    ):
        return False
    if not bus.call(ref, COMPONENT, "GrabFocus")[0]:
        return False
    return unchanged(before, snapshot(bus, ref))


def _insert(
    bus: _Transport, ref: ObjectRef, before: Mapping[str, object], text: object
) -> FieldSnapshot | None:
    if not isinstance(text, str) or len(text) > LIMIT:
        return None
    current = snapshot(bus, ref)
    if (
        current is None
        or not unchanged(before, current)
        or current["terminal"]
        or not current["editable"]
        or current["selection_start"] != current["selection_end"]
    ):
        return None
    expected = expected_text(current, text)
    if len(expected) > LIMIT:
        return None
    # InsertText preserves rich content outside the insertion point.
    # Never use whole-field SetTextContents to replace a selection.
    if not bus.call(
        ref, EDITABLE, "InsertText", "(isi)", (current["caret"], text, len(text.encode("utf-8")))
    )[0]:
        return None
    # Never retry an accepted insert if caret placement is unsupported.
    with suppress(Exception):
        bus.call(ref, TEXT, "SetCaretOffset", "(i)", (current["caret"] + len(text),))
    result = snapshot(bus, ref, window=before)
    return result if result and result["text"] == expected else None


def operate(
    bus: _Transport, request: Mapping[str, object]
) -> FieldSnapshot | _Protected | bool | None:
    action = request.get("action", "snapshot")
    before = request.get("target")
    if action == "snapshot" and not before:
        return find_focused(bus)
    if (
        not isinstance(before, dict)
        or (ref := _reference((before.get("bus"), before.get("path")))) is None
    ):
        return None
    if action == "snapshot":
        return snapshot(bus, ref, require_focus=False, window=before)
    if action == "focus":
        return _focus(bus, ref, before)
    if action == "insert":
        return _insert(bus, ref, before, request.get("text"))
    raise ValueError("Unknown accessibility action")


def main() -> None:
    try:
        line = sys.stdin.buffer.readline(1024 * 1024 + 1)
        if len(line) > 1024 * 1024:
            raise ValueError("Accessibility request too large")
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("Invalid accessibility request")  # noqa: TRY004 - malformed JSON payload
        value = operate(Bus(), request)
    # Do not log private toolkit errors or request payloads.
    except Exception:  # allow: exception-handling  # noqa: BLE001
        value = None
    print(json.dumps(value, ensure_ascii=False))  # noqa: T201


if __name__ == "__main__":
    main()
