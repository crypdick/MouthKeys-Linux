"""Run a GlobalShortcuts portal session using the system Python's optional GI."""

import json
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from functools import partial
from typing import Protocol, TypedDict, runtime_checkable

DESTINATION = "org.freedesktop.portal.Desktop"
DESKTOP = "/org/freedesktop/portal/desktop"
SHORTCUTS = "org.freedesktop.portal.GlobalShortcuts"
REQUEST = "org.freedesktop.portal.Request"
SESSION = "org.freedesktop.portal.Session"
MAX_MESSAGE_BYTES = 65536


@runtime_checkable
class Variant(Protocol):
    def unpack(self) -> object: ...


class ShortcutEntry(TypedDict):
    id: str
    action: str
    trigger: str


def _entry(value: object) -> ShortcutEntry:
    match value:
        case {"id": str(identity), "action": str(action), "trigger": str(trigger)} if (
            identity and action and trigger
        ):
            return {"id": identity, "action": action, "trigger": trigger}
    raise ValueError("Invalid portal shortcut entry")


def read_entries() -> list[ShortcutEntry]:
    payload = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Oversized portal shortcut configuration")
    data = json.loads(payload)
    if not isinstance(data, list):
        raise TypeError("Invalid portal shortcut configuration")
    entries = [_entry(value) for value in data]
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate portal shortcut ID")
    return entries


class PortalSession:
    def __init__(self, entries: list[ShortcutEntry]) -> None:
        import gi  # noqa: PLC0415 - the parent process does not require system GI

        gi.require_version("Gio", "2.0")
        from gi.repository import Gio, GLib  # noqa: PLC0415 - import after selecting the version

        self.entries = entries
        self.actions = {entry["id"]: entry["action"] for entry in entries}
        self.connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self.loop = GLib.MainLoop()
        self.variant = GLib.Variant
        self.timeout_add = GLib.timeout_add_seconds
        self.dbus_error = GLib.Error
        self.call_flags = Gio.DBusCallFlags.NONE
        self.signal_flags = Gio.DBusSignalFlags.NONE
        self.pending: dict[str, Callable[[object], None]] = {}
        self.granted: set[str] = set()
        self.session: str | None = None
        self.error: str | None = None

    def _emit(self, payload: dict[str, object]) -> None:
        print(json.dumps(payload), flush=True)  # noqa: T201 - child protocol

    def _subscribe(
        self, interface: str, signal: str, path: str | None, callback: Callable[[str, object], None]
    ) -> None:
        def receive(
            _connection: object,
            _sender: str,
            object_path: str,
            _iface: str,
            _signal: str,
            parameters: Variant,
            _user_data: object,
        ) -> None:
            callback(object_path, parameters.unpack())

        self.connection.signal_subscribe(
            DESTINATION, interface, signal, path, None, self.signal_flags, receive, None
        )

    def _request(
        self,
        method: str,
        signature: str,
        parameters: tuple[object, ...],
        callback: Callable[[object], None],
        *,
        options: dict[str, object] | None = None,
    ) -> None:
        token = f"mouthkeys_{uuid.uuid4().hex}"
        sender = self.connection.get_unique_name()[1:].replace(".", "_")
        expected = f"{DESKTOP}/request/{sender}/{token}"
        self.pending[expected] = callback
        handle = self.connection.call_sync(
            DESTINATION,
            DESKTOP,
            SHORTCUTS,
            method,
            self.variant(
                signature,
                (*parameters, {**(options or {}), "handle_token": self.variant("s", token)}),
            ),
            None,
            self.call_flags,
            2000,
            None,
        ).unpack()[0]
        if handle != expected:
            self.pending[handle] = self.pending.pop(expected)

    def _fail(self, message: str) -> None:
        self.error = message
        self.loop.quit()

    def _response(self, path: str, unpacked: object) -> None:
        callback = self.pending.pop(path, None)
        if callback is None:
            return
        try:
            if not isinstance(unpacked, tuple) or len(unpacked) != 2:
                raise ValueError("Invalid GlobalShortcuts portal response")
            code, results = unpacked
            if code != 0:
                self._fail("GlobalShortcuts setup cancelled or denied")
            else:
                callback(results)
        # Report callback failure over stdout, not in the GLib signal dispatch.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            self._fail(str(exc))

    def _created(self, results: object) -> None:
        session = results.get("session_handle") if isinstance(results, dict) else None
        if not isinstance(session, str) or not session.startswith("/"):
            raise ValueError("GlobalShortcuts portal returned an invalid session handle")
        self.session = session
        self._subscribe(SESSION, "Closed", session, self._closed)
        shortcuts = [
            (
                entry["id"],
                {
                    "description": self.variant("s", "MouthKeys " + entry["action"]),
                    "preferred_trigger": self.variant("s", entry["trigger"]),
                },
            )
            for entry in self.entries
        ]
        self._request("BindShortcuts", "(oa(sa{sv})sa{sv})", (session, shortcuts, ""), self._bound)

    def _bound(self, results: object) -> None:
        shortcuts = results.get("shortcuts") if isinstance(results, dict) else None
        if not isinstance(shortcuts, list):
            raise TypeError("GlobalShortcuts portal returned invalid bindings")
        self.granted = {identity for identity, _ in shortcuts if identity in self.actions}
        if not self.granted:
            self._fail("GlobalShortcuts portal did not bind any shortcuts")

    def _activation(self, signal: str, _path: str, unpacked: object) -> None:
        if not isinstance(unpacked, tuple) or len(unpacked) != 4:
            return
        session, identity, *_ = unpacked
        if session == self.session and isinstance(identity, str) and identity in self.granted:
            self._emit(
                {
                    "action": self.actions[identity],
                    "id": identity,
                    "down": signal == "Activated",
                    "at": time.monotonic(),
                }
            )

    def _closed(self, _path: str, _unpacked: object) -> None:
        self.session = None
        if self.granted:
            self.loop.quit()
        else:
            self._fail("GlobalShortcuts session closed before setup completed")

    def _setup_timeout(self) -> bool:
        if not self.granted:
            self._fail("GlobalShortcuts setup timed out")
        return False

    def _close(self, path: str, interface: str, timeout: int) -> None:
        # Closing is best-effort; the portal may already have detached this object.
        with suppress(self.dbus_error):
            self.connection.call_sync(
                DESTINATION, path, interface, "Close", None, None, self.call_flags, timeout, None
            )

    def run(self) -> None:
        self._subscribe(REQUEST, "Response", None, self._response)
        for signal in ("Activated", "Deactivated"):
            self._subscribe(
                SHORTCUTS,
                signal,
                DESKTOP,
                partial(self._activation, signal),
            )
        try:
            self._request(
                "CreateSession",
                "(a{sv})",
                (),
                self._created,
                options={
                    "session_handle_token": self.variant("s", f"mouthkeys_{uuid.uuid4().hex}")
                },
            )
            self.timeout_add(120, self._setup_timeout)
            self.loop.run()
            if self.error is not None:
                raise RuntimeError(self.error)
        finally:
            for path in self.pending:
                self._close(path, REQUEST, 300)
            if self.session is not None:
                self._close(self.session, SESSION, 500)


def main() -> int:
    try:
        PortalSession(read_entries()).run()
    # Never write a traceback into the event stream; report failure via its protocol.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        print(json.dumps({"error": f"GlobalShortcuts portal: {exc}"}), flush=True)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
