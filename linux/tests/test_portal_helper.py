import io
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from mouthkeys import portal_helper as helper

ENTRIES = [
    {"id": "toggle_0", "action": "toggle", "trigger": "CTRL+space"},
    {"id": "cancel_1", "action": "cancel", "trigger": "ALT+Escape"},
]
OWN = helper.DESKTOP + "/session/1_9/current"
OTHER = helper.DESKTOP + "/session/1_8/other"


class Variant:
    def __init__(self, signature, value):
        self.signature = signature
        self.value = value

    def unpack(self):
        return self.value


class FakePortal:
    class DisconnectedError(Exception):
        pass

    def __init__(self):
        self.loop = Mock()
        self.subscriptions = {}
        self.calls = {}
        self.closed = []
        self.requests = {}
        self.timeout = None
        self.legacy_handle = False
        self.detach_on_close = False

    def signal_subscribe(self, destination, interface, signal, path, arg0, flags, receiver, data):
        assert (destination, arg0, flags, data) == (helper.DESTINATION, None, 0, None)
        self.subscriptions[(interface, signal)] = (path, receiver)

    def get_unique_name(self):
        return ":1.9"

    def call_sync(
        self, destination, path, interface, method, parameters, result, flags, delay, cancel
    ):
        assert (destination, result, flags, cancel) == (helper.DESTINATION, None, 0, None)
        if method == "Close":
            self.closed.append((path, interface, delay))
            if self.detach_on_close:
                raise self.DisconnectedError
            return None
        assert (path, interface, delay) == (helper.DESKTOP, helper.SHORTCUTS, 2000)
        self.calls[method] = parameters
        token = parameters.value[-1]["handle_token"].value
        location = "legacy" if self.legacy_handle else "1_9"
        request = f"{helper.DESKTOP}/request/{location}/{token}"
        self.requests[method] = request
        return Variant("(o)", (request,))

    def emit(self, interface, signal, path, data):
        scope, callback = self.subscriptions[(interface, signal)]
        if scope is None or scope == path:
            callback(self, ":1.1", path, interface, signal, Variant("", data), None)

    def reply(self, method, values, code=0):
        self.emit(helper.REQUEST, "Response", self.requests[method], (code, values))

    def timeout_add(self, seconds, action):
        assert seconds == 120
        self.timeout = action


@pytest.fixture
def portal(monkeypatch):
    bus = FakePortal()
    require = Mock()
    gi = ModuleType("gi")
    gi.require_version = require
    repo = ModuleType("gi.repository")
    repo.Gio = SimpleNamespace(
        BusType=SimpleNamespace(SESSION="session"),
        DBusCallFlags=SimpleNamespace(NONE=0),
        DBusSignalFlags=SimpleNamespace(NONE=0),
        bus_get_sync=Mock(return_value=bus),
    )
    repo.GLib = SimpleNamespace(
        MainLoop=lambda: bus.loop,
        Variant=Variant,
        Error=bus.DisconnectedError,
        timeout_add_seconds=bus.timeout_add,
    )
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repo)
    stdin(monkeypatch, json.dumps(ENTRIES).encode() + b"\n")
    return bus, require, repo.Gio.bus_get_sync


def stdin(monkeypatch, payload):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload)))


@pytest.mark.parametrize("legacy", [False, True])
def test_helper_uses_granted_bindings_and_ignores_other_sessions(
    portal, monkeypatch, capsys, legacy
):
    bus, require, connect = portal
    bus.legacy_handle = legacy
    monkeypatch.setattr(helper, "time", SimpleNamespace(monotonic=Mock(return_value=5.0)))

    def run():
        create = bus.calls["CreateSession"]
        assert create.signature == "(a{sv})"
        assert create.value[0]["session_handle_token"].signature == "s"
        bus.emit(helper.REQUEST, "Response", "/unrelated", ())
        bus.reply("CreateSession", {"session_handle": OWN})
        bind = bus.calls["BindShortcuts"]
        assert bind.signature == "(oa(sa{sv})sa{sv})"
        session, shortcuts, parent, options = bind.value
        assert (session, parent) == (OWN, "")
        assert options["handle_token"].signature == "s"
        assert [(key, fields["preferred_trigger"].value) for key, fields in shortcuts] == [
            ("toggle_0", "CTRL+space"),
            ("cancel_1", "ALT+Escape"),
        ]
        bus.emit(helper.SHORTCUTS, "Activated", helper.DESKTOP, (OWN, "toggle_0", 0, {}))
        bus.emit(helper.SESSION, "Closed", OTHER, ({},))
        bus.loop.quit.assert_not_called()
        bus.reply("BindShortcuts", {"shortcuts": [("toggle_0", {}), ("not_requested", {})]})
        assert not bus.timeout()
        bus.emit(helper.SHORTCUTS, "Activated", helper.DESKTOP, (OTHER, "toggle_0", 0, {}))
        bus.emit(helper.SHORTCUTS, "Activated", helper.DESKTOP, (OWN, 4, 0, {}))
        bus.emit(helper.SHORTCUTS, "Activated", helper.DESKTOP, (OWN, "cancel_1", 0, {}))
        bus.emit(helper.SHORTCUTS, "Activated", helper.DESKTOP, ())
        for signal in ("Activated", "Deactivated"):
            bus.emit(helper.SHORTCUTS, signal, helper.DESKTOP, (OWN, "toggle_0", 0, {}))
        bus.emit(helper.SESSION, "Closed", OWN, ({},))
        bus.loop.quit.assert_called_once_with()

    bus.loop.run.side_effect = run
    assert helper.main() == 0
    require.assert_called_once_with("Gio", "2.0")
    connect.assert_called_once_with("session", None)
    assert not bus.closed
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages == [
        {"action": "toggle", "id": "toggle_0", "down": down, "at": 5.0} for down in (True, False)
    ]


@pytest.mark.parametrize(
    ("event", "expected", "closes_request", "closes_session"),
    [
        ("create-denied", "cancelled or denied", False, False),
        ("create-timeout", "timed out", True, False),
        ("malformed-response", "Invalid GlobalShortcuts portal response", False, False),
        ("invalid-session", "invalid session handle", False, False),
        ("bind-denied", "cancelled or denied", False, True),
        ("bind-timeout", "timed out", True, True),
        ("invalid-bindings", "invalid bindings", False, True),
        ("empty-grant", "did not bind any shortcuts", False, True),
        ("closed-before-bind", "closed before setup completed", True, False),
    ],
)
def test_helper_reports_failures_and_closes_only_owned_objects(
    portal, capsys, event, expected, closes_request, closes_session
):
    bus, _, _ = portal
    bus.detach_on_close = True

    def run():
        if event == "create-denied":
            bus.reply("CreateSession", {}, code=1)
        elif event == "create-timeout":
            assert not bus.timeout()
        elif event == "malformed-response":
            bus.emit(helper.REQUEST, "Response", bus.requests["CreateSession"], ())
        elif event == "invalid-session":
            bus.reply("CreateSession", {"session_handle": "invalid"})
        else:
            bus.reply("CreateSession", {"session_handle": OWN})
            if event == "bind-denied":
                bus.reply("BindShortcuts", {}, code=2)
            elif event == "bind-timeout":
                assert not bus.timeout()
            elif event == "invalid-bindings":
                bus.reply("BindShortcuts", {"shortcuts": "invalid"})
            elif event == "empty-grant":
                bus.reply("BindShortcuts", {"shortcuts": []})
            else:
                bus.emit(helper.SESSION, "Closed", OWN, ({},))

    bus.loop.run.side_effect = run
    assert helper.main() == 1
    assert expected in json.loads(capsys.readouterr().out)["error"]
    closed = []
    if closes_request:
        request = list(bus.requests.values())[-1]
        closed.append((request, helper.REQUEST, 300))
    if closes_session:
        closed.append((OWN, helper.SESSION, 500))
    assert bus.closed == closed


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b"x" * (helper.MAX_MESSAGE_BYTES + 1), "Oversized portal shortcut configuration"),
        (b"{}", "Invalid portal shortcut configuration"),
        (b"[42]", "Invalid portal shortcut entry"),
        (json.dumps([ENTRIES[0], ENTRIES[0]]).encode(), "Duplicate portal shortcut ID"),
    ],
)
def test_invalid_helper_input_never_opens_session(portal, monkeypatch, capsys, payload, error):
    _, _, connect = portal
    stdin(monkeypatch, payload)
    assert helper.main() == 1
    assert error in json.loads(capsys.readouterr().out)["error"]
    connect.assert_not_called()


def test_packaged_helper_can_run_directly_under_isolated_python():
    result = subprocess.run(  # noqa: S603 - fixed isolated package entry point
        [sys.executable, "-I", str(Path(helper.__file__).resolve())],
        input=b"{}\n",
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "error": "GlobalShortcuts portal: Invalid portal shortcut configuration"
    }


def test_helper_entrypoint_propagates_failure_exit_status(monkeypatch, capsys):
    stdin(monkeypatch, b"{}\n")
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(Path(helper.__file__).resolve()), run_name="__main__")
    assert exit_info.value.code == 1
    assert "Invalid portal shortcut configuration" in json.loads(capsys.readouterr().out)["error"]
