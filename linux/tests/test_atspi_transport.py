import io
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from mouthkeys import atspi_helper as atspi


def test_bus_connects_via_accessibility_address_and_bounds_object_calls(monkeypatch):
    calls = []

    def reply(*values):
        return SimpleNamespace(unpack=lambda: values)

    def session_call(name, path, interface, method, params, result, flags, timeout, cancellable):
        assert (name, path, interface, method) == (
            "org.a11y.Bus",
            "/org/a11y/bus",
            "org.a11y.Bus",
            "GetAddress",
        )
        assert (params, result, flags, timeout, cancellable) == (None, None, 0, 300, None)
        return reply("test:private-bus")

    def object_call(name, path, interface, method, params, result, flags, timeout, cancellable):
        assert name == ":1.1"
        assert path == "/org/a11y/atspi/accessible/7"
        assert (result, flags, timeout, cancellable) == (None, 0, 80, None)
        calls.append((interface, method, params))
        return reply(123)

    def connection(address, flags, observer, cancellable):
        assert (address, flags, observer, cancellable) == ("test:private-bus", 3, None, None)
        return SimpleNamespace(call_sync=object_call)

    def require_version(module, version):
        assert (module, version) == ("Atspi", "2.0")

    def get_bus(kind, cancellable):
        assert (kind, cancellable) == ("session", None)
        return SimpleNamespace(call_sync=session_call)

    gi = ModuleType("gi")
    gi.require_version = require_version
    repository = ModuleType("gi.repository")
    repository.Atspi = SimpleNamespace(
        Role=SimpleNamespace(PASSWORD_TEXT=39, TERMINAL=60, FRAME=23, WINDOW=69, DIALOG=16),
        StateType=SimpleNamespace(FOCUSED=12, EDITABLE=7, ACTIVE=1),
    )
    repository.Gio = SimpleNamespace(
        BusType=SimpleNamespace(SESSION="session"),
        DBusCallFlags=SimpleNamespace(NONE=0),
        DBusConnectionFlags=SimpleNamespace(AUTHENTICATION_CLIENT=1, MESSAGE_BUS_CONNECTION=2),
        DBusConnection=SimpleNamespace(new_for_address_sync=connection),
        bus_get_sync=get_bus,
    )
    repository.GLib = SimpleNamespace(Variant=lambda signature, args: (signature, args))
    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)

    bus = atspi.Bus()
    target = (":1.1", "/org/a11y/atspi/accessible/7")
    assert bus.password_role == 39
    assert bus.window_roles == {16, 23, 69}
    assert bus.call(target, atspi.ACCESSIBLE, "GetRole") == (123,)
    assert bus.prop(target, atspi.TEXT, "CharacterCount") == 123
    bus.deadline = 0
    with pytest.raises(TimeoutError, match="Accessibility query timed out"):
        bus.call(target, atspi.ACCESSIBLE, "GetRole")
    assert calls == [
        (atspi.ACCESSIBLE, "GetRole", None),
        ("org.freedesktop.DBus.Properties", "Get", ("(ss)", (atspi.TEXT, "CharacterCount"))),
    ]


def test_helper_rejects_oversized_stdin_before_opening_bus(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"x" * (1024 * 1024 + 1))))
    monkeypatch.setattr(atspi, "Bus", lambda: pytest.fail("Unexpected desktop bus connection"))
    atspi.main()
    assert capsys.readouterr().out == "null\n"


def test_helper_reports_unavailable_without_leaking_transport_diagnostics(monkeypatch, capsys):
    def no_bus():
        raise RuntimeError("private toolkit diagnostic")

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"action":"snapshot"}')))
    monkeypatch.setattr(atspi, "Bus", no_bus)
    atspi.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) is None
    assert not captured.err
