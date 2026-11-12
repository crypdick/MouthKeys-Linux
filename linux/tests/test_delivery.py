from dataclasses import replace
from types import SimpleNamespace

import pytest

from mouthkeys import delivery
from mouthkeys.config import Config


@pytest.mark.parametrize("desktop,target", [("x11", "10"), ("wayland", "sway:42")])
def test_default_guard_never_delivers_or_sends_to_window_only_identity(
    monkeypatch, desktop, target
):
    monkeypatch.setattr(delivery, "session", lambda: desktop)
    adapter = delivery.Delivery()
    calls = []
    monkeypatch.setattr(adapter, "copy", lambda text: calls.append(text))
    monkeypatch.setattr(adapter, "focus", lambda: target)
    monkeypatch.setattr(adapter, "paste_key", lambda cfg: pytest.fail("Window-only paste"))
    monkeypatch.setattr(delivery, "run", lambda *a, **kw: pytest.fail("Window-only Send"))
    config = Config(delivery=desktop, spoken_send="on")
    assert adapter.deliver("retained", config, target) == "copied"
    assert not adapter.send_key(config, target, allow_without_insert=True)
    assert calls == ["retained"]


@pytest.mark.parametrize("desktop", ["x11", "wayland"])
def test_copy_unicode_uses_stdin_not_shell(monkeypatch, desktop):
    monkeypatch.setattr(delivery, "session", lambda: desktop)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(delivery.subprocess, "run", run)
    text = '日本語 café\n$(touch /tmp/never) "hello"'
    delivery.Delivery().copy(text)
    args, kwargs = calls[0]
    assert text not in args
    assert kwargs["input"] == text.encode()
    assert not kwargs.get("shell")
    # Forked selection owners must not keep the launcher's capture pipes alive.
    assert kwargs["stdout"] == kwargs["stderr"] == delivery.subprocess.DEVNULL


@pytest.mark.parametrize(
    "focuses,target,expected",
    [
        (["10", "10"], "10", "paste_requested"),
        (["20"], "10", "copied"),
        (["10", "20"], "10", "copied"),
        ([], None, "copied"),
    ],
)
def test_x11_focus_guard(monkeypatch, focuses, target, expected):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)
    adapter = delivery.Delivery()
    calls = []
    focus = iter(focuses)
    monkeypatch.setattr(adapter, "focus", lambda: next(focus))
    monkeypatch.setattr(adapter, "copy", lambda text: calls.append("copy"))
    monkeypatch.setattr(adapter, "paste_key", lambda _: calls.append("paste"))
    assert (
        adapter.deliver("hello", Config(destination_guard="window", delivery="x11"), target)
        == expected
    )
    assert calls == (["copy", "paste"] if expected == "paste_requested" else ["copy"])


def test_wayland_never_auto_pastes_even_with_xwayland_display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert delivery.session() == "wayland"
    adapter = delivery.Delivery()
    calls = []
    monkeypatch.setattr(adapter, "copy", lambda _: calls.append("copy"))
    monkeypatch.setattr(adapter, "paste_key", lambda _: pytest.fail("Unexpected auto paste"))
    assert (
        adapter.deliver("hello", Config(destination_guard="window", delivery="x11"), "123")
        == "copied"
    )
    assert calls == ["copy"]


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("xdotool", ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"]),
        ("wtype", ["wtype", "-M", "ctrl", "-M", "shift", "-k", "v", "-m", "shift", "-m", "ctrl"]),
        ("ydotool", ["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]),
    ],
)
def test_terminal_paste_chords_release_modifiers(monkeypatch, backend, expected):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args: calls.append(args))
    delivery.Delivery().paste_key(
        replace(
            Config(
                destination_guard="window",
            ),
            paste_backend=backend,
        )
    )
    assert calls == [expected]


def test_wayland_auto_backend_requires_explicit_configuration(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    with pytest.raises(ValueError, match="Wayland"):
        delivery.Delivery().paste_key(
            Config(
                destination_guard="window",
            )
        )


def test_headless_never_changes_clipboard():
    with pytest.raises(RuntimeError, match="No desktop"):
        delivery.Delivery().copy("hello")


@pytest.mark.parametrize(
    "focuses,expected", [(["10", "10"], True), (["10", "20"], False), (["20"], False)]
)
def test_spoken_send_rechecks_focus_and_never_activates_target(monkeypatch, focuses, expected):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    adapter = delivery.Delivery()
    focus = iter(focuses)
    monkeypatch.setattr(adapter, "focus", lambda: next(focus))
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args: calls.append(args))
    assert (
        adapter.send_key(
            Config(destination_guard="window", spoken_send="on", spoken_send_key="shift+return"),
            "10",
        )
        is expected
    )
    assert calls == ([["xdotool", "key", "--clearmodifiers", "shift+Return"]] if expected else [])


def test_spoken_send_terminal_uses_plain_return(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: "10")
    monkeypatch.setattr(adapter, "terminal", lambda _: True)
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args: calls.append(args))
    assert adapter.send_key(
        Config(destination_guard="window", spoken_send="on", spoken_send_key="ctrl+return"), "10"
    )
    assert calls[0][-1] == "Return"


def test_wayland_auto_paste_requires_explicit_delivery_and_stop_target(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)
    adapter = delivery.Delivery()
    calls = []
    monkeypatch.setattr(adapter, "copy", lambda _: calls.append("copy"))
    monkeypatch.setattr(adapter, "paste_key", lambda _: calls.append("paste"))
    config = Config(destination_guard="window", delivery="wayland", paste_backend="wtype")
    assert adapter.deliver("hello", config, None) == "copied"
    assert calls == ["copy"]
    calls.clear()
    monkeypatch.setattr(adapter, "focus", lambda: "sway:42")
    assert adapter.deliver("hello", config, "sway:42") == "paste_requested"
    assert calls == ["copy", "paste"]
    assert not adapter.send_key(config, "sway:42")
    calls.clear()
    monkeypatch.setattr(adapter, "focus", lambda: "sway:99")
    assert adapter.deliver("hello", config, "sway:42") == "copied"
    assert calls == ["copy"]


def test_wayland_focus_uses_compositor_identity_or_fails_closed(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    adapter = delivery.Delivery()
    assert adapter.focus() is None
    monkeypatch.setenv("SWAYSOCK", "/fake/sway.sock")
    monkeypatch.setattr(
        delivery,
        "run",
        lambda args: SimpleNamespace(stdout=b'{"nodes":[{"type":"con","id":42,"focused":true}]}'),
    )
    assert adapter.focus() == "sway:42"
    monkeypatch.delenv("SWAYSOCK")
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "fake")
    monkeypatch.setattr(
        delivery, "run", lambda args: SimpleNamespace(stdout=b'{"address":"0xabcd"}')
    )
    assert adapter.focus() == "hypr:0xabcd"


@pytest.mark.parametrize("changed,restore", [(False, True), (True, False)])
def test_clipboard_restores_only_after_paste_request(monkeypatch, changed, restore):
    from mouthkeys import clipboard

    monkeypatch.setattr(delivery, "session", lambda: "x11")
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)
    adapter = delivery.Delivery()
    focuses = iter(["10", "10", "20" if changed else "10"])
    monkeypatch.setattr(adapter, "focus", lambda: next(focuses))
    calls = []
    delays = []

    def snapshot(text, **kwargs):
        delays.append(kwargs["delay_ms"])
        return SimpleNamespace(finish=lambda value: calls.append(value))

    monkeypatch.setattr(
        clipboard,
        "ClipboardPaste",
        snapshot,
    )
    monkeypatch.setattr(adapter, "copy", lambda _: pytest.fail("Snapshot already copied"))
    monkeypatch.setattr(adapter, "paste_key", lambda _: calls.append("paste"))
    result = adapter.deliver(
        "hello",
        Config(
            destination_guard="window",
            delivery="x11",
            restore_clipboard="on",
            clipboard_restore_delay_ms=750,
        ),
        "10",
    )
    assert result == ("copied" if changed else "paste_requested")
    assert calls == ([False] if changed else ["paste", restore])
    assert delays == [750]


def test_direct_insertion_uses_stdin_and_never_types_into_terminals(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: "10")
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(adapter, "copy", lambda _: calls.append("copy"))
    config = Config(
        destination_guard="window", delivery="x11", insertion_mode="direct", copy_to_clipboard="off"
    )
    assert adapter.deliver("café $(never)", config, "10") == "insert_requested"
    assert calls[0][1]["input"] == "café $(never)".encode()
    calls.clear()
    monkeypatch.setattr(adapter, "terminal", lambda _: True)
    monkeypatch.setattr(adapter, "paste_key", lambda _: calls.append("paste"))
    assert adapter.deliver("hello", config, "10") == "paste_requested"
    assert calls == ["copy", "paste"]
    calls.clear()
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    assert adapter.deliver("hello\nworld", config, "10") == "paste_requested"
    assert calls == ["copy", "paste"]  # Newlines must not become simulated Return presses.


def test_destination_metadata_uses_real_x11_properties_and_not_guesswork(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    calls = []

    def run(args):
        calls.append(args)
        return SimpleNamespace(
            stdout=(
                b'WM_CLASS(STRING) = "instance", "Ghostty"\n'
                b'_NET_WM_NAME(UTF8_STRING) = "project \\"notes\\""\n'
                b'WM_NAME(STRING) = "fallback"\n'
            )
        )

    monkeypatch.setattr(delivery, "run", run)
    assert delivery.Delivery().target_metadata("42") == {
        "app": "Ghostty",
        "window": 'project "notes"',
    }
    assert calls[0][:3] == ["xprop", "-id", "42"]
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    assert delivery.Delivery().target_metadata("42") == {}
    assert len(calls) == 1


def test_wayland_container_restore_uses_actual_clipboard_owner_helper_when_supported(monkeypatch):
    from mouthkeys import clipboard

    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    monkeypatch.setattr(delivery.time, "sleep", lambda _: None)
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: "sway:77")
    calls = []
    monkeypatch.setattr(
        clipboard,
        "ClipboardPaste",
        lambda *a, **kw: SimpleNamespace(finish=lambda value: calls.append(("finish", value))),
    )
    monkeypatch.setattr(adapter, "copy", lambda _: pytest.fail("Helper already copied"))
    monkeypatch.setattr(adapter, "paste_key", lambda _: calls.append("paste"))
    config = Config(
        destination_guard="window",
        delivery="wayland",
        paste_backend="wtype",
        restore_clipboard="on",
    )
    assert adapter.deliver("hello", config, "sway:77") == "paste_requested"
    assert calls == ["paste", ("finish", True)]
