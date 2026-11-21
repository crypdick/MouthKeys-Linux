from threading import Event
from types import SimpleNamespace

import pytest

from mouthkeys import accessibility, clipboard, delivery, desktop_input
from mouthkeys.config import Config


@pytest.fixture
def field():
    return {
        "field_id": "captured-field",
        "text": "hello world",
        "protected": False,
        "focused": True,
        "foreground": True,
        "selection_start": 6,
        "selection_end": 6,
        "caret": 6,
    }


@pytest.mark.parametrize(
    ("scenario", "expected", "activated", "refocused"),
    [
        ("disabled", False, False, False),
        ("edited", False, False, False),
        ("activation failed", False, True, False),
        ("focus not confirmed", False, True, False),
        ("field focus rejected", False, True, True),
        ("confirmed", True, True, True),
    ],
)
def test_automatic_return_to_start_requires_opt_in_and_rechecks_focus(
    monkeypatch, field, scenario, expected, activated, refocused
):
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    adapter = delivery.Delivery()
    focus = ["99"]
    probes = []
    calls = []
    monkeypatch.setattr(adapter, "focus", lambda: focus[0])

    def snapshot(target):
        probes.append(target)
        return field | {"text": "changed"} if scenario == "edited" else field

    def activate(args):
        calls.append(args)
        if scenario == "activation failed":
            raise RuntimeError("Window closed")
        if scenario != "focus not confirmed":
            focus[0] = "42"

    def focus_field(target):
        calls.append(target)
        return scenario != "field focus rejected"

    monkeypatch.setattr(accessibility, "read_snapshot", snapshot)
    monkeypatch.setattr(accessibility, "focus_target", focus_field)
    monkeypatch.setattr(desktop_input, "run", activate)
    config = Config(return_to_start="off" if scenario == "disabled" else "on")
    assert adapter.prepare_target(config, {"window": "42", "field": field}) is expected
    commands = [["xdotool", "windowactivate", "--sync", "42"]] if activated else []
    assert calls == commands + ([field] if refocused else [])
    assert len(probes) == (0 if scenario == "disabled" else (2 if expected else 1))


@pytest.mark.parametrize(
    ("scenario", "status", "key_requested", "restored"),
    [
        ("verified", "pasted", True, True),
        ("unverified", "paste_requested", True, False),
        ("changed field", "copied", False, False),
        ("cancelled", "cancelled", False, False),
        ("expired", "paste_requested", True, False),
        ("disabled verification", "paste_requested", True, False),
        ("invalid identity", "paste_requested", True, False),
        ("invalid range", "paste_requested", True, False),
    ],
)
def test_temporary_clipboard_is_restored_only_after_exact_field_paste(
    monkeypatch, field, scenario, status, key_requested, restored
):
    field["selection_end"] = 11
    if scenario == "invalid identity":
        field["field_id"] = None
    if scenario == "invalid range":
        field["text_start"] = "unknown"
    target = {"window": "42", "field": field}
    current = field.copy()
    adapter = delivery.Delivery()
    cancelled = Event()
    calls = []
    waits = []
    clock = iter((0, 1) if scenario == "expired" else (0, 0.1, 0.2, 0.3))
    monkeypatch.setattr(
        delivery, "time", SimpleNamespace(monotonic=lambda: next(clock), sleep=waits.append)
    )
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    monkeypatch.setattr(adapter, "focus", lambda: "42")
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: current.copy())
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    monkeypatch.setattr(
        accessibility, "insert_text", lambda *_: pytest.fail("Selected rich text needs paste")
    )

    def temporary(text, **options):
        calls.append((text, options))
        if scenario == "changed field":
            current["field_id"] = "other-field"
        if scenario == "cancelled":
            cancelled.set()
        return SimpleNamespace(finish=lambda restore: calls.append(("finish", restore)))

    def paste(_config):
        calls.append("paste")
        if scenario == "verified":
            current["text"] = "hello new world"
        elif scenario == "unverified":
            current.update({"text": "hello new world", "character_count": len("hello world")})

    monkeypatch.setattr(clipboard, "ClipboardPaste", temporary)
    monkeypatch.setattr(adapter, "paste_key", paste)
    monkeypatch.setattr(
        adapter, "copy", lambda text: pytest.fail("Temporary clipboard already set")
    )
    config = Config(
        delivery="x11",
        restore_clipboard="on",
        verify_paste="off" if scenario == "disabled verification" else "on",
    )
    assert adapter.replace_selection("new world", config, target, cancel_event=cancelled) == status
    assert calls == [
        ("new world", {"delay_ms": config.clipboard_restore_delay_ms, "exclude_history": True}),
        *(["paste"] if key_requested else []),
        ("finish", restored),
    ]
    assert (adapter.last_verified is not None) is restored
    assert waits == ([0.05] * 3 if scenario == "unverified" else [])


def test_snapshot_or_key_failure_keeps_dictation_available_after_window_check(monkeypatch):
    adapter = delivery.Delivery()
    config = Config(delivery="x11", restore_clipboard="on")
    calls = []
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    monkeypatch.setattr(adapter, "focus", lambda: "42")
    monkeypatch.setattr(delivery, "time", SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setattr(adapter, "copy", lambda text: calls.append(("copy", text)))

    def no_snapshot(*_args, **_options):
        raise RuntimeError("Clipboard unavailable")

    def failed_key(_config):
        raise OSError("Cannot send key")

    monkeypatch.setattr(clipboard, "ClipboardPaste", no_snapshot)
    monkeypatch.setattr(adapter, "paste_key", failed_key)
    with pytest.raises(OSError, match="Cannot send key"):
        adapter.deliver("recoverable", config, "42")
    assert calls == [("copy", "recoverable")]

    def snapshot(text, **_options):
        calls.append(("temporary", text))
        return SimpleNamespace(finish=lambda restore: calls.append(("restore", restore)))

    monkeypatch.setattr(clipboard, "ClipboardPaste", snapshot)
    with pytest.raises(OSError, match="Cannot send key"):
        adapter.deliver("recoverable", config, "42")
    assert calls[-2:] == [("temporary", "recoverable"), ("restore", False)]


@pytest.mark.parametrize(
    ("snap", "windows", "expected"),
    [
        ("headless", [None, None], None),
        ("inactive", ["42", "42"], "42"),
        ("confirmed", ["42", "99"], None),
        ("confirmed", ["42", "42"], "field"),
    ],
)
def test_target_capture_requires_stable_foreground_and_probes_no_headless_fields(
    monkeypatch, field, snap, windows, expected
):
    probe = []
    focus = iter(windows)
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: next(focus))
    monkeypatch.setattr(
        desktop_input, "session", lambda: "headless" if snap == "headless" else "x11"
    )

    def snapshot(*, include_protected):
        probe.append(include_protected)
        return field | {"foreground": snap == "confirmed"}

    monkeypatch.setattr(accessibility, "read_snapshot", snapshot)
    target = {"window": "42", "field": field} if expected == "field" else expected
    assert adapter.capture_target(Config()) == target
    assert probe == ([] if snap == "headless" else [True])


@pytest.mark.parametrize(
    ("case", "result", "copied"),
    [
        ("copy disabled", "pasted", False),
        ("copy enabled", "pasted", True),
        ("unverified", "insert_requested", True),
        ("cancelled", "cancelled", False),
    ],
)
def test_native_field_insert_never_retries_with_keys_when_write_is_unverified(
    monkeypatch, field, case, result, copied
):
    adapter = delivery.Delivery()
    event = Event()
    calls = []
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    monkeypatch.setattr(adapter, "focus", lambda: "42")
    monkeypatch.setattr(adapter, "copy", lambda text: calls.append(("copy", text)))
    monkeypatch.setattr(accessibility, "read_snapshot", lambda target: target.copy())
    monkeypatch.setattr(
        adapter, "paste_key", lambda _: pytest.fail("An insertion may have succeeded")
    )
    verified = field | {"text": "hello new world"}

    def terminal(_):
        if case == "cancelled":
            event.set()
        return False

    def insert(target, text):
        calls.append(("insert", target, text))
        return None if case == "unverified" else verified

    monkeypatch.setattr(adapter, "terminal", terminal)
    monkeypatch.setattr(accessibility, "insert_text", insert)
    config = Config(copy_to_clipboard="on" if case == "copy enabled" else "off")
    assert (
        adapter.replace_selection(
            "new ", config, {"window": "42", "field": field}, cancel_event=event
        )
        == result
    )
    assert calls == [
        *([] if case == "cancelled" else [("insert", field, "new ")]),
        *([("copy", "new ")] if copied else []),
    ]
    assert adapter.last_verified == (verified if result == "pasted" else None)


def test_edit_falls_back_to_copy_without_an_editable_field_and_honors_prior_cancel(
    monkeypatch, field
):
    adapter = delivery.Delivery()
    cancelled = Event()
    calls = []
    monkeypatch.setattr(adapter, "copy", calls.append)
    target = {"field": field | {"terminal": True}}
    assert (
        adapter.replace_selection("reviewed", Config(), target, cancel_event=cancelled) == "copied"
    )
    cancelled.set()
    assert (
        adapter.replace_selection("late edit", Config(), target, cancel_event=cancelled)
        == "cancelled"
    )
    assert calls == ["reviewed"]


def test_field_metadata_only_fills_in_unidentified_window_apps(monkeypatch, field):
    adapter = delivery.Delivery()
    properties = {"app": "", "window": "project notes"}
    monkeypatch.setattr(desktop_input, "window_metadata", lambda _: properties.copy())
    target = {"window": "42", "field": field | {"app": "org.example.Editor"}}
    assert adapter.target_metadata(target) == {
        "app": "org.example.Editor",
        "window": "project notes",
    }
    properties["app"] = "From X11"
    assert adapter.target_metadata(target) == properties
    assert adapter.target_metadata({"window": "42"}) == properties


@pytest.mark.parametrize(
    ("desktop", "window", "sent"),
    [("x11", "42", True), ("x11", "99", False), ("headless", "42", False)],
)
def test_field_send_requires_confirmed_window_and_a_supported_desktop(
    monkeypatch, field, desktop, window, sent
):
    adapter = delivery.Delivery()
    adapter.last_verified = field
    calls = []
    monkeypatch.setattr(desktop_input, "session", lambda: desktop)
    monkeypatch.setattr(desktop_input, "run", calls.append)
    monkeypatch.setattr(adapter, "focus", lambda: window)
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    monkeypatch.setattr(accessibility, "read_snapshot", lambda target: target.copy())
    config = Config(spoken_send="on", spoken_send_key="ctrl+return")
    target = {"window": "42", "field": field}
    assert adapter.send_key(config, target) is sent
    assert calls == ([["xdotool", "key", "--clearmodifiers", "ctrl+Return"]] if sent else [])


def test_window_identity_alone_cannot_prepare_a_field_or_send_enter(monkeypatch, field):
    adapter = delivery.Delivery()
    target = {"window": "42", "field": field}
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    monkeypatch.setattr(adapter, "focus", lambda: "42")
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: field | {"text": "user edit"})
    monkeypatch.setattr(desktop_input, "run", lambda *_: pytest.fail("No field verified"))
    assert not adapter.prepare_target(Config(), "42")
    assert not adapter.prepare_target(Config(), target)
    assert not adapter.send_key(Config(spoken_send="on"), {"window": "42"})


@pytest.mark.parametrize(
    ("stage", "preflights"),
    [
        ("first refused", 1),
        ("second refused", 2),
        ("second accepted", 2),
        ("clipboard failed", 2),
    ],
)
def test_cancellation_during_edit_preflight_or_clipboard_fallback_prevents_delivery(
    monkeypatch, field, stage, preflights
):
    field["selection_end"] += 1
    adapter = delivery.Delivery()
    event = Event()
    checks = []
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    monkeypatch.setattr(adapter, "terminal", lambda _: False)
    monkeypatch.setattr(adapter, "copy", lambda *_: pytest.fail("Copied cancelled edit"))
    monkeypatch.setattr(adapter, "paste_key", lambda *_: pytest.fail("Pasted cancelled edit"))

    def prepare(*_):
        checks.append("prepare")
        if stage == "clipboard failed" or len(checks) != preflights:
            return True
        event.set()
        return stage == "second accepted"

    def unavailable(*_args, **_options):
        assert stage == "clipboard failed"
        event.set()
        raise OSError("Temporary clipboard unavailable")

    monkeypatch.setattr(adapter, "prepare_target", prepare)
    monkeypatch.setattr(clipboard, "ClipboardPaste", unavailable)
    target = {"window": "42", "field": field}
    assert (
        adapter.replace_selection("new", Config(restore_clipboard="on"), target, cancel_event=event)
        == "cancelled"
    )
    assert checks == ["prepare"] * preflights
