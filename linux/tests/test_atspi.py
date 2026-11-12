import time
from copy import deepcopy

import pytest

from mouthkeys import accessibility, atspi_helper, delivery
from mouthkeys.config import Config

REF = (":1.42", "/org/a11y/atspi/accessible/7")


class FakeBus:
    password_role = 39
    terminal_role = 60
    focused_state = 12
    editable_state = 7
    active_state = 1
    window_roles = {23}
    window_ref = (":1.42", "/org/a11y/atspi/accessible/window")

    def __init__(self):
        self.deadline = time.monotonic() + 10
        self.text = "hello old world"
        self.caret = 9
        self.selection = (6, 9)
        self.focused = True
        self.active = True
        self.role = 1
        self.writes = []

    def prop(self, ref, interface, name):
        if name == "Parent":
            return self.window_ref
        return {"CharacterCount": len(self.text), "CaretOffset": self.caret}[name]

    def call(self, ref, interface, method, signature=None, args=()):
        if method == "GetRole":
            return (23 if ref == self.window_ref else self.role,)
        if method == "GetState":
            if ref == self.window_ref:
                return ([1 << self.active_state if self.active else 0],)
            return (
                [1 << self.editable_state | ((1 << self.focused_state) if self.focused else 0)],
            )
        if method == "GetText":
            return (self.text[args[0] : args[1]],)
        if method == "GetNSelections":
            return (int(self.selection[0] != self.selection[1]),)
        if method == "GetSelection":
            return self.selection
        if method == "GrabFocus":
            self.writes.append("focus")
            self.focused = True
            return (True,)
        if method == "InsertText":
            self.writes.append(("insert", args))
            self.text = self.text[: args[0]] + args[1] + self.text[args[0] :]
            return (True,)
        if method == "SetCaretOffset":
            self.caret = args[0]
            self.selection = (self.caret, self.caret)
            return (True,)
        raise AssertionError(method)


def test_snapshot_identity_is_unique_bus_and_object_not_names():
    bus = FakeBus()
    value = atspi_helper.snapshot(bus, REF)
    assert value["field_id"] == ":1.42|/org/a11y/atspi/accessible/7"
    assert value["text"] == "hello old world"
    assert value["context"] == "hello "
    assert value["selection_start"] == 6
    assert value["selection_end"] == 9
    assert value["foreground"] is True
    bus.role = bus.password_role
    assert atspi_helper.snapshot(bus, REF) is None


def test_local_focused_child_in_inactive_window_is_not_foreground_evidence(monkeypatch):
    bus = FakeBus()
    bus.active = False
    value = atspi_helper.snapshot(bus, REF)
    assert value["focused"] is True
    assert value["foreground"] is False
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    monkeypatch.setattr(accessibility, "read_snapshot", lambda *args: value)
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: None)
    monkeypatch.setattr(adapter, "copy", lambda _: None)
    monkeypatch.setattr(delivery, "run", lambda args: pytest.fail("Keys to inactive field"))
    target = {"window": None, "field": value}
    adapter.last_verified = value
    config = Config(delivery="wayland", paste_backend="wtype", spoken_send="on")
    assert adapter.deliver("hello", config, target) == "copied"
    assert not adapter.send_key(config, target, allow_without_insert=True)


def test_direct_insert_uses_one_insertion_and_preserves_existing_content():
    bus = FakeBus()
    bus.caret = 6
    bus.selection = (6, 6)
    before = atspi_helper.snapshot(bus, REF)
    result = atspi_helper.operate(bus, {"action": "insert", "target": before, "text": "new\nline"})
    assert result["text"] == "hello new\nlineold world"
    assert bus.writes == [("insert", (6, "new\nline", 8))]
    assert result["caret"] == len("hello new\nline")


def test_native_selected_range_never_rewrites_whole_rich_text_document():
    bus = FakeBus()
    before = atspi_helper.snapshot(bus, REF)
    assert atspi_helper.operate(bus, {"action": "insert", "target": before, "text": "new"}) is None
    assert not bus.writes


@pytest.mark.parametrize("change", ["focus", "caret", "text"])
def test_snapshot_rejects_user_gesture_during_multi_call_read(change):
    bus = FakeBus()
    original = bus.call
    changed = False

    def call(ref, interface, method, signature=None, args=()):
        nonlocal changed
        result = original(ref, interface, method, signature, args)
        if method == "GetText" and not changed:
            changed = True
            if change == "focus":
                bus.focused = False
            elif change == "caret":
                bus.caret -= 1
            else:
                bus.text = "j" + bus.text[1:]
        return result

    bus.call = call
    assert atspi_helper.snapshot(bus, REF) is None


@pytest.mark.parametrize("change", ["text", "selection", "focus"])
def test_direct_edit_refuses_changed_content_selection_or_focus(change):
    bus = FakeBus()
    before = atspi_helper.snapshot(bus, REF)
    if change == "text":
        bus.text += "typed"
    elif change == "selection":
        bus.selection = (0, 3)
    else:
        bus.focused = False
    assert atspi_helper.operate(bus, {"action": "insert", "target": before, "text": "new"}) is None
    assert not bus.writes


def test_return_focus_is_explicit_and_refuses_changed_field():
    bus = FakeBus()
    before = atspi_helper.snapshot(bus, REF)
    bus.focused = False
    assert atspi_helper.operate(bus, {"action": "focus", "target": before}) is True
    assert bus.writes == ["focus"]
    bus.text += " changed"
    assert not atspi_helper.operate(bus, {"action": "focus", "target": before})
    assert bus.writes == ["focus"]


def field_target():
    return {"window": "42", "field": atspi_helper.snapshot(FakeBus(), REF)}


def test_verified_paste_uses_same_field_and_exact_expected_text(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    target = field_target()
    current = deepcopy(target["field"])
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: deepcopy(current))
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: "42")
    calls = []
    monkeypatch.setattr(adapter, "copy", lambda text: calls.append(text))

    def paste(config):
        current["text"] = "hello new world"

    monkeypatch.setattr(adapter, "paste_key", paste)
    assert adapter.deliver("new", Config(delivery="x11"), target) == "pasted"
    assert adapter.last_verified["text"] == "hello new world"
    assert calls == ["new"]


def test_changed_pane_even_in_same_window_never_receives_paste(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "x11")
    target = field_target()
    wrong = {**target["field"], "field_id": ":1.42|/org/a11y/atspi/accessible/other"}
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: wrong)
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "focus", lambda: "42")
    monkeypatch.setattr(adapter, "copy", lambda _: None)
    monkeypatch.setattr(adapter, "paste_key", lambda _: pytest.fail("Wrong pane pasted"))
    assert adapter.deliver("new", Config(delivery="x11"), target) == "copied"


def test_wayland_verified_field_send_releases_modifiers_and_blocks_later_typing(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    target = field_target()
    target["window"] = None
    current = deepcopy(target["field"])
    current["selection_start"] = current["selection_end"] = current["caret"]
    adapter = delivery.Delivery()
    adapter.last_verified = deepcopy(current)
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: current)
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args: calls.append(args))
    config = Config(
        delivery="wayland",
        paste_backend="wtype",
        spoken_send="on",
        spoken_send_key="shift+return",
        destination_guard="window",
    )
    assert adapter.send_key(config, target)
    assert calls == [["wtype", "-M", "shift", "-k", "Return", "-m", "shift"]]
    current["text"] += " new typing"
    assert not adapter.send_key(config, target)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["caret", "selection", "count"])
def test_spoken_send_refuses_caret_or_selection_changes_without_text_change(monkeypatch, change):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    target = field_target()
    target["window"] = None
    verified = deepcopy(target["field"])
    verified["selection_start"] = verified["selection_end"] = verified["caret"]
    current = deepcopy(verified)
    if change == "caret":
        current["caret"] -= 1
        current["selection_start"] = current["selection_end"] = current["caret"]
    elif change == "selection":
        current["selection_start"] -= 1
    else:
        current["character_count"] += 1
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: current)
    monkeypatch.setattr(delivery, "run", lambda args: pytest.fail("Stale Return sent"))
    adapter = delivery.Delivery()
    adapter.last_verified = verified
    config = Config(delivery="wayland", paste_backend="wtype", spoken_send="on")
    assert not adapter.send_key(config, target)


def test_cancel_during_slow_field_probe_is_checked_at_actual_send_boundary(monkeypatch):
    import threading

    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    target = field_target()
    target["window"] = None
    verified = deepcopy(target["field"])
    verified["selection_start"] = verified["selection_end"] = verified["caret"]
    entered, release = threading.Event(), threading.Event()
    cancelled = {"value": False}

    def probe(_):
        entered.set()
        assert release.wait(2)
        return verified

    monkeypatch.setattr(accessibility, "read_snapshot", probe)
    monkeypatch.setattr(delivery, "run", lambda args: pytest.fail("Return after cancel"))
    adapter = delivery.Delivery()
    adapter.last_verified = verified
    config = Config(delivery="wayland", paste_backend="wtype", spoken_send="on")
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            adapter.send_key(config, target, before_send=lambda: not cancelled["value"])
        )
    )
    worker.start()
    assert entered.wait(2)
    cancelled["value"] = True
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert result == [False]


def test_terminal_application_identity_still_forces_clipboard_when_field_role_is_plain(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    target = field_target()
    target["window"] = None
    target["field"]["app"] = "com.mitchellh.ghostty"
    target["field"]["selection_start"] = target["field"]["selection_end"] = target["field"]["caret"]
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: target["field"])
    monkeypatch.setattr(
        accessibility, "insert_text", lambda *args: pytest.fail("Direct terminal insert")
    )
    adapter = delivery.Delivery()
    calls = []
    monkeypatch.setattr(adapter, "copy", lambda _: calls.append("copy"))
    monkeypatch.setattr(adapter, "paste_key", lambda _: calls.append("paste"))
    config = Config(
        delivery="wayland", insertion_mode="direct", paste_backend="wtype", verify_paste="off"
    )
    assert adapter.deliver("hello", config, target) == "paste_requested"
    assert calls == ["copy", "paste"]


def test_terminal_detection_does_not_guess_football_is_foot_terminal():
    assert delivery.terminal_name("org.codeberg.dnkl.foot")
    assert delivery.terminal_name('WM_CLASS = "ghostty", "com.mitchellh.ghostty"')
    assert not delivery.terminal_name("Football note editor")


@pytest.mark.parametrize("change", [None, "text", "caret", "selection", "protected"])
def test_explicit_send_only_uses_unchanged_captured_field_without_prior_insertion(
    monkeypatch, change
):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    target = field_target()
    target["window"] = None
    target["field"]["selection_start"] = target["field"]["selection_end"] = target["field"]["caret"]
    current = deepcopy(target["field"])
    if change == "text":
        current["text"] += " user edit"
    elif change == "caret":
        current["caret"] -= 1
        current["selection_start"] = current["selection_end"] = current["caret"]
    elif change == "selection":
        current["selection_start"] -= 1
    elif change == "protected":
        current["protected"] = True
    monkeypatch.setattr(accessibility, "read_snapshot", lambda _: current)
    adapter = delivery.Delivery()
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args: calls.append(args))
    config = Config(delivery="wayland", paste_backend="wtype", spoken_send="on")
    assert adapter.send_key(config, target, allow_without_insert=True) is (change is None)
    assert len(calls) == int(change is None)
    calls.clear()
    assert not adapter.send_key(config, target)  # No fallback after a failed/unverified paste.
    assert not calls


def test_snapshot_api_never_accepts_protected_results(monkeypatch):
    monkeypatch.setattr(
        accessibility, "_request", lambda *args: {"text": "secret", "protected": True}
    )
    assert accessibility.read_snapshot() is None


def test_large_document_keeps_field_identity_with_bounded_window_but_refuses_whole_edit():
    bus = FakeBus()
    bus.text = "x" * 150000
    bus.caret = 140000
    bus.selection = (139999, 140000)
    value = atspi_helper.snapshot(bus, REF)
    assert value["field_id"] == ":1.42|/org/a11y/atspi/accessible/7"
    assert len(value["text"]) == 4001
    assert value["text_start"] == 137999
    assert not value["complete"]
    assert atspi_helper.operate(bus, {"action": "insert", "target": value, "text": "new"}) is None
    assert not bus.writes


def test_native_wayland_metadata_reads_actual_compositor_ids(monkeypatch):
    import json
    from types import SimpleNamespace

    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    tree = {
        "id": 1,
        "nodes": [{"id": 77, "app_id": "org.example.Editor", "name": "Actual window", "nodes": []}],
    }
    monkeypatch.setattr(
        delivery, "run", lambda args: SimpleNamespace(stdout=json.dumps(tree).encode())
    )
    assert delivery.Delivery().target_metadata("sway:77") == {
        "app": "org.example.Editor",
        "window": "Actual window",
    }


def test_wayland_container_spoken_send_rechecks_native_focus(monkeypatch):
    monkeypatch.setattr(delivery, "session", lambda: "wayland")
    adapter = delivery.Delivery()
    monkeypatch.setattr(adapter, "terminal", lambda _: True)
    focus = iter(["sway:77", "sway:77"])
    monkeypatch.setattr(adapter, "focus", lambda: next(focus))
    calls = []
    monkeypatch.setattr(delivery, "run", lambda args: calls.append(args))
    config = Config(
        delivery="wayland",
        paste_backend="wtype",
        spoken_send="on",
        spoken_send_key="shift+return",
        destination_guard="window",
    )
    assert adapter.send_key(config, "sway:77")
    assert calls == [["wtype", "-k", "Return"]]
    focus = iter(["sway:77", "sway:88"])
    assert not adapter.send_key(config, "sway:77")
    assert len(calls) == 1
