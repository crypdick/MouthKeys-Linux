from dataclasses import replace

import pytest

from mouthkeys.config import Config
from mouthkeys.hotkeys import Activation, bindings, ensure_hotkeys, parse_chord
from mouthkeys.portal_shortcuts import preferred_trigger


def test_shortcut_parsing_multiple_primary_mouse_and_portal_names():
    assert parse_chord("Super+Alt+Space") == (72, "space")
    assert parse_chord("ctrl+button8") == (4, "button8")
    assert preferred_trigger(72, "space") == "ALT+LOGO+space"
    assert preferred_trigger(5, "Return") == "CTRL+SHIFT+Return"
    with pytest.raises(ValueError, match="mouse"):
        preferred_trigger(4, "button8")
    config = Config(toggle_shortcut="super+alt+space,ctrl+button8")
    assert bindings(config)[:2] == [("toggle", 72, "space"), ("toggle", 4, "button8")]
    with pytest.raises(ValueError, match="same"):
        bindings(replace(config, cancel_shortcut="ctrl+button8"))
    with pytest.raises(ValueError, match="modifier"):
        parse_chord("a")


@pytest.mark.parametrize(
    "mode,held,expected",
    [
        ("toggle", 1, ["toggle"]),
        ("hold", 1, ["start", "stop"]),
        ("automatic", 0.1, ["toggle", "release_hold"]),
        ("automatic", 1, ["toggle", "stop"]),
    ],
)
def test_toggle_hold_and_automatic_modes_ignore_auto_repeat(mode, held, expected):
    calls = []
    activation = Activation(
        mode, lambda action, **kw: calls.append(action) or {"phase": "recording"}
    )
    activation.event("toggle", True, 1)
    activation.event("toggle", True, 1.01)
    activation.event("toggle", False, 1 + held)
    assert calls == expected


def test_multiple_primary_shortcuts_hold_until_last_key_release():
    calls = []
    activation = Activation(
        "hold", lambda action, **kw: calls.append(action) or {"phase": "recording"}
    )
    activation.event("toggle", True, 1, "keyboard")
    activation.event("toggle", True, 1.1, "mouse")
    activation.event("toggle", False, 1.2, "keyboard")
    assert calls == ["start"]
    activation.event("toggle", False, 1.3, "mouse")
    assert calls == ["start", "stop"]


def test_automatic_stop_of_existing_recording_does_not_double_stop():
    calls = []
    activation = Activation(
        "automatic", lambda action, **kw: calls.append(action) or {"phase": "stopping"}
    )
    activation.event("toggle", True, 1)
    activation.event("toggle", False, 2)
    assert calls == ["toggle"]


def test_desktop_default_never_starts_native_grabs_or_portal(monkeypatch):
    from mouthkeys import hotkeys

    monkeypatch.setattr(
        hotkeys.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Grab spawned")
    )
    ensure_hotkeys()


def test_profile_shortcut_routes_per_recording_and_release_is_id_scoped(monkeypatch):
    from mouthkeys import profiles

    profile = "e252cc1c-26c5-4a62-b5ac-69e3c100c1b9"
    monkeypatch.setattr(profiles, "load_profiles", lambda: [{"id": profile, "shortcut": "ctrl+F8"}])
    assert bindings(Config())[-1] == ("profile:" + profile, 4, "F8")
    calls = []

    def send(action, **options):
        calls.append((action, options))
        return {"phase": "recording", "id": "recording-a"}

    activation = Activation("automatic", send)
    activation.event("profile:" + profile, True, 1)
    activation.event("profile:" + profile, False, 1.1)
    assert calls == [
        ("toggle", {"profile": profile, "held": True}),
        ("release_hold", {"id": "recording-a"}),
    ]


def test_command_edit_shortcuts_use_same_hold_activation_contract():
    calls = []

    def send(action, **options):
        calls.append((action, options))
        return {"phase": "recording", "id": "edit-a"}

    activation = Activation("hold", send)
    activation.event("edit", True, 1)
    activation.event("edit", False, 2)
    assert calls == [("start", {"mode": "edit", "held": True}), ("stop", {"id": "edit-a"})]


def test_listener_closure_finalizes_only_owned_held_recording_without_paste():
    calls = []

    def send(action, **options):
        calls.append((action, options))
        return {"phase": "recording", "id": "gesture-a"}

    activation = Activation("hold", send)
    activation.event("toggle", True, 1)
    activation.finish()
    assert calls[-1] == ("stop", {"automatic": True, "id": "gesture-a"})
    activation.finish()
    assert len(calls) == 2
