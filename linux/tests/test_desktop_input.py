import json
import subprocess

import pytest

from mouthkeys import desktop_input
from mouthkeys.config import Config


@pytest.fixture
def command(monkeypatch):
    def stub(*, stdout=b"", error=None):
        calls = []

        def run(args, **options):
            calls.append((args, options))
            if error is not None:
                raise error
            return subprocess.CompletedProcess(args, 0, stdout)

        monkeypatch.setattr(subprocess, "run", run)
        return calls

    return stub


@pytest.mark.parametrize(
    ("env", "output", "expected"),
    [
        ("DISPLAY", b" 42\n", "42"),
        ("DISPLAY", b"\n", None),
        ("XDG_SESSION_TYPE", b'{"address":"0xab12"}', "hypr:0xab12"),
        ("XDG_SESSION_TYPE", b'{"address":"0x0"}', None),
        ("XDG_SESSION_TYPE", b'{"address":"bad-address"}', None),
        ("XDG_SESSION_TYPE", b"[]", None),
        ("XDG_SESSION_TYPE", b"incomplete", None),
    ],
)
def test_window_focus_is_identified_by_the_selected_compositor(
    monkeypatch, command, env, output, expected
):
    monkeypatch.setenv("DISPLAY", ":91")  # XWayland also has a DISPLAY.
    monkeypatch.setenv(env, "wayland" if env == "XDG_SESSION_TYPE" else ":91")
    monkeypatch.delenv("SWAYSOCK", raising=False)
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "fake")
    calls = command(stdout=output)
    assert desktop_input.focus() == expected
    wanted = (
        ["xdotool", "getwindowfocus"] if env == "DISPLAY" else ["hyprctl", "activewindow", "-j"]
    )
    assert calls == [(wanted, {"input": None, "check": True, "timeout": 3, "capture_output": True})]


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        ({"floating_nodes": [{"type": "con", "id": 7, "focused": True}]}, "sway:7"),
        ({"nodes": [{"type": "workspace", "id": 7, "focused": True}]}, None),
        ({"nodes": [{"type": "con", "id": 7, "focused": "false"}]}, None),
        ({"nodes": [{"type": "con", "id": False, "focused": True}]}, None),
        ({"nodes": 7}, None),
        ([], None),
    ],
)
def test_sway_focus_checks_node_type_and_integer_identity(monkeypatch, command, tree, expected):
    monkeypatch.setenv("WAYLAND_DISPLAY", "fake")
    monkeypatch.setenv("SWAYSOCK", "/fake/sway.sock")
    calls = command(stdout=json.dumps(tree).encode())
    assert desktop_input.focus() == expected
    assert [args for args, _ in calls] == [["swaymsg", "-t", "get_tree", "-r"]]


@pytest.mark.parametrize(
    ("target", "tree", "expected"),
    [
        (
            "sway:7",
            {
                "floating_nodes": [
                    {"id": 7, "window_properties": {"class": "Ghostty"}, "name": "pane"}
                ]
            },
            {"app": "Ghostty", "window": "pane"},
        ),
        ("sway:8", {"nodes": [{"id": 7}]}, {}),
        (
            "hypr:0xabc",
            [None, {"address": "0xdef"}, {"address": "0xabc", "class": "Ghostty", "title": "pane"}],
            {"app": "Ghostty", "window": "pane"},
        ),
        ("hypr:0xabc", {"address": "0xabc"}, {}),
        ("hypr:0xabc", [{"address": "0xdef"}], {"app": "", "window": ""}),
    ],
)
def test_wayland_window_metadata_never_uses_a_different_container(
    monkeypatch, command, target, tree, expected
):
    monkeypatch.setenv("WAYLAND_DISPLAY", "fake")
    calls = command(stdout=json.dumps(tree).encode())
    assert desktop_input.window_metadata(target) == expected
    wanted = (
        ["swaymsg", "-t", "get_tree", "-r"]
        if target.startswith("sway:")
        else ["hyprctl", "clients", "-j"]
    )
    assert [args for args, _ in calls] == [wanted]


def test_x11_window_properties_distinguish_missing_metadata_from_unknown_class(
    monkeypatch, command
):
    monkeypatch.setenv("DISPLAY", ":91")
    calls = command(
        stdout=b'WM_CLASS(STRING) = "Instance", "Ghostty"\nWM_NAME(STRING) = "fallback"\n'
    )
    assert desktop_input.window_metadata("0x12") == {"app": "Ghostty", "window": "fallback"}
    assert '"ghostty"' in desktop_input.window_class("0x12")
    assert [args for args, _ in calls] == [
        ["xprop", "-id", "0x12", "WM_CLASS", "_NET_WM_NAME", "WM_NAME"],
        ["xprop", "-id", "0x12", "WM_CLASS"],
    ]
    command(stdout=b"WM_CLASS: not found.\n")
    assert desktop_input.window_class("0x12") is None
    assert desktop_input.window_metadata("0x12") == {"app": "", "window": ""}


def test_unknown_desktop_and_invalid_window_ids_never_run_a_command(monkeypatch, command):
    calls = command()
    assert desktop_input.focus() is None
    assert desktop_input.window_class("42") is None
    assert desktop_input.window_metadata(None) == {}
    assert desktop_input.window_metadata("42") == {}
    monkeypatch.setenv("DISPLAY", ":91")
    for invalid in ("", "not-an-id", "-root", "sway:7; kill", "hypr:0xdb; kill"):
        assert desktop_input.window_class(invalid) is None
        assert desktop_input.window_metadata(invalid) == {}
        assert not desktop_input.activate_window(invalid)
    assert calls == []


@pytest.mark.parametrize(
    ("target", "args"),
    [
        ("42", ["xdotool", "windowactivate", "--sync", "42"]),
        ("sway:7", ["swaymsg", "[con_id=7]", "focus"]),
        ("hypr:0xabc", ["hyprctl", "dispatch", "focuswindow", "address:0xabc"]),
    ],
)
def test_explicit_activation_sends_only_a_valid_window_identity(monkeypatch, command, target, args):
    monkeypatch.setenv("DISPLAY", ":91")
    if ":" in target:
        monkeypatch.setenv("WAYLAND_DISPLAY", "fake")
    calls = command()
    assert desktop_input.activate_window(target)
    assert [executed for executed, _ in calls] == [args]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError("xdotool"), "Install xdotool"),
        (
            subprocess.CalledProcessError(1, ["xdotool"], stderr=b"  focus lost\n"),
            "failed: focus lost",
        ),
        (subprocess.CalledProcessError(1, ["xdotool"]), "failed:"),
    ],
)
def test_failed_command_is_reported_for_writes_and_fails_closed_for_probes(
    monkeypatch, command, error, message
):
    monkeypatch.setenv("DISPLAY", ":91")
    command(error=error)
    assert desktop_input.focus() is None
    assert desktop_input.window_class("42") is None
    assert desktop_input.window_metadata("42") == {}
    with pytest.raises(RuntimeError, match=message):
        desktop_input.paste_key(Config(paste_backend="xdotool"))


def test_missing_clipboard_tool_and_unavailable_paste_have_actionable_errors(monkeypatch, command):
    monkeypatch.setenv("WAYLAND_DISPLAY", "fake")
    calls = command(error=FileNotFoundError("wl-copy"))
    with pytest.raises(RuntimeError, match="Install wl-copy"):
        desktop_input.copy("retained")
    assert [args for args, _ in calls] == [["wl-copy", "--type", "text/plain;charset=utf-8"]]
    with pytest.raises(ValueError, match="cannot paste into native Wayland"):
        desktop_input.paste_key(Config(paste_backend="xdotool"))
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("auto", ["xdotool", "key", "--clearmodifiers", "ctrl+v"]),
        ("wtype", ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]),
        ("ydotool", ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]),
    ],
)
def test_plain_paste_does_not_press_shift(monkeypatch, command, backend, expected):
    monkeypatch.setenv("DISPLAY", ":91")
    calls = command()
    desktop_input.paste_key(Config(paste_backend=backend, paste_shortcut="ctrl+v"))
    assert [args for args, _ in calls] == [expected]


@pytest.mark.parametrize(
    ("backend", "key", "expected"),
    [
        ("ydotool", "return", ["ydotool", "key", "28:1", "28:0"]),
        ("ydotool", "ctrl+return", ["ydotool", "key", "29:1", "28:1", "28:0", "29:0"]),
        ("wtype", "shift+return", ["wtype", "-M", "shift", "-k", "Return", "-m", "shift"]),
        ("auto", "return", None),
    ],
)
def test_wayland_send_reserves_key_action_only_when_configured(command, backend, key, expected):
    calls = command()
    reservations = []
    config = Config(spoken_send_key=key, paste_backend=backend)

    def reserve():
        assert calls == []
        reservations.append("attempt")
        return len(reservations) == 2

    assert not desktop_input.send_wayland(config, terminal=False, before_send=reserve)
    assert desktop_input.send_wayland(config, terminal=False, before_send=reserve) is (
        expected is not None
    )
    assert reservations == ([] if expected is None else ["attempt", "attempt"])
    assert [args for args, _ in calls] == ([] if expected is None else [expected])


def test_x11_send_checks_cancellation_and_rejects_unknown_key(command):
    calls = command()

    def cancelled():
        return False

    assert not desktop_input.send_x11(Config(), terminal=False, before_send=cancelled)
    config = Config(spoken_send_key="unknown")
    with pytest.raises(ValueError, match="Unsupported Spoken Send key"):
        desktop_input.send_x11(config, terminal=False, before_send=cancelled)
    assert calls == []


@pytest.mark.parametrize(
    ("desktop", "config", "tools"),
    [
        ("headless", Config(), {"ffmpeg"}),
        (
            "wayland",
            Config(
                delivery="wayland",
                paste_backend="wtype",
                insertion_mode="direct",
                pause_media="on",
                sound_stop_file="stop.wav",
            ),
            {"ffmpeg", "wl-copy", "wtype", "xprop", "playerctl", "paplay"},
        ),
        (
            "x11",
            Config(delivery="x11", sound_preset="mouthkeys"),
            {"ffmpeg", "xclip", "xdotool", "paplay"},
        ),
        ("x11", Config(paste_backend="ydotool"), {"ffmpeg", "xclip", "ydotool"}),
    ],
)
def test_doctor_reports_configured_dependencies_without_running_desktop_tools(
    monkeypatch, command, desktop, config, tools
):
    calls = command()
    monkeypatch.setattr(desktop_input, "session", lambda: desktop)
    monkeypatch.setattr(desktop_input.shutil, "which", lambda name: f"/fake/{name}")
    assert desktop_input.doctor(config) == {
        "session": desktop,
        "commands": {name: f"/fake/{name}" for name in tools},
        "audio_source": config.audio_source,
        "model": config.model,
        "delivery": config.delivery,
        "delivery_warning": None,
    }
    assert calls == []


@pytest.mark.parametrize(
    ("desktop", "config", "message"),
    [
        ("headless", Config(delivery="wayland"), "detected session is headless"),
        ("wayland", Config(delivery="wayland"), "compatible wtype"),
        ("x11", Config(delivery="x11", paste_backend="ydotool"), "Install xdotool for X11"),
    ],
)
def test_delivery_warning_is_available_without_probing_focus(
    monkeypatch, command, desktop, config, message
):
    calls = command()
    monkeypatch.setattr(desktop_input, "session", lambda: desktop)
    monkeypatch.setattr(
        desktop_input.shutil, "which", lambda name: None if name == "xdotool" else name
    )
    assert message in desktop_input.automatic_delivery_warning(config)
    assert calls == []
