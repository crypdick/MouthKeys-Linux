import io
import json
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from mouthkeys import portal_shortcuts as portal
from mouthkeys.config import Config

POPEN_TYPE = subprocess.Popen
BASE = Config(shortcut_backend="portal", toggle_shortcut="ctrl+space", cancel_shortcut="alt+Escape")
EVENT = {"action": "toggle", "id": "toggle_0", "down": True, "at": 1.0}


def frame(value):
    return json.dumps(value).encode() + b"\n"


@pytest.fixture
def transport(monkeypatch):
    children = []
    channels = {}
    popen = Mock(side_effect=lambda *_args, **_kwargs: children.pop(0))
    monkeypatch.setattr(
        portal,
        "subprocess",
        SimpleNamespace(
            PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired, Popen=popen
        ),
    )
    monkeypatch.setattr(
        portal, "python_command", Mock(return_value=["python", "-P", "-m", "helper"])
    )
    monkeypatch.setattr(
        portal,
        "importlib",
        SimpleNamespace(util=SimpleNamespace(find_spec=Mock(return_value=True))),
    )
    monkeypatch.setattr(portal, "time", SimpleNamespace(monotonic=Mock(return_value=1.0)))
    send = Mock(return_value={"phase": "recording", "id": "portal-session"})
    monkeypatch.setattr(portal, "request", send)
    monkeypatch.setattr(Config, "load", Mock(return_value=BASE))

    def select(descriptors, *_args):
        channel = channels[descriptors[0]]
        if channel[0] is None:
            channel.popleft()
            return [], [], []
        return descriptors, [], []

    def read(fd, size):
        assert size == 4096
        return channels[fd].popleft()

    def child(chunks, *, polled=0, exit_code=0):
        process = Mock()
        process.__class__ = POPEN_TYPE
        process.stdin = Mock()
        process.stdout = io.BytesIO()
        fd = len(channels) + 10
        process.stdout.fileno = Mock(return_value=fd)
        process.stdout.close = Mock(wraps=process.stdout.close)
        channels[fd] = deque(chunks)
        process.poll.return_value = polled
        process.wait.return_value = exit_code
        children.append(process)
        return process

    monkeypatch.setattr(portal, "select", SimpleNamespace(select=select))
    monkeypatch.setattr(portal, "os", SimpleNamespace(read=read))
    return SimpleNamespace(child=child, popen=popen, send=send)


def test_portal_rebind_waits_for_release_then_uses_new_mode(transport, monkeypatch):
    start = frame(EVENT)
    release = frame({**EVENT, "down": False, "at": 2.0})
    first = transport.child([start[:9], start[9:], None, release], polled=None)
    second = transport.child([start, release], polled=None)
    updated = replace(BASE, hotkey_mode="hold", toggle_shortcut="super+space")
    desktop = replace(updated, shortcut_backend="desktop")
    Config.load.side_effect = [BASE, updated, updated, updated, desktop]
    portal.time.monotonic.side_effect = [0.2, 2, 2.2, 3.5, 5, 7]
    assert portal.launch() == 0
    assert transport.send.call_args_list == [
        call("toggle"),
        call("start", held=True),
        call("stop", id="portal-session"),
    ]
    for process, key in ((first, "CTRL+space"), (second, "LOGO+space")):
        entries = json.loads(process.stdin.write.call_args.args[0])
        assert entries[0] == {"id": "toggle_0", "action": "toggle", "trigger": key}
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=2)
        process.stdout.close.assert_called_once_with()
    assert transport.popen.call_count == 2


@pytest.mark.parametrize("code", [0, 1])
def test_eof_reaps_child_and_finishes_held_recording(transport, code):
    Config.load.return_value = replace(BASE, hotkey_mode="hold")
    process = transport.child([frame(EVENT), b""], polled=code, exit_code=code)
    if code:
        with pytest.raises(RuntimeError, match="portal unavailable, cancelled, or denied"):
            portal.launch()
    else:
        assert portal.launch() == 0
    assert transport.send.call_args_list == [
        call("start", held=True),
        call("stop", automatic=True, id="portal-session"),
    ]
    process.wait.assert_called_once_with(timeout=2)
    process.terminate.assert_not_called()
    process.stdout.close.assert_called_once_with()


def test_backend_switch_reaps_child_even_when_finishing_owned_recording_fails(transport):
    held = replace(BASE, hotkey_mode="hold")
    Config.load.side_effect = [held, replace(held, shortcut_backend="desktop")]
    process = transport.child([frame(EVENT)], polled=None)
    transport.send.side_effect = [
        {"phase": "recording", "id": "owned"},
        OSError("Daemon unavailable"),
    ]
    process.wait.side_effect = [subprocess.TimeoutExpired("helper", 2), -9]
    with pytest.raises(OSError, match="Daemon unavailable"):
        portal.launch()
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_args_list == [call(timeout=2), call(timeout=2)]
    process.stdout.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (frame({"error": "User denied consent"}), RuntimeError, "User denied consent"),
        (frame(None), TypeError, "Invalid portal shortcut event"),
        (frame({**EVENT, "id": "unknown"}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "action": "cancel"}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "id": 42}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "action": None}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "down": 1}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "at": True}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "at": None}), ValueError, "Invalid portal shortcut event"),
        (frame({**EVENT, "at": float("inf")}), ValueError, "Invalid portal shortcut timestamp"),
        (frame({**EVENT, "at": 10**400}), ValueError, "Invalid portal shortcut timestamp"),
        (b"{", RuntimeError, "Incomplete shortcut event"),
    ],
)
def test_invalid_child_frames_never_issue_commands(transport, payload, error, message):
    process = transport.child([payload, b""], polled=None)
    with pytest.raises(error, match=message):
        portal.launch()
    transport.send.assert_not_called()
    process.terminate.assert_called_once_with()


@pytest.mark.parametrize("end", [b"!", b"!\n"])
def test_oversized_child_frame_is_rejected(transport, end):
    process = transport.child([b"x" * 4096] * 16 + [end], polled=None)
    with pytest.raises(RuntimeError, match="Oversized shortcut event"):
        portal.launch()
    transport.send.assert_not_called()
    process.terminate.assert_called_once_with()


def test_oversized_config_cleans_child_without_sending_partial_input(transport, monkeypatch):
    monkeypatch.setattr(portal, "MAX_MESSAGE_BYTES", 8)
    process = transport.child([], polled=None)
    with pytest.raises(ValueError, match="Oversized portal shortcut configuration"):
        portal.launch()
    process.stdin.write.assert_not_called()
    process.stdin.close.assert_called_once_with()
    process.terminate.assert_called_once_with()


@pytest.mark.parametrize(
    ("has_gi", "has_system_python"), [(False, True), (False, False), (True, True)]
)
def test_helper_activation_selects_system_gi_only_when_needed(
    transport, monkeypatch, has_gi, has_system_python
):
    portal.importlib.util.find_spec.return_value = True if has_gi else None
    python = Mock(spec=Path)
    python.is_file.return_value = has_system_python
    python.__str__ = Mock(return_value="/usr/bin/python3")
    monkeypatch.setattr(
        portal, "Path", lambda name: python if name == "/usr/bin/python3" else Path(name)
    )
    transport.child([b""])
    assert portal.launch() == 0
    command = transport.popen.call_args.args[0]
    if not has_gi and has_system_python:
        assert command[:2] == ["/usr/bin/python3", "-I"]
        assert Path(command[2]).name == "portal_helper.py"
        assert Path(command[2]).is_file()
    else:
        portal.python_command.assert_called_once_with("mouthkeys.portal_helper")
        assert command == ["python", "-P", "-m", "helper"]


def test_real_child_pipes_carry_events_without_a_desktop(monkeypatch):
    Config(shortcut_backend="portal", hotkey_mode="hold", toggle_shortcut="ctrl+space").save()
    script = (
        "import json, sys\n"
        "entry = json.load(sys.stdin)[0]\n"
        "for down, at in ((True, 1.0), (False, 2.0)):\n"
        "    print(json.dumps(dict(action=entry['action'], id=entry['id'], down=down, at=at)))\n"
    )
    monkeypatch.setattr(portal, "python_command", lambda *_: [sys.executable, "-u", "-c", script])
    monkeypatch.setattr(
        portal,
        "importlib",
        SimpleNamespace(util=SimpleNamespace(find_spec=Mock(return_value=True))),
    )
    send = Mock(return_value={"phase": "recording", "id": "real-pipe-test"})
    monkeypatch.setattr(portal, "request", send)
    assert portal.launch() == 0
    assert send.call_args_list == [call("start", held=True), call("stop", id="real-pipe-test")]
