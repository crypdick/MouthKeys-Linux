from pathlib import Path
from types import SimpleNamespace

import pytest

from mouthkeys import desktop
from mouthkeys.config import Config


def test_microphone_discovery_keeps_default_when_desktop_probe_is_unavailable(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")

    def unavailable(*_args):
        raise FileNotFoundError("pactl is not installed")

    monkeypatch.setattr(desktop, "command", unavailable)
    assert desktop.choose_source(Config()) == "default"
    assert desktop.source_candidates(Config(audio_source="explicit")) == ["explicit"]


@pytest.mark.parametrize(
    ("default", "available", "fallback", "expected"),
    [
        ("desk", ["desk", "backup"], "backup", "desk"),
        ("desk", ["desk", "backup"], "backup,desk", "backup"),
        ("missing", ["monitor.monitor", "backup"], "missing", "backup"),
        ("missing", ["monitor.monitor"], "missing", None),
        ("", [], "missing", None),
    ],
)
def test_automatic_source_uses_order_without_opening_monitor_inputs(
    monkeypatch, default, available, fallback, expected
):
    monkeypatch.setattr(desktop, "source_names", lambda: available)
    monkeypatch.setattr(desktop, "command", lambda _: default)
    config = Config(microphone_fallback=fallback)
    if expected is None:
        with pytest.raises(ValueError, match="No available microphone"):
            desktop.choose_source(config)
    else:
        assert desktop.choose_source(config) == expected


def test_alternate_device_order_does_not_promote_default_from_fallback_list(monkeypatch):
    monkeypatch.setattr(desktop, "source_names", lambda: ["backup", "default-hw", "other"])
    monkeypatch.setattr(desktop, "command", lambda _: "default-hw")
    config = Config(audio_source="usb", microphone_fallback="backup,default-hw")
    assert desktop.source_candidates(config, excluded=["usb"]) == ["backup", "default-hw", "other"]


def test_media_probe_and_player_failures_do_not_block_capture_cleanup(monkeypatch):
    media = desktop.MediaSession()

    def unavailable(*_args):
        raise OSError("Media control unavailable")

    monkeypatch.setattr(desktop, "command", unavailable)
    media.pause(Config(pause_media="on"))
    assert media.paused == []
    calls = []

    def players(args):
        calls.append(args[-1])
        if args[-1] == "--list-all":
            return "broken\nplaying\n"
        if args[2] == "broken":
            return unavailable()
        return "Playing"

    monkeypatch.setattr(desktop, "command", players)
    media.pause(Config(pause_media="on"))
    assert media.paused == ["playing"]
    assert calls == ["--list-all", "status", "status", "pause"]
    monkeypatch.setattr(desktop, "command", unavailable)
    media.resume()
    assert media.paused == []


@pytest.mark.parametrize(
    ("settings", "stop", "preview", "expected"),
    [
        ({"sound_preset": "none"}, False, True, None),
        ({"sound_preset": "fluid_sfx_2"}, True, False, None),
        ({"sound_preset": "fluid_sfx_0"}, False, False, "FV_start_0.wav"),
        ({"sound_preset": "mouthkeys"}, True, False, "FV_end.wav"),
        ({"sound_preset": "custom", "sounds_enabled": "off"}, True, True, "custom.wav"),
        ({"sound_output": "default"}, True, True, "custom.wav"),
    ],
)
def test_capture_cues_select_only_local_files_and_never_call_live_playback(
    monkeypatch, tmp_path, settings, stop, preview, expected
):
    custom = tmp_path / "custom.wav"
    custom.write_bytes(b"not played")
    dispatched = []
    executed = []

    def thread(*, target, args, daemon):
        assert daemon
        dispatched.append(Path(args[1]).name)
        return SimpleNamespace(start=lambda: target(*args))

    def play(args, **options):
        executed.append((args, options))
        raise OSError("Player unavailable")

    monkeypatch.setattr(desktop, "threading", SimpleNamespace(Thread=thread))
    monkeypatch.setattr(desktop.subprocess, "run", play)
    output = {"sound_output": "test", **settings}
    config = Config(sound_stop_file=str(custom), sound_volume=50, **output)
    desktop.play_sound(config, stop=stop, preview=preview)
    assert dispatched == ([] if expected is None else [expected])
    if expected == "custom.wav":
        device = [] if config.sound_output == "default" else ["--device", "test"]
        assert executed == [
            (
                ["paplay", "--volume", "32768", *device, str(custom)],
                {"check": True, "capture_output": True, "timeout": 3},
            )
        ]


def test_missing_cue_file_does_not_attempt_playback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        desktop,
        "threading",
        SimpleNamespace(
            Thread=lambda *, target, args, daemon: SimpleNamespace(start=lambda: target(*args)),
        ),
    )
    monkeypatch.setattr(
        desktop.subprocess, "run", lambda *args, **kwargs: pytest.fail("Missing sound playback")
    )
    desktop.play_sound(Config(sound_file=str(tmp_path / "missing.wav")), preview=True)
