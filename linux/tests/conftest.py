import os

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    for kind in ("CONFIG", "STATE", "CACHE", "DATA"):
        monkeypatch.setenv(f"XDG_{kind}_HOME", str(tmp_path / kind.lower()))
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # Never let tests accidentally use a real microphone/clipboard or synthesize keys.
    import subprocess

    original_run, original_popen = subprocess.run, subprocess.Popen
    forbidden = {
        "ffmpeg",
        "pw-record",
        "parec",
        "wl-copy",
        "xclip",
        "xdotool",
        "wtype",
        "ydotool",
        "xprop",
        "swaymsg",
        "hyprctl",
        "playerctl",
        "paplay",
    }

    def guarded(function):
        def call(args, *a, **kw):
            executable = args[0] if isinstance(args, list | tuple) else args.split()[0]
            assert os.path.basename(executable) not in forbidden, "Live desktop operation in test"
            return function(args, *a, **kw)

        return call

    monkeypatch.setattr(subprocess, "run", guarded(original_run))
    monkeypatch.setattr(subprocess, "Popen", guarded(original_popen))
