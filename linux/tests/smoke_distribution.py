"""Opt-in built-wheel acceptance, using an isolated non-editable installation.

    uv venv --python 3.13 /tmp/mouthkeys-wheel-env
    uv pip install --offline --python /tmp/mouthkeys-wheel-env/bin/python \
        './linux/dist/mouthkeys_linux-0.2.1-py3-none-any.whl[gui]'
    timeout --kill-after=5s 45s /tmp/mouthkeys-wheel-env/bin/python -I \
        linux/tests/smoke_distribution.py \
        --expect-version 0.2.1 --report /tmp/mouthkeys-wheel-evidence.json

No package installation occurs in this script. Only temporary XDG directories
are used. No visible windows, microphone, clipboard, key synthesis or sounds.
Internet sockets are denied. A separately launched isolated daemon is shut down
before success is reported. The user's installed application is not touched.
The external timeout also bounds a native Qt constructor/import hang.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

if not sys.flags.isolated:
    raise SystemExit("Run this probe with python -I from an isolated wheel environment")

from mouthkeys.native_worker import isolate_network
from mouthkeys.service import request

temporary = tempfile.TemporaryDirectory(prefix="mouthkeys-wheel-probe-")
root = Path(temporary.name)
for kind in ("CONFIG", "STATE", "CACHE", "DATA", "RUNTIME"):
    path = root / kind.lower()
    path.mkdir(mode=0o700, exist_ok=True)
    os.environ[f"XDG_{kind}_HOME" if kind != "RUNTIME" else "XDG_RUNTIME_DIR"] = str(path)
for key in (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_SESSION_TYPE",
    "DBUS_SESSION_BUS_ADDRESS",
    "PYTHONPATH",
    "PYTHONHOME",
):
    os.environ.pop(key, None)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
isolate_network()
executable = Path(sys.executable).parent / "mouthkeys"
started = False
daemon = None
app = None
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--expect-version", required=True)
parser.add_argument("--report", type=Path)
args = parser.parse_args()


def cli(*args):
    assert args[0] in ("--version", "config"), "No capture/delivery actions allowed"
    result = subprocess.run(
        [str(executable), *args], check=True, capture_output=True, text=True, timeout=20, cwd=root
    )
    return result.stdout


def close_gui(application):
    if application is None:
        return
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent, QTimer

    for widget in application.topLevelWidgets():
        for timer in widget.findChildren(QTimer):
            timer.stop()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    for widget in application.topLevelWidgets():
        if shiboken6.isValid(widget) and widget.parentWidget() is None:
            shiboken6.delete(widget)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not application.topLevelWidgets()
    shiboken6.delete(application)


try:
    import mouthkeys

    assert Path(mouthkeys.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), (
        "Use an isolated non-editable wheel installation"
    )
    installation = json.loads(
        importlib.metadata.distribution("mouthkeys-linux").read_text("direct_url.json") or "{}"
    )
    assert not installation.get("dir_info", {}).get("editable"), (
        "Editable checkout is not a wheel test"
    )
    assert cli("--version").strip() == f"MouthKeys {args.expect_version}"
    # Own the process instead of relying on ensure_daemon's detached child.
    daemon = subprocess.Popen(
        [sys.executable, "-I", "-m", "mouthkeys", "daemon"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=root,
    )
    socket = root / "runtime/mouthkeys/control.sock"
    deadline = time.monotonic() + 5
    while not socket.exists() and daemon.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert socket.exists() and daemon.poll() is None, "Isolated daemon failed to start"
    started = True
    # Direct requests cannot auto-spawn an unowned replacement daemon on failure.
    status = request("status")
    assert status["version"] == args.expect_version and status["phase"] == "idle"
    cli("config", "spoken_send", "on")
    from mouthkeys.config import Config

    assert Config.load().spoken_send == "on"
    from mouthkeys.postprocess import BASE_DICTATION_PROMPT

    assert BASE_DICTATION_PROMPT.startswith("You are a voice-to-text dictation cleaner.")
    from dataclasses import asdict

    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    def forbidden_clipboard():
        raise AssertionError("Real clipboard access in wheel probe")

    QGuiApplication.clipboard = staticmethod(forbidden_clipboard)
    from mouthkeys import gui

    gui.Window.poll = lambda self: None
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    window = gui.Window()
    window.timer.stop()
    assert not window.isVisible()
    assert set(window.fields) == set(asdict(Config()))
    field = window.fields["spoken_send"]
    assert field.currentData() == "on"
    for module in (
        "overlay_assistant",
        "regional_ui",
        "setup_languages",
        "native_worker",
        "portal_shortcuts",
        "atspi_helper",
        "credentials",
        "data_lock",
        "dictionary_transfer",
    ):
        __import__("mouthkeys." + module)
    assets = Path(mouthkeys.__file__).parent / "assets/sounds"
    assert all(
        (assets / (name + ".wav")).is_file()
        for name in ("FV_start_0", "FV_end_0", "FV_start", "FV_start_2", "sfx_3", "sfx_4")
    )
    count = len(window.fields)
    window.close()
    app.processEvents()
    close_gui(app)
    app = None
    evidence = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "result": "passed",
        "version": args.expect_version,
        "distribution": "built non-editable wheel",
        "config_fields_in_settings": count,
        "checks": [
            "installed CLI version",
            "owned isolated daemon start/status/shutdown",
            "CLI-to-GUI configuration roundtrip",
            "all configuration fields exposed",
            "hidden offscreen Settings construction",
            "new worker/UI imports",
            "packaged base prompt and all sound presets",
        ],
        "network": "AF_INET/AF_INET6 denied by seccomp during the probe",
        "desktop": "no display, microphone, clipboard, keys, sounds or visible windows",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mouthkeys-linux", "PySide6-Essentials", "ctranslate2", "faster-whisper")
        },
    }
finally:
    try:
        close_gui(app)
    finally:
        try:
            if started:
                request("shutdown")
                daemon.wait(timeout=3)
                assert not socket.exists(), "Isolated daemon did not remove its socket"
        finally:
            try:
                if daemon is not None and daemon.poll() is None:
                    daemon.terminate()
                    try:
                        daemon.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        daemon.kill()
                        daemon.wait(timeout=3)
            finally:
                temporary.cleanup()

if args.report:
    from mouthkeys.operations import write_private

    write_private(args.report, json.dumps(evidence, indent=2).encode())
print(json.dumps(evidence))
