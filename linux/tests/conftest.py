import os
import time

import pytest


@pytest.fixture(scope="session")
def qt_application_owner():
    # The function-scoped fixture creates it only after XDG/offscreen isolation.
    applications = []
    yield applications
    if applications:
        import shiboken6

        application = applications[0]
        application.processEvents()
        shiboken6.delete(application)


@pytest.fixture(autouse=True)
def qt_object_lifetime(isolated_environment, qt_application_owner):
    try:
        import shiboken6
        from PySide6.QtCore import QCoreApplication, QEvent, QProcess, QThread, QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        yield
        return
    if not qt_application_owner:
        application = QApplication.instance() or QApplication([])
        application.setQuitOnLastWindowClosed(False)
        qt_application_owner.append(application)
    application = qt_application_owner[0]
    yield
    # close() hides widgets; it does not destroy them. Bound Qt signal callbacks
    # retain closed windows and their popup widgets until PySide's unsafe atexit
    # wrapper sweep. Destroy test-owned widgets while Qt and Python are both alive.
    widgets = list(application.topLevelWidgets())
    for widget in widgets:
        if not shiboken6.isValid(widget):
            continue
        for timer in widget.findChildren(QTimer):
            timer.stop()
    deadline = time.monotonic() + 2
    while True:
        application.processEvents()
        running = [
            obj
            for widget in widgets
            if shiboken6.isValid(widget)
            for obj in [*widget.findChildren(QThread), *widget.findChildren(QProcess)]
            if shiboken6.isValid(obj)
            and (
                obj.isRunning()
                if isinstance(obj, QThread)
                else obj.state() != QProcess.ProcessState.NotRunning
            )
        ]
        if not running or time.monotonic() >= deadline:
            break
        time.sleep(0.001)
    assert not running, "Qt test left running workers/processes; do not destroy them at shutdown"
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    for widget in widgets:
        if shiboken6.isValid(widget) and widget.parentWidget() is None:
            shiboken6.delete(widget)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not application.topLevelWidgets(), "Qt test left native windows alive after cleanup"


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
