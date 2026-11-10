from dataclasses import replace

from mouthkeys import desktop
from mouthkeys.config import Config


def test_microphone_priority_exclusions_and_fallback(monkeypatch):
    monkeypatch.setattr(desktop, "source_names", lambda: ["bad", "backup", "usb"])
    monkeypatch.setattr(desktop, "command", lambda args: "bad\n")
    config = Config(
        microphone_preferred="missing,usb", microphone_fallback="backup", microphone_never="bad"
    )
    assert desktop.choose_source(config) == "usb"
    assert desktop.choose_source(replace(config, microphone_preferred="missing")) == "backup"


def test_microphone_never_applies_to_explicit_selection():
    import pytest

    with pytest.raises(ValueError, match="Never"):
        desktop.choose_source(Config(audio_source="bad", microphone_never="bad"))


def test_media_resumes_only_players_it_paused_and_only_if_still_paused(monkeypatch):
    statuses = {"playing": "Playing", "idle": "Paused", "changed": "Playing"}
    calls = []

    def command(args):
        calls.append(args)
        if args[-1] == "--list-all":
            return "playing\nidle\nchanged\n"
        player = args[2]
        if args[-1] == "status":
            return statuses[player]
        if args[-1] == "pause":
            statuses[player] = "Paused"
        return ""

    monkeypatch.setattr(desktop, "command", command)
    media = desktop.MediaSession()
    media.pause(Config(pause_media="on"))
    assert media.paused == ["playing", "changed"]
    statuses["changed"] = "Playing"  # User resumed it; MouthKeys must not interfere.
    media.resume()
    assert [args[2] for args in calls if args[-1] == "play"] == ["playing"]
    assert not media.paused


def test_sound_disabled_never_spawns_process(monkeypatch):
    monkeypatch.setattr(
        desktop.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected sound")),
    )
    desktop.play_sound(Config())


def test_configurable_overlay_scales_without_showing_window():
    from PySide6.QtWidgets import QApplication

    from mouthkeys.overlay import Overlay, live_counters

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
    panel.timer.stop()
    panel.apply_settings(Config(overlay_size=150, theme="light", overlay_border="off"))
    assert (panel.width(), panel.height()) == (636, 219)
    assert panel.surface == "#ffffff"
    assert live_counters("hello café world", 6) == (3, 30)
    assert live_counters("hello", 0) == (1, 0)
    assert not panel.isVisible()
    panel.close()
    app.processEvents()


def test_overlay_history_popup_copies_selected_entry_without_pasting(monkeypatch):
    from PySide6.QtWidgets import QApplication, QMenu

    from mouthkeys.overlay import Overlay

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
    panel.timer.stop()
    calls = []
    monkeypatch.setattr(QMenu, "popup", lambda self, point: None)
    monkeypatch.setattr(panel, "send", lambda action, **kw: calls.append((action, kw)))
    panel.show_history({"result": [{"id": "sample", "text": "hello world"}]})
    entry = panel.history_menu.actions()[0].menu()
    entry.actions()[0].trigger()
    assert calls == [("copy", {"item_id": "sample"})]
    assert not panel.isVisible()
    panel.close()
    app.processEvents()


def test_clipboard_snapshot_restores_every_format_but_not_new_user_copy(monkeypatch):
    from types import SimpleNamespace

    import pytest
    from PySide6.QtCore import QByteArray, QMimeData

    from mouthkeys import clipboard

    mime = QMimeData()
    mime.setData("text/plain", QByteArray(b"hello"))
    mime.setData("image/png", QByteArray(b"\x89PNG\r\n"))
    saved = clipboard.capture_formats(mime)
    assert saved == {"text/plain": b"hello", "image/png": b"\x89PNG\r\n"}
    restored = []
    fake = SimpleNamespace(ownsClipboard=lambda: True, setMimeData=restored.append)
    assert clipboard.restore_formats(fake, saved)
    assert clipboard.capture_formats(restored[0]) == saved
    fake.ownsClipboard = lambda: False
    assert not clipboard.restore_formats(fake, saved)
    assert len(restored) == 1
    monkeypatch.setattr(clipboard, "MAX_BYTES", 2)
    with pytest.raises(ValueError, match="32 MiB"):
        clipboard.capture_formats(mime)


def test_context_uses_only_foreground_snapshot_and_skips_unknown_focus(monkeypatch):
    from mouthkeys import accessibility

    monkeypatch.setattr(
        accessibility, "read_snapshot", lambda: {"foreground": True, "context": "before cursor"}
    )
    assert accessibility.read_context() == "before cursor"
    monkeypatch.setattr(
        accessibility, "read_snapshot", lambda: {"foreground": False, "context": "inactive window"}
    )
    assert accessibility.read_context() is None


def test_accessibility_uses_isolated_distro_python_when_venv_has_no_gi(monkeypatch):
    from types import SimpleNamespace

    from mouthkeys import accessibility

    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(accessibility.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(accessibility.Path, "is_file", lambda self: True)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="null")

    monkeypatch.setattr(accessibility.subprocess, "run", run)
    assert accessibility.read_snapshot() is None
    assert calls[0][0][:2] == ["/usr/bin/python3", "-I"]
    assert calls[0][0][2].endswith("/atspi_helper.py")
    assert calls[0][1]["timeout"] == 2


def test_tray_alerts_are_opt_in_deduplicated_and_do_not_claim_verified_paste(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mouthkeys.tray import Tray

    app = QApplication.instance() or QApplication([])
    tray = Tray()
    tray.timer.stop()
    notices = []
    monkeypatch.setattr(tray, "showMessage", lambda *args: notices.append(args))
    initial = {"phase": "idle", "microphone": "default", "id": None}
    tray.notify_status(initial)
    Config(show_microphone_alerts="on", show_paste_alerts="on", delivery="x11").save()
    done = {
        "phase": "idle",
        "microphone": "usb",
        "id": "a",
        "preview": "hello",
        "message": "Insertion requested — transcript saved in history",
    }
    tray.notify_status(done)
    tray.notify_status(done)
    assert len(notices) == 2
    assert notices[0] == ("Microphone changed", "usb")
    assert "cannot be verified" in notices[1][1]
    Config().save()
    tray.notify_status({**done, "id": "b", "microphone": "other"})
    assert len(notices) == 2
    tray.deleteLater()
    app.processEvents()


def test_tray_cannot_quit_during_file_transcription():
    from types import SimpleNamespace

    from mouthkeys.tray import Tray

    tips = []
    fake = SimpleNamespace(
        window=SimpleNamespace(jobs=set(), download=None, file_process=object()),
        quitting=False,
        setToolTip=tips.append,
    )
    Tray.request_quit(fake)
    assert not fake.quitting
    assert tips == ["Wait for current settings operation before quitting"]


def test_desktop_default_source_is_resolved_for_actual_microphone_label(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(desktop, "command", lambda args: "alsa_input.actual_usb\n")
    assert desktop.choose_source(Config()) == "alsa_input.actual_usb"


def test_preset_start_stop_cues_use_bundled_pcm_files_without_playback(monkeypatch):
    from mouthkeys import desktop

    calls = []

    class Thread:
        def __init__(self, *, target, args, daemon):
            calls.append(args)

        def start(self):
            pass  # Never play sounds in tests.

    monkeypatch.setattr(desktop.threading, "Thread", Thread)
    config = Config(sound_preset="mouthkeys")
    desktop.play_sound(config, preview=True)
    desktop.play_sound(config, stop=True)
    assert calls[0][1].name == "FV_start.wav"
    assert calls[1][1].name == "FV_end.wav"
    assert calls[0][1].read_bytes()[:4] == b"RIFF"


def test_temporary_clipboard_history_hint_does_not_modify_normal_copy_formats():
    from mouthkeys.clipboard import capture_formats, temporary_mime

    assert capture_formats(temporary_mime("hello", False)) == {"text/plain": b"hello"}
    hidden = capture_formats(temporary_mime("hello", True))
    assert hidden["x-kde-passwordManagerHint"] == b"secret"
    assert hidden["text/plain"] == b"hello"


def test_queued_clipboard_transactions_wait_for_restoration_ack_before_next_snapshot(monkeypatch):
    import json

    from mouthkeys import clipboard

    state = {"clipboard": "original"}
    snapshots = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.original = state["clipboard"]
            snapshots.append(self.original)
            self.ready = False
            self.stdin = self.stdout = self

        def write(self, value):
            if value.startswith(b"{"):
                state["clipboard"] = json.loads(value)["text"]

        def flush(self):
            pass

        def readline(self, size):
            if not self.ready:
                self.ready = True
                return b"ready\n"
            state["clipboard"] = self.original
            return b"restored\n"

        def close(self):
            pass

    monkeypatch.setattr(clipboard.subprocess, "Popen", Process)
    monkeypatch.setattr(clipboard.select, "select", lambda readers, *args: (readers, [], []))
    first = clipboard.ClipboardPaste("dictation A")
    assert first.finish(True)
    second = clipboard.ClipboardPaste("dictation B")
    assert second.finish(True)
    assert snapshots == ["original", "original"]
    assert state["clipboard"] == "original"
