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


def test_accessibility_context_ignores_passwords_and_reads_only_before_caret():
    from types import SimpleNamespace

    from mouthkeys.accessibility import focused_context

    requested = []
    text = SimpleNamespace(
        get_caret_offset=lambda: 500,
        get_text=lambda start, end: requested.append((start, end)) or "context",
    )
    node = SimpleNamespace(
        get_state_set=lambda: SimpleNamespace(contains=lambda _: True),
        get_role=lambda: "text",
        get_text_iface=lambda: text,
    )
    atspi = SimpleNamespace(
        set_timeout=lambda *args: None,
        get_desktop=lambda _: node,
        StateType=SimpleNamespace(FOCUSED=1, EDITABLE=2),
        Role=SimpleNamespace(PASSWORD_TEXT="password"),
    )
    assert focused_context(atspi) == "context"
    assert requested == [(300, 500)]
    node.get_role = lambda: "password"
    requested.clear()
    assert focused_context(atspi) is None
    assert requested == []


def test_accessibility_uses_isolated_distro_python_when_venv_has_no_gi(monkeypatch):
    from types import SimpleNamespace

    from mouthkeys import accessibility

    monkeypatch.setattr(accessibility.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(accessibility.Path, "is_file", lambda self: True)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout='"before cursor"')

    monkeypatch.setattr(accessibility.subprocess, "run", run)
    assert accessibility.read_context() == "before cursor"
    assert calls[0][0][:2] == ["/usr/bin/python3", "-I"]
    assert calls[0][0][2].endswith("/accessibility.py")
    assert calls[0][1]["timeout"] == 1


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
