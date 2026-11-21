import builtins
import io
import json
import runpy
import sys
from types import SimpleNamespace

import pytest
from PySide6 import QtCore, QtGui

from mouthkeys import clipboard


class FakeClipboard:
    def __init__(self):
        self.current = None
        self.owned = False
        self.claim_allowed = True
        self.changed = SimpleNamespace(connect=self.connect)
        self.on_change = None

    def connect(self, callback):
        self.on_change = callback

    def mimeData(self):  # noqa: N802 - Qt interface
        return self.current

    def ownsClipboard(self):  # noqa: N802 - Qt interface
        return self.owned

    def setMimeData(self, mime):  # noqa: N802 - Qt interface
        self.current = mime
        self.owned = self.claim_allowed

    def clear(self):
        self.current = None
        self.owned = False

    def user_copy(self, text):
        self.current = clipboard.temporary_mime(text, False)
        self.owned = False
        self.on_change()


class FakeStdin(io.BytesIO):
    @property
    def buffer(self):
        return self

    def fileno(self):
        return 0


class FakeOwner:
    def __init__(self, monkeypatch, action=b"restore\n"):
        self.selection = FakeClipboard()
        self.quit_requested = False
        self.on_read = None
        self.on_delay = None
        self.before_read = lambda: None
        self.after_read = lambda: None
        self.after_delay = lambda: None
        line = json.dumps({"text": "new transcript", "delay_ms": 250}).encode() + b"\n"
        monkeypatch.setattr(sys, "stdin", FakeStdin(line + action))
        monkeypatch.setattr(QtGui, "QGuiApplication", lambda args: self)
        monkeypatch.setattr(QtCore, "QSocketNotifier", self.notifier())
        monkeypatch.setattr(QtCore, "QTimer", SimpleNamespace(singleShot=self.single_shot))

    def notifier(self):
        owner = self

        class Notifier:
            Type = SimpleNamespace(Read="read")

            def __init__(self, fd, kind):
                assert (fd, kind) == (0, self.Type.Read)
                self.activated = SimpleNamespace(connect=self.connect)

            def connect(self, callback):
                owner.on_read = callback

            def setEnabled(self, enabled):  # noqa: N802 - Qt interface
                assert not enabled

        return Notifier

    def clipboard(self):
        return self.selection

    def processEvents(self):  # noqa: N802 - Qt interface
        pass

    def single_shot(self, delay, callback):
        assert delay == 250
        self.on_delay = callback

    def quit(self):
        self.quit_requested = True

    def exec(self):
        assert self.selection.current.text() == "new transcript"
        self.before_read()
        self.on_read()
        self.after_read()
        if self.on_delay is not None and not self.quit_requested:
            self.on_delay()
            self.after_delay()
        assert self.quit_requested
        return 0


def test_owner_restores_all_formats_then_leaves_when_user_copies(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch)
    original = QtCore.QMimeData()
    original.setText("old text")
    original.setData("application/x-test", b"\x00\x01")
    owner.selection.current = original

    def after_restore():
        assert not owner.quit_requested
        assert clipboard.capture_formats(owner.selection.current) == {
            "text/plain": b"old text",
            "application/x-test": b"\x00\x01",
        }
        owner.selection.user_copy("user text")

    owner.after_delay = after_restore
    assert clipboard.main() == 0
    assert capsys.readouterr().out.splitlines() == ["ready", "restored"]
    assert owner.selection.current.text() == "user text"


@pytest.mark.parametrize("loss", ["before command", "during delay"])
def test_owner_never_overwrites_a_newer_copy(monkeypatch, capsys, loss):
    owner = FakeOwner(monkeypatch)
    if loss == "before command":
        owner.before_read = lambda: owner.selection.user_copy("user text")
    else:
        owner.after_read = lambda: owner.selection.user_copy("user text")
    assert clipboard.main() == 0
    assert capsys.readouterr().out.splitlines() == ["ready", "skipped"]
    assert owner.selection.current.text() == "user text"


def test_owner_clears_temporary_text_when_original_selection_was_empty(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch)
    assert clipboard.main() == 0
    assert capsys.readouterr().out.splitlines() == ["ready", "skipped"]
    assert owner.selection.current is None


def test_keep_preserves_temporary_text_until_newer_copy(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch, action=b"keep\n")

    def after_keep():
        assert not owner.quit_requested
        assert owner.selection.current.text() == "new transcript"
        owner.selection.user_copy("user text")

    owner.after_read = after_keep
    assert clipboard.main() == 0
    assert capsys.readouterr().out.splitlines() == ["ready"]
    assert owner.selection.current.text() == "user text"


@pytest.mark.parametrize(
    ("line", "limit"),
    [
        (b"too many bytes\n", 10),
        (b'{"text":"new transcript","delay_ms":true}\n', None),
        (b'{"text":"new transcript","delay_ms":5001}\n', None),
    ],
)
def test_owner_rejects_invalid_requests_before_reading_clipboard(monkeypatch, line, limit):
    if limit is not None:
        monkeypatch.setattr(clipboard, "MAX_BYTES", limit)
    monkeypatch.setattr(sys, "stdin", FakeStdin(line))
    monkeypatch.setattr(
        QtGui, "QGuiApplication", lambda args: pytest.fail("Unexpected desktop access")
    )
    assert clipboard.main() == 1


def test_owner_exits_if_it_cannot_claim_selection(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch)
    owner.selection.claim_allowed = False
    assert clipboard.main() == 1
    assert capsys.readouterr().out == ""


def test_owner_preserves_original_if_snapshot_cannot_fit(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch)
    original = clipboard.temporary_mime("x" * 300, False)
    owner.selection.current = original
    monkeypatch.setattr(clipboard, "MAX_BYTES", 200)
    assert clipboard.main() == 1
    assert capsys.readouterr().out == ""
    assert owner.selection.current is original


def test_failed_acknowledgement_does_not_discard_restored_ownership(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch)
    owner.selection.current = clipboard.temporary_mime("original", False)
    real_print = builtins.print

    def close_ack_pipe(message, **kwargs):
        if message == "restored":
            raise BrokenPipeError
        real_print(message, **kwargs)

    def after_delay():
        assert owner.selection.current.text() == "original"
        assert owner.selection.owned
        owner.selection.user_copy("new user copy")

    monkeypatch.setattr(builtins, "print", close_ack_pipe)
    owner.after_delay = after_delay
    assert clipboard.main() == 0
    assert capsys.readouterr().out.splitlines() == ["ready"]
    assert owner.selection.current.text() == "new user copy"


def test_keep_exits_immediately_if_clipboard_changed_before_command(monkeypatch, capsys):
    owner = FakeOwner(monkeypatch, action=b"keep\n")
    owner.before_read = lambda: owner.selection.user_copy("new user copy")
    assert clipboard.main() == 0
    assert capsys.readouterr().out.splitlines() == ["ready"]
    assert owner.selection.current.text() == "new user copy"


def test_module_entry_point_exits_with_helper_status(monkeypatch):
    monkeypatch.setattr(sys, "stdin", FakeStdin(b'{"delay_ms":99}\n'))
    monkeypatch.delitem(sys.modules, "mouthkeys.clipboard")
    with pytest.raises(SystemExit) as status:
        runpy.run_module("mouthkeys.clipboard", run_name="__main__")
    assert status.value.code == 1
