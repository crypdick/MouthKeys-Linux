import io
import json

import pytest

from mouthkeys import clipboard


@pytest.fixture
def process(monkeypatch):
    process = FakeProcess()

    def start(args, **options):
        assert args[1:] == ["-P", "-m", "mouthkeys.clipboard_owner"]
        assert options["stdin"] == clipboard.subprocess.PIPE
        assert options["stdout"] == clipboard.subprocess.PIPE
        assert options["stderr"] == clipboard.subprocess.DEVNULL
        return process

    monkeypatch.setattr(clipboard.subprocess, "Popen", start)
    monkeypatch.setattr(clipboard.select, "select", process.select)
    return process


class FakeProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"ready\nrestored\n")
        self.timeouts = []
        self.waits = []
        self.select_ready = True
        self.killed = False

    def select(self, readers, writers, errors, timeout):
        self.timeouts.append(timeout)
        return readers if self.select_ready else [], writers, errors

    def kill(self):
        self.killed = True

    def wait(self, *, timeout):
        self.waits.append(timeout)


@pytest.mark.parametrize("response", [None, b"not ready\n"])
def test_unready_owner_is_killed_and_reaped_before_fallback(process, response):
    process.stdout = io.BytesIO(response or b"")
    process.select_ready = response is not None
    with pytest.raises(RuntimeError, match="Clipboard snapshot unavailable"):
        clipboard.ClipboardPaste("temporary")
    assert process.killed
    assert process.waits == [2]
    assert process.timeouts == [3]
    assert process.stdin.closed
    assert process.stdout.closed


def test_keep_writes_protocol_command_without_waiting_for_restore(process):
    paste = clipboard.ClipboardPaste("temporary", delay_ms=250, exclude_history=True)
    assert json.loads(process.stdin.getvalue()) == {
        "text": "temporary",
        "delay_ms": 250,
        "exclude_history": True,
    }
    written = []
    process.stdin.write = written.append
    assert paste.finish(False)
    assert written == [b"keep\n"]
    assert process.timeouts == [3]
    assert process.stdin.closed
    assert process.stdout.closed


@pytest.mark.parametrize("response", [None, b"unexpected\n", b"skipped\n"])
def test_restore_waits_only_for_a_bounded_acknowledgement(process, response):
    process.stdout = io.BytesIO(b"ready\n" + (response or b""))
    paste = clipboard.ClipboardPaste("temporary", delay_ms=250)
    process.select_ready = response is not None
    assert paste.finish(True) is (response == b"skipped\n")
    assert process.timeouts == [3, 2.25]
    assert process.stdin.closed
    assert process.stdout.closed


def test_dead_owner_cannot_report_success_for_a_failed_write(process):
    paste = clipboard.ClipboardPaste("temporary")
    process.stdin.close()
    assert not paste.finish(True)
    assert process.stdout.closed


@pytest.mark.parametrize("missing", ["stdin", "stdout"])
def test_missing_protocol_pipe_reaps_owner_and_closes_the_other_end(process, missing):
    removed = getattr(process, missing)
    removed.close()
    setattr(process, missing, None)
    with pytest.raises(RuntimeError, match="both protocol pipes"):
        clipboard.ClipboardPaste("temporary")
    assert process.killed
    assert process.waits == [2]
    assert (process.stdout if missing == "stdin" else process.stdin).closed
