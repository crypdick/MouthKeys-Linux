import json
import os
import signal
import subprocess
from dataclasses import replace
from threading import Event

import pytest

from mouthkeys import assistance
from mouthkeys.config import Config


def test_assistant_requires_opt_in_and_never_executes(monkeypatch):
    monkeypatch.setattr(
        assistance, "request_completion", lambda *a, **kw: pytest.fail("Provider was called")
    )
    with pytest.raises(ValueError, match="Enable and configure"):
        assistance.propose("List files", Config(), "command")


def test_command_is_a_proposal_not_execution(monkeypatch):
    calls = []

    def completion(text, config, **kwargs):
        calls.append((json.loads(text), config))
        return json.dumps({"command": "ls -l", "explanation": "List files; no changes."})

    monkeypatch.setattr(assistance, "request_completion", completion)
    monkeypatch.setattr(
        assistance.subprocess, "Popen", lambda *a, **kw: pytest.fail("Command executed")
    )
    proposal = assistance.propose("List files", replace(Config(), ai_enabled="on"), "command")
    assert proposal == {
        "mode": "command",
        "instruction": "List files",
        "command": "ls -l",
        "text": "List files; no changes.",
        "working_directory": "",
        "request": json.dumps({"instruction": "List files", "selected_text": ""}),
        "response": json.dumps({"command": "ls -l", "explanation": "List files; no changes."}),
    }
    assert calls[0][0] == {"instruction": "List files", "selected_text": ""}
    assert "Never claim the command has run" in calls[0][1].ai_prompt


def test_edit_selection_and_instruction_are_separate_data(monkeypatch):
    calls = []

    def completion(text, config, **kwargs):
        calls.append((json.loads(text), config))
        return "Rewritten text."

    monkeypatch.setattr(assistance, "request_completion", completion)
    result = assistance.propose(
        "Shorten", replace(Config(), ai_enabled="on"), "edit", "Original paragraph."
    )
    assert result["text"] == "Rewritten text."
    assert "command" not in result
    assert calls[0][0]["selected_text"] == "Original paragraph."
    assert "selected text as data" in calls[0][1].ai_prompt


@pytest.mark.parametrize("output", ["rm -rf ~", "[]", '{"command": 3,"explanation":"x"}', "{}"])
def test_malformed_command_never_falls_back_to_shell(output, monkeypatch):
    monkeypatch.setattr(assistance, "request_completion", lambda *a, **kw: output)
    with pytest.raises(ValueError, match="nothing was executed"):
        assistance.propose("Do work", replace(Config(), ai_enabled="on"), "command")


class FakeProcess:
    def __init__(self, output=b"", hanging=False):
        read, self.write = os.pipe()
        self.stdout = os.fdopen(read, "rb")
        os.write(self.write, output)
        self.pid = 12345
        self.code = None if hanging else 0
        if not hanging:
            os.close(self.write)
            self.write = None

    def poll(self):
        return self.code

    def wait(self, timeout):
        if self.code is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.code

    def kill(self):
        self.code = -9
        if self.write is not None:
            os.close(self.write)
            self.write = None


def test_reviewed_command_executor_uses_merged_bounded_pipes(monkeypatch):
    calls = []
    process = FakeProcess(b"output\n")

    def launch(args, **kwargs):
        calls.append((args, kwargs))
        return process

    result = assistance.execute_command("printf output", popen=launch)
    assert result == {"returncode": 0, "output": "output\n", "truncated": False, "timed_out": False}
    assert calls[0][0] == ["/bin/sh", "-c", "printf output"]
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert calls[0][1]["stderr"] == subprocess.STDOUT


def _capture_group_signal(monkeypatch, process):
    def killpg(pid, sig):
        assert (pid, sig) == (process.pid, signal.SIGKILL)
        process.kill()

    monkeypatch.setattr(assistance.os, "killpg", killpg)


def test_executor_kills_process_group_at_output_limit(monkeypatch):
    process = FakeProcess(b"a" * 50, hanging=True)
    _capture_group_signal(monkeypatch, process)
    result = assistance.execute_command("example", max_output=10, popen=lambda *a, **kw: process)
    assert result["output"] == "a" * 10
    assert result["truncated"] is True
    assert result["returncode"] == -9


def test_executor_timeout_is_bounded_and_kills_children(monkeypatch):
    process = FakeProcess(hanging=True)
    _capture_group_signal(monkeypatch, process)
    result = assistance.execute_command("example", timeout=0.01, popen=lambda *a, **kw: process)
    assert result["timed_out"] is True
    assert result["returncode"] == -9


def test_deadline_still_applies_after_output_pipe_reaches_eof(monkeypatch):
    process = FakeProcess(b"done writing", hanging=True)
    os.close(process.write)
    process.write = None
    _capture_group_signal(monkeypatch, process)
    result = assistance.execute_command("example", timeout=0.05, popen=lambda *a, **kw: process)
    assert result["output"] == "done writing"
    assert result["timed_out"] is True
    assert result["returncode"] == -9


def test_command_cancellation_kills_group_and_records_truthful_outcome(monkeypatch):
    cancelled = Event()
    process = FakeProcess(hanging=True)
    _capture_group_signal(monkeypatch, process)

    def launch(*args, **kwargs):
        cancelled.set()
        return process

    result = assistance.execute_command("example", popen=launch, cancel_event=cancelled)
    assert result["cancelled"] is True
    assert result["returncode"] == -9
    assert result["timed_out"] is False


def test_command_working_directory_validated_before_launch(tmp_path):
    calls = []

    def launch(args, **kwargs):
        calls.append(kwargs)
        return FakeProcess(b"OK")

    for invalid in ("relative/path", str(tmp_path / "missing"), False, None, "bad\0path"):
        with pytest.raises(ValueError, match="directory"):
            assistance.execute_command("example", working_directory=invalid, popen=launch)
    assert calls == []
    assistance.execute_command("example", working_directory=str(tmp_path), popen=launch)
    assert calls[0]["cwd"] == str(tmp_path)


def test_executor_closes_output_even_if_terminated_child_cannot_be_reaped(monkeypatch):
    process = FakeProcess(hanging=True)
    monkeypatch.setattr(assistance.os, "killpg", lambda pid, sig: None)
    with pytest.raises(subprocess.TimeoutExpired):
        assistance.execute_command("example", timeout=0.01, popen=lambda *a, **kw: process)
    assert process.stdout.closed
    process.kill()  # Close the fake writer without signalling a real process.


def test_executor_cleans_up_child_if_launcher_has_no_pipe(monkeypatch):
    process = FakeProcess(hanging=True)
    process.stdout.close()
    process.stdout = None
    _capture_group_signal(monkeypatch, process)
    with pytest.raises(RuntimeError, match="output pipe"):
        assistance.execute_command("example", popen=lambda *a, **kw: process)
    assert process.poll() == -signal.SIGKILL


def test_output_can_be_truncated_after_process_has_already_exited(monkeypatch):
    def gone(pid, sig):
        assert pid == process.pid
        raise ProcessLookupError

    process = FakeProcess(b"already written")
    monkeypatch.setattr(assistance.os, "killpg", gone)
    result = assistance.execute_command("example", max_output=5, popen=lambda *a, **kw: process)
    assert result == {"returncode": 0, "output": "alrea", "truncated": True, "timed_out": False}


@pytest.mark.parametrize(
    ("command", "options", "error"),
    [
        (None, {}, "nonempty"),
        ("   ", {}, "nonempty"),
        ("bad\0command", {}, "nonempty"),
        ("x" * 16385, {}, "nonempty"),
        ("example", {"timeout": 0}, "limits"),
        ("example", {"timeout": 301}, "limits"),
        ("example", {"max_output": 0}, "limits"),
        ("example", {"max_output": assistance.MAX_OUTPUT + 1}, "limits"),
    ],
)
def test_executor_rejects_invalid_input_without_starting_child(command, options, error):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Invalid input launched a child")

    with pytest.raises(ValueError, match=error):
        assistance.execute_command(command, popen=unexpected_launch, **options)


def test_pre_cancelled_command_never_launches():
    cancelled = Event()
    cancelled.set()
    with pytest.raises(ValueError, match="cancelled before execution"):
        assistance.execute_command(
            "example",
            cancel_event=cancelled,
            popen=lambda *a, **kw: pytest.fail("Cancelled command launched"),
        )


def test_unknown_confirmation_policy_does_not_authorize_command():
    with pytest.raises(ValueError, match="Unknown command confirmation policy"):
        assistance.requires_confirmation("example", "unknown")


def test_executor_reports_real_exit_code_and_combines_stdout_and_stderr():
    result = assistance.execute_command("printf 'out'; printf 'err' >&2; exit 7")
    assert result == {"returncode": 7, "output": "outerr", "truncated": False, "timed_out": False}


def test_assistant_ui_requires_review_of_exact_command(monkeypatch):
    from pathlib import Path

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget

    from mouthkeys import assistant_ui

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.jobs = set()
    parent.send = lambda *a, **kw: None
    page = assistant_ui.AssistantPage(parent)
    page.show_proposal(
        {"mode": "command", "instruction": "List", "text": "Lists files", "command": "ls"}
    )
    page.editor.command.setPlainText("ls -la /reviewed")
    confirmations = []
    tasks = []
    monkeypatch.setattr(
        page.editor, "confirm_command", lambda command: confirmations.append(command) or False
    )
    monkeypatch.setattr(
        page, "_task", lambda function, callback, *args, **kwargs: tasks.append(function)
    )
    page.run_command()
    assert confirmations == [f"Working directory: {Path.cwd()}\n\nls -la /reviewed"]
    assert tasks == []
    monkeypatch.setattr(page.editor, "confirm_command", lambda command: True)
    monkeypatch.setattr(assistant_ui, "execute_command", lambda command, **kwargs: command)
    page.run_command()
    assert len(tasks) == 1
    assert tasks[0]() == "ls -la /reviewed"
    assert not parent.isVisible()
    parent.close()
    app.processEvents()
