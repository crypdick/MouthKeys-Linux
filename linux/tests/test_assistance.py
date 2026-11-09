import json
import os
import subprocess
from dataclasses import replace

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


def test_executor_kills_process_group_at_output_limit(monkeypatch):
    process = FakeProcess(b"a" * 50, hanging=True)
    monkeypatch.setattr(assistance, "_kill_group", lambda p: p.kill())
    result = assistance.execute_command("example", max_output=10, popen=lambda *a, **kw: process)
    assert result["output"] == "a" * 10
    assert result["truncated"] is True
    assert result["returncode"] == -9


def test_executor_timeout_is_bounded_and_kills_children(monkeypatch):
    process = FakeProcess(hanging=True)
    monkeypatch.setattr(assistance, "_kill_group", lambda p: p.kill())
    result = assistance.execute_command("example", timeout=0.01, popen=lambda *a, **kw: process)
    assert result["timed_out"] is True
    assert result["returncode"] == -9


def test_assistant_ui_requires_review_of_exact_command(monkeypatch):
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
    page.command.setPlainText("ls -la /reviewed")
    confirmations = []
    tasks = []
    monkeypatch.setattr(
        page, "confirm_command", lambda command: confirmations.append(command) or False
    )
    monkeypatch.setattr(page, "_task", lambda function, callback: tasks.append(function))
    page.run_command()
    assert confirmations == ["ls -la /reviewed"]
    assert tasks == []
    monkeypatch.setattr(page, "confirm_command", lambda command: True)
    monkeypatch.setattr(assistant_ui, "execute_command", lambda command: command)
    page.run_command()
    assert len(tasks) == 1
    assert tasks[0]() == "ls -la /reviewed"
    assert not parent.isVisible()
    parent.close()
    app.processEvents()
