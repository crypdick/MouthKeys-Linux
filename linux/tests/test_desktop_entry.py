import shlex
import sys
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest

from mouthkeys import cli, desktop_entry
from mouthkeys.config import Config


def executable_line(entry):
    return next(
        line.removeprefix("Exec=")
        for line in entry.read_text().splitlines()
        if line.startswith("Exec=")
    )


@pytest.mark.parametrize(
    ("window", "tray", "qt", "command"),
    [
        ("on", "on", True, "gui"),
        ("off", "on", True, "tray"),
        ("off", "on", False, "daemon"),
        ("off", "off", True, "daemon"),
    ],
)
def test_cli_installs_and_removes_login_entry_without_launching(
    monkeypatch, tmp_path, capsys, window, tray, qt, command
):
    Config(show_window_at_login=window, tray=tray).save()
    available = Mock(return_value=object() if qt else None)
    monkeypatch.setattr(desktop_entry, "find_spec", available)
    assert cli.main(["install-desktop", "--autostart"]) == 0
    desktop = Path(capsys.readouterr().out.strip())
    startup = tmp_path / "config/autostart/mouthkeys.desktop"
    assert Config.load().launch_at_startup == "on"
    assert shlex.split(executable_line(desktop))[-1] == "gui"
    assert shlex.split(executable_line(startup))[-1] == command
    assert available.call_count == (window == "off" and tray == "on")

    assert cli.main(["config", "launch_at_startup", "off"]) == 0
    assert not startup.exists()
    assert desktop.exists()
    assert cli.main(["config", "launch_at_startup", "off"]) == 0
    assert not startup.exists()


def test_module_install_uses_current_python_without_resolving_venv_symlink(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop_entry.shutil, "which", Mock(return_value=None))
    python = tmp_path / "venv path" / "bin" / "python"
    monkeypatch.setattr(sys, "executable", str(python))
    entry = desktop_entry.install_desktop()
    assert shlex.split(executable_line(entry)) == [str(python), "-P", "-m", "mouthkeys", "gui"]


@pytest.mark.parametrize(
    ("filename", "encoded"),
    [
        ("with spaces", "with spaces"),
        ('with"quote', r'with\\"quote'),
        ("with\\backslash", r"with\\\\backslash"),
        ("with$dollar", r"with\\$dollar"),
        ("with`backtick", r"with\\`backtick"),
        ("with%field", "with%%field"),
        ("with\nnewline", r"with\nnewline"),
        ("with\ttab", r"with\ttab"),
    ],
)
def test_entry_escapes_exec_arguments_and_desktop_file_values(
    monkeypatch, tmp_path, filename, encoded
):
    monkeypatch.setattr(desktop_entry.shutil, "which", Mock(return_value=str(tmp_path / filename)))
    entry = desktop_entry.install_desktop()
    assert executable_line(entry) == f'"{tmp_path}/{encoded}" gui'
    assert entry.read_text().count("\n") == 8


def test_settings_share_desktop_entry_updater(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    gui = import_module("mouthkeys.gui")
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    startup = tmp_path / "config/autostart/mouthkeys.desktop"
    assert not startup.exists()
    field = window.fields["launch_at_startup"]
    field.setCurrentIndex(field.findData("on"))
    assert window.save()
    assert startup.exists()
    field.setCurrentIndex(field.findData("off"))
    assert window.save()
    assert not startup.exists()
    assert not window.isVisible()
