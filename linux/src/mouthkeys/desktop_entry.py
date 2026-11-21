"""Install the XDG application launcher and manage optional login startup."""

import os
import shutil
from importlib.util import find_spec
from pathlib import Path

from .config import Config
from .processes import python_command

_EXEC_ESCAPES = str.maketrans(
    {symbol: "\\" + symbol for symbol in ("\\", '"', "$", "`")} | {"%": "%%"}
)
_ENTRY_ESCAPES = str.maketrans({"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"})


def _quote_argument(argument: str) -> str:
    # Desktop entries decode the string escapes before parsing the Exec quoting.
    return '"' + argument.translate(_EXEC_ESCAPES).translate(_ENTRY_ESCAPES) + '"'


def _entry_command() -> list[str]:
    executable = shutil.which("mouthkeys")
    if executable is not None:
        return [str(Path(executable).resolve())]
    # When invoked as a module, argv[0] is not a launcher.
    return python_command("mouthkeys")


def _autostart_entry() -> Path:
    config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return config / "autostart" / "mouthkeys.desktop"


def _startup_command(config: Config) -> str:
    if config.show_window_at_login == "on":
        return "gui"
    if config.tray == "on" and find_spec("PySide6") is not None:
        return "tray"
    return "daemon"


def _write_entry(entry: Path, executable: list[str], command: str) -> None:
    invocation = " ".join([*map(_quote_argument, executable), command])
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=MouthKeys\n"
        "Comment=Local voice to text\nIcon=audio-input-microphone\n"
        f"Exec={invocation}\nTerminal=false\nCategories=AudioVideo;Audio;Utility;\n",
        encoding="utf-8",
    )


def install_desktop(*, autostart: bool = False) -> Path:
    data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    entry = data / "applications" / "mouthkeys.desktop"
    executable = _entry_command()
    _write_entry(entry, executable, "gui")
    if autostart:
        _write_entry(_autostart_entry(), executable, _startup_command(Config.load()))
    return entry


def apply_startup(config: Config) -> None:
    if config.launch_at_startup == "on":
        install_desktop(autostart=True)
    else:
        _autostart_entry().unlink(missing_ok=True)
