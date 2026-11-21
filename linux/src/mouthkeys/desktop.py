"""Optional Linux desktop affordances; no global grabs or privileged services."""

import os
import re
import subprocess
import threading
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Protocol, runtime_checkable

TERMINALS = (
    "ghostty",
    "c11",
    "kitty",
    "alacritty",
    "wezterm",
    "xterm",
    "urxvt",
    "gnome-terminal",
    "org.gnome.terminal",
    "gnome terminal",
    "konsole",
    "tilix",
    "terminator",
    "foot",
    "st-256color",
    "xfce4-terminal",
    "qterminal",
    "lxterminal",
)


def terminal_name(value: str) -> bool:
    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", value.lower())
        for name in TERMINALS
    )


def command(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=2).stdout  # noqa: S603


def source_names() -> list[str]:
    return [
        line.split()[1]
        for line in command(["pactl", "list", "short", "sources"]).splitlines()
        if len(line.split()) >= 2
    ]


def source_list(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


@runtime_checkable
class SourceSettings(Protocol):
    @property
    def audio_source(self) -> str: ...
    @property
    def microphone_never(self) -> str: ...
    @property
    def microphone_preferred(self) -> str: ...
    @property
    def microphone_fallback(self) -> str: ...


def choose_source(config: SourceSettings) -> str:
    """Explicit source wins, except an explicitly excluded source is never opened."""
    never = set(source_list(config.microphone_never))
    if config.audio_source != "default":
        if config.audio_source in never:
            raise ValueError("Selected microphone is in Never list")
        return config.audio_source
    preferred = source_list(config.microphone_preferred)
    fallback = source_list(config.microphone_fallback)
    if not (preferred or fallback or never):
        source = "default"
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            with suppress(OSError, subprocess.SubprocessError):
                source = command(["pactl", "get-default-source"]).strip() or source
        return source
    available = source_names()
    for name in preferred:
        if name in available and name not in never:
            return name
    default = command(["pactl", "get-default-source"]).strip()
    if default in available and default not in never and default not in fallback:
        return default
    for name in fallback:
        if name in available and name not in never:
            return name
    for name in available:
        if name not in never and not name.endswith(".monitor"):
            return name
    raise ValueError("No available microphone outside Never list")


def source_candidates(config: SourceSettings, *, excluded: Iterable[str] = ()) -> list[str]:
    """Resolve ranked, non-monitor alternate inputs after a capture failure."""
    unavailable = set(excluded) | set(source_list(config.microphone_never))
    selected = choose_source(config)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not (
        config.microphone_preferred or config.microphone_fallback
    ):
        return [] if selected in unavailable else [selected]
    try:
        available = source_names()
        default = command(["pactl", "get-default-source"]).strip()
    except (OSError, subprocess.SubprocessError):
        available, default = [], ""
    fallback = source_list(config.microphone_fallback)
    ordered = [selected, *source_list(config.microphone_preferred)]
    if default not in fallback:
        ordered.append(default)
    ordered.extend(fallback)
    ordered.extend(available)
    result = []
    for source in ordered:
        if (
            source
            and source not in unavailable
            and source not in result
            and (source == selected or source in available)
            and (not source.endswith(".monitor") or source == selected)
        ):
            result.append(source)
    return result


@runtime_checkable
class MediaSettings(Protocol):
    @property
    def pause_media(self) -> str: ...


@runtime_checkable
class SoundSettings(Protocol):
    @property
    def sounds_enabled(self) -> str: ...
    @property
    def sound_preset(self) -> str: ...
    @property
    def sound_file(self) -> str: ...
    @property
    def sound_stop_file(self) -> str: ...
    @property
    def sound_volume(self) -> int: ...
    @property
    def sound_output(self) -> str: ...


class MediaSession:
    """Resume only players this recording actually paused, and only if still paused."""

    def __init__(self) -> None:
        self.paused: list[str] = []

    def pause(self, config: MediaSettings) -> None:
        if config.pause_media != "on":
            return
        try:
            players = command(["playerctl", "--list-all"]).splitlines()
        except (OSError, subprocess.SubprocessError):
            return
        for player in players:
            try:
                if command(["playerctl", "--player", player, "status"]).strip() == "Playing":
                    command(["playerctl", "--player", player, "pause"])
                    self.paused.append(player)
            except (OSError, subprocess.SubprocessError):
                continue

    def resume(self) -> None:
        players, self.paused = self.paused, []
        for player in players:
            try:
                if command(["playerctl", "--player", player, "status"]).strip() == "Paused":
                    command(["playerctl", "--player", player, "play"])
            except (OSError, subprocess.SubprocessError):
                continue


def play_sound(config: SoundSettings, *, stop: bool = False, preview: bool = False) -> None:
    """Opt-in local file only. Never download cues or block microphone lifecycle."""
    preset = config.sound_preset
    if not preview and config.sounds_enabled != "on":
        return
    if preset == "none":
        return
    path: str | Path | None
    if preset == "mouthkeys":
        path = Path(__file__).parent / "assets/sounds" / ("FV_end.wav" if stop else "FV_start.wav")
    elif preset.startswith("fluid_sfx_"):
        sounds = {
            "fluid_sfx_0": ("FV_start_0", "FV_end_0"),
            "fluid_sfx_1": ("FV_start", None),
            "fluid_sfx_2": ("FV_start_2", None),
            "fluid_sfx_3": ("sfx_3", None),
            "fluid_sfx_4": ("sfx_4", None),
        }
        name = sounds.get(preset, (None, None))[bool(stop)]
        path = Path(__file__).parent / "assets/sounds" / f"{name}.wav" if name else None
    else:
        path = config.sound_stop_file if stop else config.sound_file
    if not path:
        return
    threading.Thread(target=_play_sound, args=(config, path), daemon=True).start()


def _play_sound(config: SoundSettings, path: str | Path) -> None:
    path = Path(path).expanduser()
    if not path.is_file():
        return
    args = ["paplay", "--volume", str(round(config.sound_volume / 100 * 65536))]
    if config.sound_output != "default":
        args += ["--device", config.sound_output]
    # An unavailable cue must never discard a recording.
    with suppress(OSError, subprocess.SubprocessError):
        subprocess.run([*args, str(path)], check=True, capture_output=True, timeout=3)  # noqa: S603
