"""Optional Linux desktop affordances; no global grabs or privileged services."""

import os
import subprocess
import threading
from pathlib import Path


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=2).stdout


def source_names():
    return [
        line.split()[1]
        for line in command(["pactl", "list", "short", "sources"]).splitlines()
        if len(line.split()) >= 2
    ]


def source_list(value):
    return [name.strip() for name in value.split(",") if name.strip()]


def choose_source(config):
    """Explicit source wins, except an explicitly excluded source is never opened."""
    never = set(source_list(config.microphone_never))
    if config.audio_source != "default":
        if config.audio_source in never:
            raise ValueError("Selected microphone is in Never list")
        return config.audio_source
    preferred = source_list(config.microphone_preferred)
    fallback = source_list(config.microphone_fallback)
    if not (preferred or fallback or never):
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            try:
                return command(["pactl", "get-default-source"]).strip() or "default"
            except (OSError, subprocess.SubprocessError):
                pass
        return "default"
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


def source_candidates(config, *, excluded=()):
    """Resolve ranked, non-monitor alternate inputs after a capture failure."""
    excluded = set(excluded) | set(source_list(config.microphone_never))
    selected = choose_source(config)
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not (
        config.microphone_preferred or config.microphone_fallback
    ):
        return [] if selected in excluded else [selected]
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
            and source not in excluded
            and source not in result
            and (source == selected or source in available)
            and (not source.endswith(".monitor") or source == selected)
        ):
            result.append(source)
    return result


class MediaSession:
    """Resume only players this recording actually paused, and only if still paused."""

    def __init__(self):
        self.paused = []

    def pause(self, config):
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

    def resume(self):
        players, self.paused = self.paused, []
        for player in players:
            try:
                if command(["playerctl", "--player", player, "status"]).strip() == "Paused":
                    command(["playerctl", "--player", player, "play"])
            except (OSError, subprocess.SubprocessError):
                continue


def play_sound(config, stop=False, preview=False):
    """Opt-in local file only. Never download cues or block microphone lifecycle."""
    preset = getattr(config, "sound_preset", "custom")
    if not preview and getattr(config, "sounds_enabled", "on") != "on":
        return
    if preset == "none":
        return
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
        path = getattr(config, "sound_stop_file", "") if stop else config.sound_file
    if not path:
        return
    threading.Thread(target=_play_sound, args=(config, path), daemon=True).start()


def _play_sound(config, path=None):
    path = Path(path or config.sound_file).expanduser()
    if not path.is_file():
        return
    args = ["paplay", "--volume", str(round(config.sound_volume / 100 * 65536))]
    if config.sound_output != "default":
        args += ["--device", config.sound_output]
    try:
        subprocess.run(args + [str(path)], check=True, capture_output=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        pass  # An unavailable cue must never discard a recording.
