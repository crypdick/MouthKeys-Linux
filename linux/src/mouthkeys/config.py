import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def xdg(kind: str, default: str) -> Path:
    return Path(os.environ.get(f"XDG_{kind}_HOME", str(Path.home() / default))) / "mouthkeys"


def state_dir() -> Path:
    return private_dir(xdg("STATE", ".local/state"))


def cache_dir() -> Path:
    return private_dir(xdg("CACHE", ".cache"))


def runtime_dir() -> Path:
    root = os.environ.get("XDG_RUNTIME_DIR")
    return private_dir(Path(root) / "mouthkeys" if root else state_dir() / "run")


def config_path() -> Path:
    return xdg("CONFIG", ".config") / "config.json"


@dataclass(frozen=True)
class Config:
    model: str = "base.en"
    language: str = "en"
    device: str = "cpu"
    compute_type: str = "default"
    audio_source: str = "default"
    delivery: str = "clipboard"
    paste_backend: str = "auto"
    paste_shortcut: str = "ctrl+shift+v"
    max_seconds: int = 300
    audio_budget_mb: int = 4000
    recording_panel: str = "on"
    engine: str = "whisper"
    remove_filler_words: str = "off"
    filler_words: str = "um,uh,er"
    lowercase_first: str = "off"
    remove_trailing_period: str = "off"
    literal_formatting: str = "off"
    question_mark_shortcut: str = "off"
    spoken_formatting: str = "off"
    literal_prefix: str = "literal"
    spoken_send: str = "off"
    send_after_pause: str = "off"
    spoken_send_phrase: str = "send it"
    spoken_send_key: str = "return"
    ai_enabled: str = "off"
    ai_endpoint: str = "http://localhost:11434/v1/chat/completions"
    ai_model: str = ""
    ai_key_env: str = "MOUTHKEYS_AI_API_KEY"
    ai_prompt: str = "Clean up dictated text without changing meaning. Return only corrected text."
    save_history: str = "on"
    save_audio: str = "off"
    live_preview: str = "on"
    skip_silent_recordings: str = "off"
    theme: str = "system"
    overlay_position: str = "bottom"
    overlay_size: int = 100
    overlay_bottom_offset: int = 40
    preview_chars: int = 240
    visualizer_sensitivity: int = 100
    overlay_border: str = "on"
    tray: str = "on"
    pause_media: str = "off"
    sound_file: str = ""
    sound_volume: int = 50
    sound_output: str = "default"
    microphone_preferred: str = ""
    microphone_fallback: str = ""
    microphone_never: str = ""
    insertion_mode: str = "clipboard"
    restore_clipboard: str = "off"
    copy_to_clipboard: str = "on"
    typing_wpm: int = 40
    weekends_break_streak: str = "on"
    api_enabled: str = "off"
    api_port: int = 47733
    api_token_env: str = "MOUTHKEYS_API_TOKEN"
    file_speaker_labels: str = "off"
    file_speaker_count: int = 0
    ai_provider: str = "compatible"
    queue_limit: int = 8
    ai_profile: str = ""
    ai_max_tokens: int = 4096
    ai_timeout: int = 30
    smart_capitalization: str = "off"
    space_between_dictations: str = "off"
    show_microphone_alerts: str = "off"
    show_paste_alerts: str = "off"
    launch_at_startup: str = "off"
    show_window_at_login: str = "off"
    clipboard_restore_delay_ms: int = 500

    def validate(self):
        for key, value in asdict(self).items():
            if key in RANGES:
                low, high = RANGES[key]
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"{key} must be between {low} and {high}")
            elif not isinstance(value, str) or (key not in OPTIONAL and not value.strip()):
                raise ValueError(
                    f"{key} must be a {'string' if key in OPTIONAL else 'nonempty string'}"
                )
            if key in CHOICES and value not in CHOICES[key]:
                raise ValueError(f"{key} must be {', '.join(CHOICES[key])}")
        for key in ("ai_key_env", "api_token_env"):
            if key == "ai_key_env" and not self.ai_key_env:
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", getattr(self, key)):
                raise ValueError(f"{key} must name an environment variable, not contain a secret")
        from .ai_providers import endpoint_url

        endpoint_url(self.ai_endpoint)
        return self

    @classmethod
    def load(cls):
        path = config_path()
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text())).validate()

    def save(self):
        self.validate()
        path = config_path()
        private_dir(path.parent)
        # Atomic replacement: a daemon cannot observe a partially written config.
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            json.dump(asdict(self), file, indent=2)
            temp = Path(file.name)
        temp.replace(path)


RANGES = {
    "max_seconds": (1, 3600),
    "audio_budget_mb": (1, 1000000),
    "overlay_size": (75, 150),
    "overlay_bottom_offset": (0, 1000),
    "preview_chars": (40, 2000),
    "visualizer_sensitivity": (10, 500),
    "sound_volume": (0, 100),
    "typing_wpm": (1, 300),
    "api_port": (1024, 65535),
    "file_speaker_count": (0, 20),
    "queue_limit": (1, 100),
    "ai_max_tokens": (1, 65536),
    "ai_timeout": (1, 300),
    "clipboard_restore_delay_ms": (100, 5000),
}
OPTIONAL = {
    "ai_key_env",
    "ai_profile",
    "ai_model",
    "ai_prompt",
    "sound_file",
    "microphone_preferred",
    "microphone_fallback",
    "microphone_never",
    "filler_words",
}
CHOICES = {
    "engine": ["whisper", "onnx"],
    "delivery": ["clipboard", "x11", "wayland"],
    "device": ["cpu", "cuda", "auto"],
    "paste_backend": ["auto", "xdotool", "wtype", "ydotool"],
    "paste_shortcut": ["ctrl+shift+v", "ctrl+v"],
    "theme": ["system", "light", "dark"],
    "overlay_position": ["bottom", "top"],
    "insertion_mode": ["clipboard", "direct"],
    "spoken_send_key": ["return", "shift+return", "ctrl+return"],
    "ai_provider": ["compatible", "anthropic", "gemini", "ollama"],
}
for _name in (
    "recording_panel",
    "remove_filler_words",
    "lowercase_first",
    "remove_trailing_period",
    "literal_formatting",
    "question_mark_shortcut",
    "spoken_formatting",
    "spoken_send",
    "send_after_pause",
    "ai_enabled",
    "save_history",
    "save_audio",
    "live_preview",
    "skip_silent_recordings",
    "overlay_border",
    "tray",
    "pause_media",
    "restore_clipboard",
    "copy_to_clipboard",
    "weekends_break_streak",
    "api_enabled",
    "file_speaker_labels",
    "smart_capitalization",
    "space_between_dictations",
    "show_microphone_alerts",
    "show_paste_alerts",
    "launch_at_startup",
    "show_window_at_login",
):
    CHOICES[_name] = ["on", "off"]
