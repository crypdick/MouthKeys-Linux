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
    filler_words: str = "um,uh,er,ah,eh,umm,uhh,err,ahh,ehh,hmm,hm,mm,mmm,erm,urm,ugh"
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
    ai_custom_prompt_only: str = "off"
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
    hide_from_taskbar: str = "off"
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
    show_ai_alerts: str = "on"
    sounds_enabled: str = "on"
    launch_at_startup: str = "off"
    show_window_at_login: str = "off"
    clipboard_restore_delay_ms: int = 500
    save_file_history: str = "on"
    file_history_limit: int = 50
    automatic_learning: str = "off"
    learning_confirmation: str = "on"
    learning_min_occurrences: int = 2
    learning_observe_seconds: int = 30
    pronunciation_matching: str = "off"
    vocabulary_boosting: str = "off"
    target_policy: str = "stop"
    return_to_start: str = "off"
    verify_paste: str = "on"
    overlay_style: str = "regular"
    mic_battery: str = "off"
    clipboard_exclude_history: str = "on"
    model_revision: str = "main"
    asr_max_tokens: int = 512
    ai_scope: str = "all"
    edit_delivery: str = "review"
    native_streaming: str = "on"
    streaming_latency: str = "model"
    preview_interval_ms: int = 2000
    onboarding_complete: str = "off"
    shortcut_backend: str = "desktop"
    hotkey_mode: str = "toggle"
    toggle_shortcut: str = "super+alt+space"
    cancel_shortcut: str = "super+alt+escape"
    paste_shortcut_key: str = ""
    reprocess_shortcut_key: str = ""
    ai_models_endpoint: str = ""
    ai_reasoning_parameter: str = ""
    ai_reasoning_value: str = ""
    sound_stop_file: str = ""
    sound_preset: str = "custom"
    native_timeout_seconds: int = 300
    ai_streaming: str = "off"
    ai_show_thinking: str = "off"
    command_continue: str = "on"
    command_confirm: str = "on"
    command_shortcut_key: str = ""
    edit_shortcut_key: str = ""
    edit_profile: str = "default"
    command_profile: str = "default"
    edit_prompt: str = ""
    command_prompt: str = ""
    file_diarization_mode: str = "turns"
    regional_filler_answered: str = "off"
    pronunciation_span_budget: int = 2000

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
        from .ai_providers import endpoint_url, validate_reasoning

        endpoint_url(self.ai_endpoint)
        if self.ai_models_endpoint:
            endpoint_url(self.ai_models_endpoint)
            from .provider_tools import models_endpoint

            models_endpoint(self)
        validate_reasoning(self)
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
    "pronunciation_span_budget": (100, 20000),
    "native_timeout_seconds": (10, 3600),
    "preview_interval_ms": (100, 10000),
    "learning_min_occurrences": (1, 10),
    "learning_observe_seconds": (5, 120),
    "asr_max_tokens": (1, 4096),
    "file_history_limit": (1, 10000),
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
    "edit_prompt",
    "command_prompt",
    "command_shortcut_key",
    "edit_shortcut_key",
    "ai_models_endpoint",
    "ai_reasoning_parameter",
    "ai_reasoning_value",
    "sound_stop_file",
    "paste_shortcut_key",
    "reprocess_shortcut_key",
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
    "command_confirm": ["on", "destructive", "off"],
    "file_diarization_mode": ["turns", "overlap"],
    "sound_preset": [
        "custom",
        "none",
        "mouthkeys",
        "fluid_sfx_0",
        "fluid_sfx_1",
        "fluid_sfx_2",
        "fluid_sfx_3",
        "fluid_sfx_4",
    ],
    "shortcut_backend": ["desktop", "x11", "portal"],
    "hotkey_mode": ["toggle", "hold", "automatic"],
    "streaming_latency": ["model", "80", "160", "320", "560", "1120"],
    "engine": ["whisper", "onnx", "transformers", "nemo"],
    "ai_scope": ["all", "selected"],
    "edit_delivery": ["review", "replace"],
    "target_policy": ["stop", "start"],
    "overlay_style": ["regular", "compact"],
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
    "ai_custom_prompt_only",
    "hide_from_taskbar",
    "regional_filler_answered",
    "ai_streaming",
    "ai_show_thinking",
    "command_continue",
    "show_ai_alerts",
    "sounds_enabled",
    "native_streaming",
    "onboarding_complete",
    "automatic_learning",
    "learning_confirmation",
    "pronunciation_matching",
    "vocabulary_boosting",
    "return_to_start",
    "verify_paste",
    "mic_battery",
    "clipboard_exclude_history",
    "save_file_history",
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
