"""Build and update the Settings form without owning the window or persistence."""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QWidget

from . import desktop_input, settings_choices
from .config import CHOICES, RANGES, Config
from .desktop_input import automatic_delivery_warning
from .profiles import PromptProfile
from .settings_fields import SettingsField, field_value

SettingOption = str | tuple[str, str]

COMPUTE_TYPES = [
    "default",
    "auto",
    "int8",
    "float32",
    "float16",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "bfloat16",
    "int16",
]

CATEGORIES = (
    "All settings",
    "Speech",
    "Microphone & media",
    "Text delivery",
    "Spoken Send",
    "Text formatting",
    "AI cleanup",
    "Appearance",
    "History & stats",
    "Desktop",
    "Local API",
    "Dictionary & learning",
)
GROUPED_KEYS = {
    "Speech": (
        "model",
        "language",
        "device",
        "compute_type",
        "engine",
        "max_seconds",
        "skip_silent_recordings",
        "queue_limit",
        "file_speaker_labels",
        "file_speaker_count",
        "file_diarization_mode",
        "model_revision",
        "native_streaming",
        "native_timeout_seconds",
        "streaming_latency",
        "preview_interval_ms",
        "asr_max_tokens",
    ),
    "AI cleanup": (
        "edit_profile",
        "command_profile",
        "edit_prompt",
        "command_prompt",
        "command_confirm",
        "command_continue",
    ),
    "Spoken Send": ("send_after_pause",),
    "Microphone & media": ("audio_source", "pause_media", "sounds_enabled", "mic_battery"),
    "Text delivery": (
        "delivery",
        "destination_guard",
        "insertion_mode",
        "restore_clipboard",
        "copy_to_clipboard",
        "target_policy",
        "return_to_start",
        "verify_paste",
        "edit_delivery",
    ),
    "Appearance": (
        "recording_panel",
        "theme",
        "preview_chars",
        "visualizer_sensitivity",
        "live_preview",
    ),
    "History & stats": ("typing_wpm", "weekends_break_streak", "file_history_limit"),
    "Desktop": (
        "hotkey_mode",
        "tray",
        "launch_at_startup",
        "show_window_at_login",
        "show_microphone_alerts",
        "show_paste_alerts",
        "show_ai_alerts",
        "hide_from_taskbar",
        "onboarding_complete",
    ),
    "Dictionary & learning": (
        "automatic_learning",
        "pronunciation_matching",
        "vocabulary_boosting",
    ),
}
GROUP_BY_KEY = {key: group for group, keys in GROUPED_KEYS.items() for key in keys}
PREFIX_GROUPS = (
    (("learning_", "pronunciation_"), "Dictionary & learning"),
    (("api_",), "Local API"),
    (("ai_",), "AI cleanup"),
    (("spoken_send",), "Spoken Send"),
    (("microphone_", "sound_"), "Microphone & media"),
    (("paste_", "clipboard_"), "Text delivery"),
    (("overlay_",), "Appearance"),
    (("save_",), "History & stats"),
)

LABELS = {
    "max_seconds": "Maximum recording length",
    "recording_panel": "Recording panel",
    "ai_key_env": "AI API key environment variable",
    "ai_credential_source": "AI credential source",
    "destination_guard": "Automatic delivery destination checks",
    "pronunciation_aligner_model": "Optional local Whisper pronunciation aligner",
    "ai_endpoint": "AI cleanup endpoint (full URL)",
    "spoken_send": "Spoken Send",
    "send_after_pause": "Send after a pause",
    "spoken_send_key": "Send key",
    "save_history": "Save transcription history",
    "microphone_preferred": "Preferred microphones (ordered, comma-separated)",
    "microphone_fallback": "Last-resort microphones (ordered, comma-separated)",
    "microphone_never": "Never use microphones (comma-separated)",
    "typing_wpm": "Typing speed for time-saved estimate (WPM)",
    "file_speaker_count": "File speaker count (0 = automatic)",
    "command_confirm": "Confirm before running commands",
    "ai_custom_prompt_only": "Send custom dictation prompt only",
    "hide_from_taskbar": "Hide main window from taskbar / app switcher",
    "show_ai_alerts": "AI processing failure notifications",
    "sounds_enabled": "Recording sounds",
}
EDITOR_TOOLTIPS = {
    "model": (
        "Approximate transcription accuracy, strongest first (English dictation).\n"
        "Accuracy varies by language and audio; .en and distilled models are English-only.\n"
        "Choose a model or enter a local model path / Hugging Face ID."
    ),
    "audio_source": (
        "Choose a microphone or enter a PulseAudio source name. Default follows your desktop."
    ),
    "compute_type": (
        "default follows the model's stored precision, with hardware fallbacks.\n"
        "auto selects the fastest supported type for your device.\n"
        "int8 reduces memory use; default may use more memory and run slower on CPU."
    ),
}
HELP_TEXT = {
    "ai_custom_prompt_only": (
        "Off prepends the built-in dictation-cleaning instructions to your custom prompt. "
        "On sends only your prompt and transcript. Command/Edit prompts are separate."
    ),
    "hide_from_taskbar": (
        "Use a Linux tool-window hint for the main window. Taskbar and app-switcher "
        "behavior depends on your compositor. Reopen through the tray or mouthkeys gui."
    ),
    "command_confirm": (
        "Every command shows the exact shell command and directory for review. "
        "Destructive uses upstream's command-text heuristic, not a sandbox. "
        "No individual confirmation runs model-generated commands as your user; "
        "continuation remains bounded to 20 steps."
    ),
    "max_seconds": ("At the limit, recording stops and text is copied without automatic paste."),
    "recording_panel": (
        "Hide the panel if your compositor tiles it or takes keyboard focus. "
        "Dictation still works; the panel also shows loading and shortcut errors."
    ),
    "spoken_send": (
        "Opt-in: send configured key only after automatic insertion into "
        "an unchanged X11 destination."
    ),
    "ai_enabled": (
        "When on, recorded transcript is sent to the configured local or remote AI endpoint."
    ),
    "ai_key_env": (
        "Set this variable in the recording service environment; API keys "
        "are never stored in backups."
    ),
    "ai_credential_source": (
        "Environment uses the named variable. Secret Service uses your desktop keyring "
        "for this exact provider endpoint; Save API key below. Settings never reads keys."
    ),
    "destination_guard": (
        "Both options check the field or pane when it can be identified. If no field was "
        "identified, window fallback can still paste or send in the same window, but "
        "cannot detect tab or pane switches. Require field or pane copies instead. "
        "Paste latest still pastes at current focus."
    ),
    "pronunciation_aligner_model": (
        "Blank disables extra alignment. A complete already-cached Whisper model/path "
        "can align engines without native word intervals for pronunciation matching. "
        "No download; separate CPU/int8 model adds RAM and processing time. "
        "Native word intervals remain preferred."
    ),
    "save_history": (
        "Off removes completed attempts after delivery; failed raw text is not retained either."
    ),
    "restore_clipboard": (
        "Restoration occurs only after supported automatic paste and only "
        "if clipboard ownership is unchanged."
    ),
    "api_enabled": (
        "Opt-in loopback HTTP API; requires a token in the configured "
        "environment variable. Restart service after changes."
    ),
}


def _profile_choices(profiles: list[PromptProfile]) -> dict[str, list[tuple[str, str]]]:
    profiles_by_name = [(profile["name"], profile["id"]) for profile in profiles]
    modes = [("Sync to global provider", "default"), ("Off", "off"), *profiles_by_name]
    return {
        "ai_profile": [("Default settings", ""), *profiles_by_name],
        "edit_profile": modes,
        "command_profile": modes,
    }


def _choices(profiles: list[PromptProfile]) -> dict[str, Sequence[SettingOption]]:
    return {
        **CHOICES,
        **_profile_choices(profiles),
        "model": settings_choices.model_choices("whisper"),
        "language": settings_choices.language_choices(),
        "compute_type": COMPUTE_TYPES,
        "audio_source": settings_choices.audio_sources(),
        "recording_panel": [("Show panel", "on"), ("Hide panel", "off")],
        "delivery": [
            ("Clipboard only", "clipboard"),
            ("X11 auto-paste", "x11"),
            ("Wayland auto-paste", "wayland"),
        ],
        "destination_guard": [
            ("Field first, window fallback", "field_then_window"),
            ("Require field or pane", "field"),
        ],
        "paste_shortcut": [
            ("Ctrl+Shift+V (Ghostty / most terminals)", "ctrl+shift+v"),
            ("Ctrl+V (most editors)", "ctrl+v"),
        ],
        "sound_preset": [
            ("Custom files", "custom"),
            ("None", "none"),
            ("Legacy MouthKeys pair", "mouthkeys"),
            *((f"Sound {index + 1}", f"fluid_sfx_{index}") for index in range(5)),
        ],
        "command_confirm": [
            ("Every command", "on"),
            ("Destructive commands (upstream heuristic)", "destructive"),
            ("No individual confirmation", "off"),
        ],
    }


def _group(key: str) -> str:
    if "shortcut" in key:
        return "Desktop"
    if key in GROUP_BY_KEY:
        return GROUP_BY_KEY[key]
    for prefixes, group in PREFIX_GROUPS:
        if key.startswith(prefixes):
            return group
    return "Text formatting"


def _editor(key: str, value: str | int, options: Sequence[SettingOption] | None) -> SettingsField:
    if key in RANGES:
        spin = QSpinBox()
        spin.setRange(*RANGES[key])
        if key == "max_seconds":
            spin.setSuffix(" seconds")
        spin.setValue(cast("int", value))
        return spin
    if options is None:
        return QLineEdit(cast("str", value))

    name = cast("str", value)
    # Faster Whisper gives local model directories precedence over aliases.
    if key == "model" and name in settings_choices.MODEL_ALIASES and not Path(name).is_dir():
        name = settings_choices.MODEL_ALIASES[name]
    combo = QComboBox()
    for option in options:
        label, data = option if isinstance(option, tuple) else (option, option)
        combo.addItem(label, data)
    if combo.findData(name) < 0:
        combo.addItem(name, name)
    combo.setCurrentIndex(combo.findData(name))
    if key in ("model", "audio_source"):
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    return combo


class SettingsForm:
    def __init__(self, layout: QFormLayout, config: Config, profiles: list[PromptProfile]) -> None:
        self.layout = layout
        self.fields: dict[str, SettingsField] = {}
        self.rows: list[tuple[int, str]] = []
        self.category = QComboBox()
        self.category.addItems(CATEGORIES)
        self.layout.addRow("Category", self.category)
        self.delivery_hint = QLabel()
        self.delivery_hint.setWordWrap(True)
        choices = _choices(profiles)
        for key, value in asdict(config).items():
            field = _editor(key, value, choices.get(key))
            self.fields[key] = field
            category = _group(key)
            label = LABELS.get(key, key.replace("_", " ").capitalize())
            self.add_grouped_row(field, category, label=label)
            if key == "delivery":
                self.add_grouped_row(self.delivery_hint, category, label="")
            elif key in HELP_TEXT:
                field.setToolTip(HELP_TEXT[key])
                hint = QLabel(HELP_TEXT[key])
                hint.setWordWrap(True)
                self.add_grouped_row(hint, category, label="")
        for key, tooltip in EDITOR_TOOLTIPS.items():
            self.fields[key].setToolTip(tooltip)
        for key in ("delivery", "paste_backend", "insertion_mode"):
            self._combo(key).currentIndexChanged.connect(self._update_delivery_hint)
        self._update_delivery_hint()
        self._combo("engine").currentTextChanged.connect(self._refresh_models)
        if config.engine != "whisper":
            self._refresh_models(config.engine)
        self.category.currentTextChanged.connect(self._filter)

    def _combo(self, key: str) -> QComboBox:
        return cast("QComboBox", self.fields[key])

    def add_grouped_row(self, widget: QWidget, group: str, *, label: str | None = None) -> None:
        self.rows.append((self.layout.rowCount(), group))
        if label is None:
            self.layout.addRow(widget)
        else:
            self.layout.addRow(label, widget)

    def _filter(self, category: str) -> None:
        for row, group in self.rows:
            self.layout.setRowVisible(row, category in ("All settings", group))

    def _refresh_models(self, engine: str) -> None:
        field = self._combo("model")
        current = field.currentText()
        names = settings_choices.model_choices(engine)
        field.clear()
        for name in names:
            field.addItem(name, name)
        if current and current not in names:
            field.addItem(current, current)
        field.setCurrentIndex(field.findData(current))

    def refresh_profiles(self, profiles: list[PromptProfile]) -> None:
        for key, options in _profile_choices(profiles).items():
            picker = self._combo(key)
            selected = picker.currentData()
            picker.clear()
            for label, value in options:
                picker.addItem(label, value)
            picker.setCurrentIndex(max(0, picker.findData(selected)))

    def _update_delivery_hint(self, _index: int = -1) -> None:
        config = Config(
            **{
                key: field_value(self.fields[key])
                for key in ("delivery", "paste_backend", "insertion_mode")
            },
        )
        session = desktop_input.session()
        warning = automatic_delivery_warning(config)
        message = (
            "Dictation copies text. Paste manually in the destination."
            if config.delivery == "clipboard"
            else "Auto-paste requires an unchanged destination. Otherwise, text is copied."
        )
        if warning:
            message = f"{warning} Clipboard fallback stays available."
            if config.insertion_mode == "direct" and config.delivery == session:
                message += " Direct insertion may still work in supported fields."
        self.delivery_hint.setText(f"Detected session: {session}. {message}")
