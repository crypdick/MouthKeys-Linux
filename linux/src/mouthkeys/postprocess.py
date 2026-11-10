"""Explicit opt-in AI cleanup with provider and prompt-profile routing."""

from dataclasses import replace
from pathlib import Path

from .ai_providers import request_completion

# Adapted from SettingsStore.baseDictationPromptText at upstream d50ad008.
BASE_DICTATION_PROMPT = (
    (Path(__file__).parent / "assets" / "dictation_base_prompt.txt").read_text().strip()
)


def dictation_prompt(config):
    body = config.ai_prompt.strip()
    if config.ai_custom_prompt_only == "on" or body.casefold().startswith(
        BASE_DICTATION_PROMPT.casefold()
    ):
        return body
    return BASE_DICTATION_PROMPT + ("\n\n" + body if body else "")


def render_dictation_user_message(prompt_text, transcript):
    """Match upstream dictation templates; Command/Edit keep their own prompts."""
    if "${transcript}" in prompt_text:
        return prompt_text.replace("${transcript}", transcript)
    if not prompt_text.strip():
        return transcript
    return prompt_text + "\n\n" + transcript


def process(text, config, *, opener=None, on_delta=None, cancel_event=None):
    if config.ai_enabled != "on":
        return text
    text = text.strip()
    if not text:
        return ""
    from .profiles import resolve_profile

    config = resolve_profile(config)
    message = render_dictation_user_message(dictation_prompt(config), text)
    return request_completion(
        message,
        replace(config, ai_prompt=""),
        opener=opener,
        on_delta=on_delta,
        cancel_event=cancel_event,
    )
