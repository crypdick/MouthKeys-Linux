"""Explicit opt-in AI cleanup with provider and prompt-profile routing."""

from dataclasses import replace

from .ai_providers import request_completion


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
    message = render_dictation_user_message(config.ai_prompt, text)
    return request_completion(
        message,
        replace(config, ai_prompt=""),
        opener=opener,
        on_delta=on_delta,
        cancel_event=cancel_event,
    )
