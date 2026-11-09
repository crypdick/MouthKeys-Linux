"""Explicit opt-in AI cleanup with provider and prompt-profile routing."""

from .ai_providers import request_completion


def process(text, config, *, opener=None):
    if config.ai_enabled != "on":
        return text
    from .profiles import resolve_profile

    return request_completion(text, resolve_profile(config), opener=opener)
