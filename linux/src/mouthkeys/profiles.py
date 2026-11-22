"""Named AI cleanup configurations. Secrets stay outside settings and backups."""

import json
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict, cast

from .ai_providers import PROVIDER_ENDPOINTS, endpoint_url
from .config import Config, config_path, private_dir
from .private_files import write_private


class PromptProfile(TypedDict):
    id: str
    name: str
    shortcut: str
    ai_provider: str
    ai_endpoint: str
    ai_model: str
    ai_key_env: str
    ai_credential_source: str
    ai_prompt: str
    ai_max_tokens: int
    ai_timeout: int
    ai_models_endpoint: str
    ai_reasoning_parameter: str
    ai_reasoning_value: str
    ai_streaming: str


PROFILE_FIELDS = (
    "ai_provider",
    "ai_endpoint",
    "ai_model",
    "ai_key_env",
    "ai_credential_source",
    "ai_prompt",
    "ai_max_tokens",
    "ai_timeout",
    "ai_models_endpoint",
    "ai_reasoning_parameter",
    "ai_reasoning_value",
    "ai_streaming",
)

PROFILE_DEFAULTS = {
    "ai_credential_source": "environment",
    "ai_models_endpoint": "",
    "ai_reasoning_parameter": "",
    "ai_reasoning_value": "",
    "ai_streaming": "off",
}


def profiles_path() -> Path:
    return config_path().parent / "profiles.json"


def validate_profiles(profiles: object) -> list[PromptProfile]:
    if not isinstance(profiles, list) or len(profiles) > 100:
        raise ValueError("AI profiles must be a list with at most 100 entries")
    profiles = [
        {"shortcut": "", **PROFILE_DEFAULTS, **profile} if isinstance(profile, dict) else profile
        for profile in profiles
    ]
    ids = set()
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != {
            "id",
            "name",
            "shortcut",
            *PROFILE_FIELDS,
        }:
            raise ValueError("AI profile has missing or unknown fields")
        for key in (
            "id",
            "name",
            "ai_provider",
            "ai_endpoint",
            "ai_model",
            "ai_key_env",
            "ai_prompt",
        ):
            if not isinstance(profile[key], str):
                raise ValueError(f"AI profile {key} must be text")
        try:
            uuid.UUID(profile["id"])
        except ValueError:
            raise ValueError("AI profile ID must be a UUID") from None
        if profile["id"] in ids:
            raise ValueError("Duplicate AI profile ID")
        ids.add(profile["id"])
        if (
            not isinstance(profile["shortcut"], str)
            or len(profile["shortcut"]) > 128
            or any(ord(character) < 32 for character in profile["shortcut"])
        ):
            raise ValueError("AI profile shortcut must be a printable chord")
        if not profile["name"].strip() or len(profile["name"]) > 100:
            raise ValueError("AI profile name must contain 1–100 characters")
        if profile["ai_provider"] not in PROVIDER_ENDPOINTS:
            raise ValueError("Unknown AI profile provider protocol")
        if not profile["ai_model"].strip() or len(profile["ai_model"]) > 500:
            raise ValueError("AI profile must name a model (maximum 500 characters)")
        if len(profile["ai_endpoint"]) > 2048:
            raise ValueError("AI profile endpoint exceeds 2048 characters")
        endpoint_url(profile["ai_endpoint"])
        for name in PROFILE_DEFAULTS:
            if not isinstance(profile[name], str):
                raise ValueError(f"AI profile {name} must be text")
        if profile["ai_models_endpoint"]:
            endpoint_url(profile["ai_models_endpoint"])
            from types import SimpleNamespace

            from .provider_tools import models_endpoint

            models_endpoint(SimpleNamespace(**profile))
        from types import SimpleNamespace

        from .ai_providers import validate_reasoning

        validate_reasoning(SimpleNamespace(**profile))
        if profile["ai_streaming"] not in ("on", "off"):
            raise ValueError("AI profile streaming must be on or off")
        if profile["ai_credential_source"] not in ("environment", "secret-service"):
            raise ValueError("AI credential source must be environment or secret-service")
        if profile["ai_key_env"] and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,127}", profile["ai_key_env"]
        ):
            raise ValueError("AI profile key must name an environment variable, never an API key")
        if len(profile["ai_prompt"].encode()) > 100000:
            raise ValueError("AI profile prompt exceeds 100 KB")
        for key, high in (("ai_max_tokens", 65536), ("ai_timeout", 300)):
            if type(profile[key]) is not int or not 1 <= profile[key] <= high:
                raise ValueError(f"AI profile {key} must be between 1 and {high}")
    if len(json.dumps(profiles).encode()) > 1024 * 1024:
        raise ValueError("AI profiles exceed 1 MB")
    return cast("list[PromptProfile]", profiles)


def load_profiles() -> list[PromptProfile]:
    path = profiles_path()
    if not path.exists():
        return []
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("AI profiles exceed 1 MB")
    return validate_profiles(json.loads(path.read_text()))


def save_profiles(profiles: object) -> None:
    validated = validate_profiles(profiles)
    path = profiles_path()
    private_dir(path.parent)
    write_private(path, json.dumps(validated, indent=2, ensure_ascii=False).encode())


def resolve_profile(config: Config, profiles: list[PromptProfile] | None = None) -> Config:
    if not getattr(config, "ai_profile", ""):
        return config
    available = load_profiles() if profiles is None else profiles
    profile = next((p for p in available if p["id"] == config.ai_profile), None)
    if profile is None:
        raise ValueError("Selected AI cleanup profile no longer exists")
    values: dict[str, Any] = dict(profile)
    return replace(config, **{field: values[field] for field in PROFILE_FIELDS}).validate()
