"""Explicit per-application and shortcut prompt selection; no application activation."""

import json
import uuid
from dataclasses import replace

from .config import config_path
from .operations import write_private
from .profiles import load_profiles, resolve_profile


def validate_routes(routes, profiles=None):
    if not isinstance(routes, list) or len(routes) > 1000:
        raise ValueError("Prompt routes must be a list of at most 1000 applications")
    ids = {p["id"] for p in (load_profiles() if profiles is None else profiles)}
    seen = set()
    for rule in routes:
        if (
            not isinstance(rule, dict)
            or set(rule) - {"app", "profile", "mode"}
            or not {"app", "profile"} <= rule.keys()
            or rule.get("mode", "dictation") not in ("dictation", "edit", "command")
            or not isinstance(rule["app"], str)
            or not rule["app"].strip()
            or len(rule["app"]) > 200
            or (rule.get("mode", "dictation"), rule["app"].casefold()) in seen
        ):
            raise ValueError("Each application needs one unique prompt route")
        if not isinstance(rule.get("profile"), str) or rule["profile"] not in {
            "default",
            "off",
            *ids,
        }:
            raise ValueError("Prompt route selects a missing profile")
        seen.add((rule.get("mode", "dictation"), rule["app"].casefold()))
    return routes


def load_routes(profiles=None):
    path = config_path().with_name("prompt_routes.json")
    if not path.exists():
        return []
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Prompt routes exceed 1 MiB")
    return validate_routes(json.loads(path.read_text()), profiles)


def save_routes(routes):
    from .config import private_dir

    path = config_path().with_name("prompt_routes.json")
    private_dir(path.parent)
    write_private(path, json.dumps(validate_routes(routes), indent=2).encode())


def select_prompt(config, app="", override=None, routes=None, mode="dictation", profiles=None):
    """Resolve shortcut > per-mode app route > per-mode settings, freezing provider.

    Legacy routes without mode apply to dictation only. Default assistant modes
    share global provider credentials/model, not the global dictation prompt.
    """
    if mode not in ("dictation", "edit", "command"):
        raise ValueError("Prompt mode must be dictation, edit, or command")
    if override is not None:
        if not isinstance(override, str):
            raise ValueError("Prompt override must name a profile, default, or off")
        selected = override
    else:
        routes = load_routes() if routes is None else routes
        if mode == "dictation":
            fallback = "off" if config.ai_scope == "selected" else "default"
        else:
            fallback = getattr(config, mode + "_profile", "default")
        selected = next(
            (
                rule["profile"]
                for rule in routes
                if rule["app"].casefold() == app.casefold()
                and rule.get("mode", "dictation") == mode
            ),
            fallback,
        )
    if selected == "off":
        return replace(config, ai_enabled="off", ai_profile="")
    custom_profile = selected != "default"
    if custom_profile:
        uuid.UUID(selected)
        config = replace(
            config,
            ai_profile=selected,
            ai_enabled="on" if override is not None or mode != "dictation" else config.ai_enabled,
        )
    if config.ai_enabled == "on":
        config = replace(resolve_profile(config, profiles), ai_profile="")
    if mode != "dictation" and not custom_profile:
        from .assistance import COMMAND_PROMPT, EDIT_PROMPT

        explicit = getattr(config, mode + "_prompt", "")
        config = replace(
            config, ai_prompt=explicit or (COMMAND_PROMPT if mode == "command" else EDIT_PROMPT)
        )
    return config
