"""Explicit Secret Service storage via libsecret; never persist keys in app data.

secret-tool's store reads a secret from stdin, not command arguments. Attribute
matching scopes keys to the exact configured provider endpoint, avoiding silent
credential reuse after a profile is pointed at a different server.
https://gnome.pages.gitlab.gnome.org/libsecret/coverage/tool/secret-tool.c.gcov.html
"""

import os
import subprocess


def validate_key(key):
    if (
        not isinstance(key, str)
        or len(key) > 4096
        or any(not 33 <= ord(character) <= 126 for character in key)
    ):
        raise ValueError("AI API key contains invalid characters")
    return key


def _attributes(config):
    from .ai_providers import endpoint_url

    return [
        "application",
        "mouthkeys-linux",
        "provider",
        config.ai_provider,
        "endpoint",
        endpoint_url(config.ai_endpoint),
    ]


def _run(config, action, *, secret=None):
    args = ["secret-tool", action]
    if action == "store":
        args += ["--label=MouthKeys AI API key"]
    args += _attributes(config)
    try:
        result = subprocess.run(
            args,
            input=secret.encode() if secret is not None else None,
            stdout=subprocess.PIPE if action == "lookup" else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=3 if action == "lookup" else 30,
        )
    except FileNotFoundError:
        raise RuntimeError("Install libsecret-tools and enable a desktop Secret Service") from None
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError(
            "Keyring unavailable or timed out; unlock your desktop keyring"
        ) from None
    if result.returncode != 0:
        # Lookup uses exit 1 for absent or inaccessible secrets; never quote helper diagnostics.
        if action == "lookup" and result.returncode == 1:
            raise RuntimeError(
                "No accessible API key for this endpoint; save or unlock it in Settings"
            )
        raise RuntimeError("Keyring operation failed; unlock your desktop keyring")
    return result.stdout if action == "lookup" else None


def store_key(config, key):
    if not validate_key(key):
        raise ValueError("Provide a nonempty API key")
    _run(config, "store", secret=key)


def delete_key(config):
    _run(config, "clear")


def api_key(config):
    if getattr(config, "ai_credential_source", "environment") == "secret-service":
        raw = _run(config, "lookup")
        try:
            # secret-tool terminates nonterminal lookup output with a newline.
            key = raw.decode("utf-8").removesuffix("\n")
        except UnicodeError:
            raise ValueError("Keyring returned an invalid API key") from None
        if not key:
            raise RuntimeError("No API key saved for this endpoint")
    else:
        key = os.environ.get(config.ai_key_env, "") if config.ai_key_env else ""
    return validate_key(key)
