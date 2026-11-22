from dataclasses import asdict
from uuid import uuid4

import pytest

from mouthkeys import private_files
from mouthkeys.config import Config, config_path
from mouthkeys.profiles import PROFILE_FIELDS, load_profiles, save_profiles
from mouthkeys.routing import load_routes, save_routes, select_prompt


def profile():
    defaults = asdict(Config(ai_model="local-model"))
    return {"id": str(uuid4()), "name": "Local", **{key: defaults[key] for key in PROFILE_FIELDS}}


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"ai_prompt": 4}, "AI profile ai_prompt must be text"),
        ({"id": "bad-uuid"}, "AI profile ID must be a UUID"),
        ({"shortcut": "ctrl+f8\n"}, "AI profile shortcut must be a printable chord"),
        ({"name": " "}, "AI profile name must contain"),
        ({"ai_model": ""}, "AI profile must name a model"),
        ({"ai_endpoint": "https://example.test/" + "x" * 2048}, "AI profile endpoint exceeds"),
        ({"ai_models_endpoint": 4}, "AI profile ai_models_endpoint must be text"),
        ({"ai_streaming": "yes"}, "AI profile streaming must be on or off"),
        ({"ai_prompt": "é" * 50001}, "AI profile prompt exceeds"),
    ],
)
def test_invalid_profile_cannot_replace_existing_profile(changes, error):
    original = profile()
    save_profiles([original])
    with pytest.raises(ValueError, match=error):
        save_profiles([original | changes])
    assert load_profiles() == [original | {"shortcut": ""}]


def test_profiles_enforce_aggregate_limit_without_writing():
    profiles = [profile() | {"ai_prompt": "x" * 100000} for _ in range(11)]
    with pytest.raises(ValueError, match="AI profiles exceed 1 MB"):
        save_profiles(profiles)
    assert not config_path().with_name("profiles.json").exists()


def test_duplicate_profile_ids_cannot_replace_existing_profile():
    original = profile()
    save_profiles([original])
    with pytest.raises(ValueError, match="Duplicate AI profile ID"):
        save_profiles([original, original | {"name": "Duplicate"}])
    assert load_profiles() == [original | {"shortcut": ""}]


def test_tempfile_open_failure_preserves_profile_store(monkeypatch):
    original = profile()
    save_profiles([original])

    def fail_open(*args, **kwargs):
        raise PermissionError("cannot open temporary file")

    with monkeypatch.context() as patch:
        patch.setattr(private_files.tempfile, "NamedTemporaryFile", fail_open)
        with pytest.raises(PermissionError, match="cannot open temporary file"):
            save_profiles([profile()])
    assert load_profiles() == [original | {"shortcut": ""}]


@pytest.mark.parametrize(
    ("loader", "filename", "message"),
    [
        (load_profiles, "profiles.json", "AI profiles exceed 1 MB"),
        (load_routes, "prompt_routes.json", "Prompt routes exceed 1 MiB"),
    ],
)
def test_oversized_profile_and_route_files_are_rejected_before_parsing(loader, filename, message):
    path = config_path().with_name(filename)
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match=message):
        loader()


@pytest.mark.parametrize(
    ("save", "invalid", "message"),
    [
        (save_profiles, {}, "AI profiles must be a list"),
        (save_routes, {"app": "Editor", "profile": "off"}, "Prompt routes must be a list"),
    ],
)
def test_profiles_and_routes_do_not_accept_single_objects(save, invalid, message):
    with pytest.raises(ValueError, match=message):
        save(invalid)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"mode": "unrecognized"}, "Prompt mode must be dictation, edit, or command"),
        ({"override": 123}, "Prompt override must name a profile, default, or off"),
    ],
)
def test_prompt_selection_rejects_invalid_mode_and_override(options, message):
    with pytest.raises(ValueError, match=message):
        select_prompt(Config(), **options)
