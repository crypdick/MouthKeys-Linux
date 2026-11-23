import json
from dataclasses import replace
from pathlib import Path

import pytest

from mouthkeys.config import Config, config_path


def test_save_rejects_blank_required_strings():
    with pytest.raises(ValueError, match="model must be a nonempty string"):
        replace(Config(), model=" \t").save()
    assert not config_path().exists()


@pytest.mark.parametrize("field", ["ai_key_env", "api_token_env"])
def test_save_rejects_invalid_environment_names_without_echoing_the_value(field):
    value = "not an environment name"
    with pytest.raises(ValueError, match=rf"{field} must name an environment variable") as error:
        replace(Config(), **{field: value}).save()
    assert value not in str(error.value)
    assert not config_path().exists()


def test_config_validation_and_atomic_roundtrip():
    config = Config(model="small", language="auto", max_seconds=60)
    config.save()
    assert Config.load() == config
    with pytest.raises(ValueError, match="max_seconds must be between"):
        Config(max_seconds=-1).save()
    assert Config.load() == config


def test_save_updates_relative_symlink_target(tmp_path):
    path = config_path()
    path.parent.mkdir(parents=True)
    target = tmp_path / "overlay" / "config.json"
    target.parent.mkdir()
    target.write_text('{"model": "small"}')
    link = Path("../../overlay/config.json")
    path.symlink_to(link)

    updated = replace(Config.load(), pause_media="on")
    updated.save()

    assert path.readlink() == link
    assert Config.from_mapping(json.loads(target.read_text())) == updated
    assert target.stat().st_mode & 0o777 == 0o600


def test_old_config_cannot_reenable_audio_retention():
    Config().save()
    path = config_path()
    values = json.loads(path.read_text())
    values.update(save_audio="on", audio_budget_mb=4000, reprocess_shortcut_key="super+r")
    path.write_text(json.dumps(values))
    loaded = Config.load()
    loaded.save()
    assert (
        not {"save_audio", "audio_budget_mb", "reprocess_shortcut_key"}
        & json.loads(path.read_text()).keys()
    )
