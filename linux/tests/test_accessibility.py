import json
import subprocess
from types import SimpleNamespace

import pytest

from mouthkeys import accessibility


def test_no_display_does_not_probe_the_installed_helper(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Desktop helper started"))
    assert accessibility.read_snapshot() is None
    assert not accessibility.focus_target({})
    assert accessibility.insert_text({}, "example") is None


def test_snapshot_uses_isolated_system_python_when_gi_is_missing(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(accessibility.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(accessibility.Path, "is_file", lambda path: True)

    def run(args, *, timeout, **kwargs):
        assert args[:2] == ["/usr/bin/python3", "-I"]
        assert args[2].endswith("/atspi_helper.py")
        assert json.loads(kwargs["input"]) == {"action": "snapshot", "target": None, "text": None}
        assert timeout == 2
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "text": "before after",
                    "foreground": True,
                    "protected": False,
                    "context": "before ",
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert accessibility.read_context() == "before "


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"text": "secret", "protected": True},
        {"text": 4, "protected": False},
        {"text": "x" * 100001, "protected": False},
        {"text": "too far", "protected": False, "foreground": False, "context": "too far"},
        {"text": "wrong type", "protected": False, "foreground": True, "context": []},
        {"text": "too long", "protected": False, "foreground": True, "context": "x" * 201},
    ],
)
def test_no_context_leaves_the_helper_for_protected_invalid_or_background_fields(
    monkeypatch, response
):
    monkeypatch.setenv("WAYLAND_DISPLAY", "fake")
    monkeypatch.setattr(accessibility.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(response))
    )
    assert accessibility.read_context() is None


@pytest.mark.parametrize("output", ["not JSON", "x" * (1024 * 1024 + 1)])
def test_snapshot_rejects_malformed_or_oversized_helper_output(monkeypatch, output):
    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(accessibility.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=output))
    assert accessibility.read_snapshot() is None


def test_insert_returns_helper_metadata_without_sending_keys(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fake")
    monkeypatch.setattr(accessibility.importlib.util, "find_spec", lambda name: object())
    target = {"field_id": "example"}
    expected = {"action": "insert", "target": target, "text": "héllo"}

    def run(args, **kwargs):
        assert args[1:] == ["-P", "-m", "mouthkeys.atspi_helper"]
        assert json.loads(kwargs["input"]) == expected
        return SimpleNamespace(stdout='{"inserted": true}')

    monkeypatch.setattr(subprocess, "run", run)
    assert accessibility.insert_text(target, "héllo") == {"inserted": True}
