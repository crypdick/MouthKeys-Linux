import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from mouthkeys.assistant_history import AssistantHistory, context_messages
from mouthkeys.config import Config


def turn(text):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": "Answer: " + text}]


def test_sessions_survive_restart_switch_rename_delete_and_preserve_concurrent_turns():
    store = AssistantHistory()
    first = store.current("command")
    second = store.new("command")
    assert AssistantHistory().current("command")["id"] == second["id"]
    store.select(first["id"])
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(AssistantHistory().append, first["id"], turn(str(i))) for i in range(2)
        ]
        for future in futures:
            future.result(timeout=3)
    assert len(store.get(first["id"])["messages"]) == 4
    store.rename(first["id"], "My command chat")
    assert AssistantHistory().current("command")["title"] == "My command chat"
    assert store.path.stat().st_mode & 0o777 == 0o600
    store.delete(first["id"])
    assert store.current("command")["id"] != first["id"]


def test_session_and_provider_context_remain_bounded_and_complete():
    store = AssistantHistory()
    current = store.new("edit")
    for index in range(105):
        store.append(current["id"], turn(str(index)))
    messages = store.get(current["id"])["messages"]
    assert len(messages) == 200
    assert messages[0]["content"] == "5"
    history = context_messages(messages)
    assert len(history) == 32
    assert history[0]["role"] == "user" and history[-1]["role"] == "assistant"
    large = turn("x" * 125000) + turn("Small")
    assert context_messages(large) == turn("Small")
    with pytest.raises(ValueError, match="completed"):
        store.append(current["id"], [{"role": "user", "content": "Unfinished"}])


def test_backup_roundtrip_includes_sessions_and_rejects_invalid_before_mutation(tmp_path):
    from mouthkeys.operations import backup_export, backup_import
    from mouthkeys.storage import History

    store = AssistantHistory()
    current = store.new("command")
    store.append(current["id"], turn("Original"))
    path = tmp_path / "backup.json"
    backup_export(History(), path)
    data = json.loads(path.read_text())
    assert data["assistant_history"][0]["messages"] == turn("Original")
    data["assistant_history"][0]["messages"][1]["role"] = "system"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        backup_import(History(), path)
    assert store.get(current["id"])["messages"] == turn("Original")
    del data["assistant_history"]
    path.write_text(json.dumps(data))
    backup_import(History(), path)
    assert store.get(current["id"])["messages"] == turn("Original")


def test_cli_saved_chat_management(capsys):
    from mouthkeys.cli import main

    assert main(["assistant-history", "--new", "--mode", "edit"]) in (None, 0)
    created = json.loads(capsys.readouterr().out)
    assert main(["assistant-history", "--id", created["id"], "--rename", "Draft"]) in (None, 0)
    assert json.loads(capsys.readouterr().out)["title"] == "Draft"
    Config(save_history="off").save()
    assert main(["assistant-history", "--new", "--mode", "command"]) == 1
    assert "save_history" in capsys.readouterr().err


def receipt(command="pwd", output="directory"):
    return {
        "command": command,
        "working_directory": "/tmp",
        "returncode": 0,
        "output": output,
        "truncated": False,
        "timed_out": False,
    }


def test_execution_receipt_survives_restart_failure_backup_and_is_not_a_fake_turn():
    from mouthkeys.assistant_history import pending_executions

    store = AssistantHistory()
    session = store.new("command")
    store.append(session["id"], turn("Proposal"))
    store.record_execution(session["id"], receipt())
    restored = AssistantHistory().get(session["id"])
    assert restored["messages"] == turn("Proposal")
    assert restored["executions"] == [{"message_count": 2, "result": receipt()}]
    assert pending_executions(restored) == [receipt()]
    exported = store.export()
    store.delete(session["id"])
    store.restore(exported)
    assert pending_executions(store.get(session["id"])) == [receipt()]
    store.append(session["id"], turn("Observed outcome reviewed"))
    assert pending_executions(store.get(session["id"])) == []
    assert len(store.get(session["id"])["executions"]) == 1


def test_execution_receipts_have_bounded_retention_and_cascade_delete(monkeypatch):
    import mouthkeys.assistant_history as module

    monkeypatch.setattr(module, "MAX_EXECUTIONS", 3)
    store = AssistantHistory()
    session = store.new("command")
    for index in range(5):
        store.record_execution(session["id"], receipt(str(index)))
    assert [event["result"]["command"] for event in store.get(session["id"])["executions"]] == [
        "2",
        "3",
        "4",
    ]
    store.delete(session["id"])
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 0


def test_message_window_rebases_execution_consumption_in_disk_and_private_memory():
    from mouthkeys.assistant_history import (
        append_private_messages,
        pending_executions,
        record_private_execution,
    )

    store = AssistantHistory()
    session = store.new("command")
    messages = turn("old") * 100
    store.append(session["id"], messages)
    store.record_execution(session["id"], receipt())
    store.append(session["id"], turn("new"))
    saved = store.get(session["id"])
    assert len(saved["messages"]) == 200
    assert saved["executions"][0]["message_count"] == 198
    assert pending_executions(saved) == []
    private = {"messages": messages}
    record_private_execution(private, receipt())
    append_private_messages(private, turn("new"))
    assert private["executions"][0]["message_count"] == 198
    assert pending_executions(private) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"returncode": True},
        {"output": []},
        {"command": "bad\0command"},
        {"working_directory": "relative"},
        {"cancelled": "yes"},
        {"secret": "key"},
    ],
)
def test_invalid_receipts_never_mutate_persistent_history(bad):
    store = AssistantHistory()
    session = store.new("command")
    with pytest.raises(ValueError):
        store.record_execution(session["id"], {**receipt(), **bad})
    assert store.get(session["id"])["executions"] == []


def test_legacy_sessions_restore_without_execution_field():
    store = AssistantHistory()
    session = store.new("command")
    del session["executions"]
    store.restore([session])
    assert store.get(session["id"])["executions"] == []


def test_receipts_prune_to_serialized_byte_bound_and_chat_retention(monkeypatch):
    import mouthkeys.assistant_history as module

    monkeypatch.setattr(module, "MAX_EXECUTION_BYTES", 1024)
    monkeypatch.setattr(module, "MAX_SESSIONS", 2)
    store = AssistantHistory()
    old = store.new("command")
    store.record_execution(old["id"], receipt("first", "x" * 512))
    store.record_execution(old["id"], receipt("latest", "y" * 512))
    events = store.get(old["id"])["executions"]
    assert len(events) == 1 and events[0]["result"]["command"] == "latest"
    assert len(json.dumps(events, ensure_ascii=False).encode()) <= 1024
    store.new("command")
    store.new("command")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM executions").fetchone()[0] == 0


def test_invalid_imported_execution_position_rejected_before_restore():
    store = AssistantHistory()
    session = store.new("command")
    invalid = {**session, "executions": [{"message_count": 2, "result": receipt()}]}
    with pytest.raises(ValueError):
        store.restore([invalid])
    assert store.get(session["id"])["executions"] == []
