import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from mouthkeys.assistant_history import AssistantHistory
from mouthkeys.backup import backup_export, backup_import, backup_preview
from mouthkeys.config import Config
from mouthkeys.storage import History

from .test_assistant_history import receipt, turn


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"updated": 10**400}, "Invalid assistant session metadata"),
        ({"title": "invalid \ud800"}, "Invalid assistant session metadata"),
        ({"updated": True}, "Invalid assistant session metadata"),
        ({"title": 42}, "Invalid assistant session metadata"),
        ({"updated": float("inf")}, "Invalid assistant session metadata"),
    ],
)
def test_invalid_session_metadata_is_rejected_before_any_backup_store_is_changed(
    tmp_path, change, message
):
    store = AssistantHistory()
    original = store.new("command")
    store.append(original["id"], turn("Saved"))
    store.record_execution(original["id"], receipt())
    original = store.get(original["id"])
    history = History()
    path = tmp_path / "backup.json"
    backup_export(history, path)
    document = json.loads(path.read_text())
    document["config"]["spoken_send"] = "on"
    document["assistant_history"][0].update(change)
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        backup_preview(path)
    with pytest.raises(ValueError, match=message):
        backup_import(history, path)
    assert store.get(original["id"]) == original
    assert Config.load().spoken_send == "off"


def test_concurrent_first_open_agrees_on_current_chat():
    workers = 4
    ready = Barrier(workers, timeout=3)

    def open_current(store):
        ready.wait()
        return store.current("command")["id"]

    stores = [AssistantHistory() for _ in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(open_current, store) for store in stores]
        ids = [future.result(timeout=4) for future in futures]
    assert len(set(ids)) == 1
    assert [row["id"] for row in stores[0].rows(mode="command")] == ids[:1]


def test_unreadable_selected_chat_is_replaced_without_deleting_saved_row():
    store = AssistantHistory()
    old = store.new("edit")
    with store.connect() as db:
        db.execute("UPDATE sessions SET messages=? WHERE id=?", ("broken", old["id"]))

    new = store.current("edit")
    assert new["id"] != old["id"]
    assert new["messages"] == []
    assert {row["id"] for row in store.rows()} == {old["id"], new["id"]}


def test_history_writes_validate_chat_mode_id_and_title():
    store = AssistantHistory()
    session = store.new("edit")
    missing = str(uuid.uuid4())
    with pytest.raises(ValueError, match="Invalid assistant mode"):
        store.current("read")
    with pytest.raises(ValueError, match="Chat title must"):
        store.rename(session["id"], " ")
    with pytest.raises(ValueError, match="Assistant session no longer exists"):
        store.append(missing, turn("Not saved"))
    with pytest.raises(ValueError, match="Command session no longer exists"):
        store.record_execution(session["id"], receipt())
    assert store.get(session["id"]) == session


@pytest.mark.parametrize(
    ("build_invalid", "message"),
    [
        (lambda _: {}, "Invalid assistant sessions"),
        (lambda _: [None], "Invalid assistant session"),
        (lambda valid: [valid, valid], "Duplicate assistant session"),
        (
            lambda valid: [
                {**valid, "mode": "edit", "executions": [{"message_count": 0, "result": receipt()}]}
            ],
            "Only command sessions",
        ),
        (
            lambda valid: [{**valid, "executions": None}],
            "Invalid command execution events",
        ),
        (
            lambda valid: [
                {
                    **valid,
                    "executions": [
                        {"message_count": 0, "result": receipt(output="x" * 261000)}
                        for _ in range(3)
                    ],
                }
            ],
            "events exceed 512 KiB",
        ),
    ],
)
def test_restore_rejects_invalid_session_batch_without_changes(build_invalid, message):
    store = AssistantHistory()
    original = store.new("command")
    with pytest.raises(ValueError, match=message):
        store.restore(build_invalid(original))
    assert store.export() == [original]
    assert store.current("command") == original


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (receipt(output="bad \ud800"), "receipt Unicode"),
        (receipt(output="字" * 180000), "receipt exceeds 512 KiB"),
    ],
)
def test_unencodable_or_oversized_receipt_is_never_written(bad_receipt, message):
    store = AssistantHistory()
    original = store.new("command")
    store.record_execution(original["id"], receipt())
    saved = store.get(original["id"])
    with pytest.raises(ValueError, match=message):
        store.record_execution(original["id"], bad_receipt)
    assert store.get(original["id"]) == saved


@pytest.mark.parametrize(
    ("text", "message"),
    [("bad \ud800", "valid Unicode"), ("x" * 525000, "session exceeds 1 MiB")],
)
def test_invalid_serialized_turn_cannot_change_prior_messages(text, message):
    store = AssistantHistory()
    session = store.new("edit")
    store.append(session["id"], turn("original"))
    with pytest.raises(ValueError, match=message):
        store.append(session["id"], turn(text))
    assert store.get(session["id"])["messages"] == turn("original")
