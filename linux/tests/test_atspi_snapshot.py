import pytest

from mouthkeys import atspi_helper as atspi

from .test_atspi import REF, FakeBus


@pytest.mark.parametrize(("method", "value"), [("GetNSelections", -1), ("GetNSelections", True)])
def test_snapshot_rejects_malformed_selection_count_before_reading_text(method, value):
    bus = FakeBus()
    original = bus.call
    reads = []

    def call(ref, interface, name, signature=None, args=()):
        reads.append(name)
        if name == method:
            return (value,)
        return original(ref, interface, name, signature, args)

    bus.call = call
    assert atspi.snapshot(bus, REF) is None
    assert "GetText" not in reads


def test_direct_insertion_refuses_caret_inconsistent_with_captured_insertion_point():
    bus = FakeBus()
    bus.selection = (9, 9)
    before = atspi.snapshot(bus, REF)
    before["caret"] = 0

    assert atspi.operate(bus, {"action": "insert", "target": before, "text": "new"}) is None
    assert bus.writes == []
    assert bus.text == "hello old world"


def test_verification_does_not_reread_text_after_role_becomes_protected():
    bus = FakeBus()
    original = bus.call
    text_reads = []

    def call(ref, interface, method, signature=None, args=()):
        if method == "GetText":
            text_reads.append(bus.role)
            result = original(ref, interface, method, signature, args)
            bus.role = bus.password_role
            return result
        return original(ref, interface, method, signature, args)

    bus.call = call
    assert atspi.snapshot(bus, REF) is None
    assert text_reads == [1]


@pytest.mark.parametrize("hierarchy", ["cycle", "deep", "stale"])
def test_missing_ancestor_evidence_never_corroborates_focused_field(hierarchy):
    bus = FakeBus()
    parent_reads = []
    original = bus.prop

    def prop(ref, interface, name):
        if name != "Parent":
            return original(ref, interface, name)
        parent_reads.append(ref)
        if hierarchy == "stale":
            raise RuntimeError("Detached toolkit object")
        if hierarchy == "cycle":
            return ref
        return (ref[0], f"/org/a11y/atspi/accessible/{len(parent_reads) + 10}")

    bus.prop = prop
    value = atspi.snapshot(bus, REF)
    assert value["focused"] is True
    assert value["foreground"] is None
    assert len(parent_reads) == (48 if hierarchy == "deep" else 2)
    assert not bus.writes


def test_discovery_skips_background_passwords_invalid_nodes_and_duplicates():
    bus = FakeBus()
    password = (REF[0], "/org/a11y/atspi/accessible/password")
    stale = (REF[0], "/org/a11y/atspi/accessible/stale")
    duplicate = (REF[0], "/org/a11y/atspi/accessible/duplicate")
    visited_duplicate = []
    original = bus.call

    def call(ref, interface, name, signature=None, args=()):
        if ref == atspi.ROOT:
            return (
                (0,) if name == "GetRole" else ([password, stale, duplicate, duplicate, [], REF],)
            )
        if ref == password:
            assert interface != atspi.TEXT
            return (bus.password_role,) if name == "GetRole" else ([0],)
        if ref == stale and name == "GetState":
            return (["not a state word"],)
        if ref == duplicate and name == "GetChildren":
            visited_duplicate.append(ref)
            return (None,)
        if ref == duplicate and name == "GetState":
            return ([0],)
        return original(ref, interface, name, signature, args)

    bus.call = call
    value = atspi.operate(bus, {})
    assert value["field_id"] == f"{REF[0]}|{REF[1]}"
    assert visited_duplicate == [duplicate]
    assert not bus.writes


def test_discovery_stops_without_a_foreground_field_or_after_deadline():
    bus = FakeBus()
    original = bus.call

    def call(ref, interface, name, signature=None, args=()):
        if name == "GetChildren":
            return ([REF] if ref == atspi.ROOT else [],)
        return original(ref, interface, name, signature, args)

    bus.call = call
    bus.active = False
    assert atspi.find_focused(bus) is None
    bus.deadline = 0
    assert atspi.find_focused(bus) is None


@pytest.mark.parametrize(
    ("name_reply", "expected"),
    [
        ("Editor" * 100, ("Editor" * 100)[:200]),
        (123, ""),
    ],
)
def test_snapshot_reports_app_name_with_provenance_and_size_bound(name_reply, expected):
    bus = FakeBus()
    original = bus.prop

    def prop(ref, interface, name):
        if name == "Name":
            assert ref == (REF[0], atspi.ROOT[1])
            return name_reply
        return original(ref, interface, name)

    bus.prop = prop
    assert atspi.snapshot(bus, REF)["app"] == expected


def test_snapshot_rejects_changed_selection_count_even_if_caret_and_range_match():
    bus = FakeBus()
    bus.selection = (9, 9)
    original = bus.call
    text_read = False

    def call(ref, interface, name, signature=None, args=()):
        nonlocal text_read
        if name == "GetText":
            text_read = True
        if name == "GetNSelections" and text_read:
            return (1,)
        return original(ref, interface, name, signature, args)

    bus.call = call
    assert atspi.snapshot(bus, REF) is None


def test_snapshot_verification_keeps_original_bounded_text_window():
    bus = FakeBus()
    bus.text = "x" * 150000
    bus.caret = 140000
    bus.selection = (bus.caret, bus.caret)
    before = atspi.snapshot(bus, REF)
    bus.text = bus.text[:140000] + "new" + bus.text[140000:]
    bus.caret = 140003
    bus.selection = (bus.caret, bus.caret)

    current = atspi.operate(bus, {"action": "snapshot", "target": before})
    assert current["text_start"] == before["text_start"]
    assert current["text"] == "x" * 2000 + "new" + "x" * 2000
    assert current["character_count"] == before["character_count"] + 3
    assert not current["complete"]


@pytest.mark.parametrize("offset", ["invalid", -1])
def test_invalid_saved_window_never_reads_arbitrary_field_text(offset):
    bus = FakeBus()
    before = atspi.snapshot(bus, REF)
    original = bus.call

    def call(ref, interface, name, signature=None, args=()):
        assert name != "GetText"
        return original(ref, interface, name, signature, args)

    bus.call = call
    assert (
        atspi.operate(
            bus,
            {"action": "snapshot", "target": {**before, "complete": False, "text_start": offset}},
        )
        is None
    )


def test_toolkit_may_refuse_explicit_focus_without_any_followup_mutation():
    bus = FakeBus()
    before = atspi.snapshot(bus, REF)
    original = bus.call

    def call(ref, interface, name, signature=None, args=()):
        if name == "GrabFocus":
            return (False,)
        return original(ref, interface, name, signature, args)

    bus.call = call
    bus.focused = False
    assert atspi.operate(bus, {"action": "focus", "target": before}) is False
    assert not bus.focused
    assert not bus.writes


@pytest.mark.parametrize("target", [None, {"bus": "org.example", "path": REF[1]}])
def test_mutating_operations_require_the_unique_captured_target(target):
    bus = FakeBus()
    assert atspi.operate(bus, {"action": "insert", "target": target, "text": "new"}) is None
    assert not bus.writes


def test_unknown_operation_does_not_mutate_a_valid_target():
    bus = FakeBus()
    before = atspi.snapshot(bus, REF)
    with pytest.raises(ValueError, match="Unknown accessibility action"):
        atspi.operate(bus, {"action": "replace everything", "target": before})
    assert not bus.writes


@pytest.mark.parametrize("text", [None, "x" * (atspi.LIMIT + 1), "x" * atspi.LIMIT])
def test_direct_insert_refuses_invalid_payload_or_result_above_text_limit(text):
    bus = FakeBus()
    bus.selection = (bus.caret, bus.caret)
    before = atspi.snapshot(bus, REF)
    assert atspi.operate(bus, {"action": "insert", "target": before, "text": text}) is None
    assert not bus.writes


@pytest.mark.parametrize("fault", ["refused", "caret", "lost-focus"])
def test_direct_insert_never_retries_a_toolkit_write(fault):
    bus = FakeBus()
    bus.selection = (bus.caret, bus.caret)
    before = atspi.snapshot(bus, REF)
    original = bus.call
    attempts = []

    def call(ref, interface, name, signature=None, args=()):
        if name == "InsertText":
            attempts.append(args)
            if fault == "refused":
                return (False,)
            if fault == "lost-focus":
                bus.focused = False
        if name == "SetCaretOffset" and fault == "caret":
            raise RuntimeError("Caret movement unsupported")
        return original(ref, interface, name, signature, args)

    bus.call = call
    result = atspi.operate(bus, {"action": "insert", "target": before, "text": "é"})
    assert attempts == [(9, "é", 2)]
    if fault == "refused":
        assert not bus.writes
        assert result is None
    elif fault == "lost-focus":
        assert len(bus.writes) == 1
        assert result is None
    else:
        assert result["text"] == "hello oldé world"
        assert result["caret"] == before["caret"]
