"""Learn only observed edits inside a verified insertion, never inferred corrections.

Adapted from upstream AutomaticDictionaryCorrectionTracker (d50ad008). Linux
uses bounded read-only AT-SPI snapshots; full field contents stay in memory only.
"""

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass

from .config import Config, config_path, private_dir
from .dictionary import Dictionary
from .operations import write_private
from .vocabulary import VocabularyStore

_BOUNDARIES = set(',!?;:"“”()[]{}<>')
_EDGES = ' \t\r\n.,!?;:"“”‘’()[]{}'
_STORE_LOCK = threading.RLock()


def changed_range(before, after):
    start = 0
    while start < min(len(before), len(after)) and before[start] == after[start]:
        start += 1
    old_end, new_end = len(before), len(after)
    while old_end > start and new_end > start and before[old_end - 1] == after[new_end - 1]:
        old_end -= 1
        new_end -= 1
    return start, old_end, new_end


def _token_range(text, start, end):
    while start > 0 and not (text[start - 1].isspace() or text[start - 1] in _BOUNDARIES):
        start -= 1
    while end < len(text) and not (text[end].isspace() or text[end] in _BOUNDARIES):
        end += 1
    return start, end


def correction_candidate(before, after, inserted_start, inserted_end):
    if before == after:
        return None
    start, old_end, new_end = changed_range(before, after)
    if start < inserted_start or old_end > inserted_end:
        return None
    # Inserting punctuation/another sentence after dictation is not a correction.
    if start == old_end == inserted_end and any(
        char.isspace() or char in _BOUNDARIES for char in after[start:new_end]
    ):
        return None
    left, right = _token_range(before, start, old_end)
    if left < inserted_start or right > inserted_end:
        return None
    new_left, new_right = _token_range(after, start, new_end)
    heard, corrected = before[left:right].strip(_EDGES), after[new_left:new_right].strip(_EDGES)
    if any(not value or len(value) > 40 or len(value.split()) > 3 for value in (heard, corrected)):
        return None
    semantic = [" ".join(re.findall(r"\w+", value.casefold())) for value in (heard, corrected)]
    if (
        len(heard) + len(corrected) > 70
        or semantic[0] == semantic[1]
        or any(len(value) < 2 or not any(char.isalpha() for char in value) for value in semantic)
    ):
        return None
    return {"heard": heard, "corrected": corrected}


@dataclass
class Observation:
    field_id: str
    before: str
    after: str
    start: int
    end: int
    current_end: int
    expires: float
    changed_at: float | None = None


class CorrectionLearner:
    def __init__(self, config=None, *, clock=time.monotonic, wall_clock=time.time):
        self.config = config or Config.load()
        self.clock, self.wall_clock = clock, wall_clock
        self.observation = None
        self.proposals = {}
        self.lock = threading.RLock()

    @staticmethod
    def path():
        return config_path().with_name("learning.json")

    def _load(self):
        path = self.path()
        if not path.exists():
            return {"pairs": {}, "last_shown": 0}
        if path.stat().st_size > 256 * 1024:
            raise ValueError("Learning policy file exceeds 256 KiB")
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("pairs"), dict):
            raise ValueError("Invalid learning policy file")
        return data

    def _save(self, data):
        path = self.path()
        private_dir(path.parent)
        write_private(path, json.dumps(data, ensure_ascii=False).encode())

    @staticmethod
    def _valid_snapshot(snapshot, *, insertion=False):
        return (
            isinstance(snapshot, dict)
            and isinstance(snapshot.get("field_id"), str)
            and bool(snapshot["field_id"])
            and isinstance(snapshot.get("text"), str)
            and len(snapshot["text"]) <= 100000
            and snapshot.get("protected") is False
            and snapshot.get("complete") is not False
            and snapshot.get("focused") is True
            and snapshot.get("foreground") is True
            and snapshot.get("editable") is True
            and not snapshot.get("terminal", False)
            and type(snapshot.get("caret")) is int
            and 0 <= snapshot["caret"] <= len(snapshot["text"])
            and type(snapshot.get("selection_start")) is int
            and type(snapshot.get("selection_end")) is int
            and 0
            <= snapshot["selection_start"]
            <= snapshot["selection_end"]
            <= len(snapshot["text"])
            and (
                not insertion
                or snapshot["selection_start"] == snapshot["selection_end"] == snapshot["caret"]
            )
        )

    def begin(self, delivered_text, initial_snapshot, config=None):
        with self.lock:
            self.cancel()
            if config is not None:
                self.config = config
            if (
                getattr(self.config, "automatic_learning", "off") != "on"
                or not isinstance(delivered_text, str)
                or not delivered_text
                or not self._valid_snapshot(initial_snapshot, insertion=True)
            ):
                return False
            end = initial_snapshot["caret"]
            start = end - len(delivered_text)
            if start < 0 or initial_snapshot["text"][start:end] != delivered_text:
                return False
            self.observation = Observation(
                initial_snapshot["field_id"],
                initial_snapshot["text"],
                initial_snapshot["text"],
                start,
                end,
                end,
                self.clock() + getattr(self.config, "learning_observe_seconds", 30),
            )
            return True

    def cancel(self):
        with self.lock:
            self.observation = None

    def observe(self, snapshot, now=None):
        with self.lock:
            observation = self.observation
            now = self.clock() if now is None else now
            if observation is None:
                return None
            if (
                now > observation.expires
                or not self._valid_snapshot(snapshot)
                or snapshot["field_id"] != observation.field_id
            ):
                self.cancel()
                return None
            text = snapshot["text"]
            if text != observation.after:
                start, old_end, new_end = changed_range(observation.after, text)
                if start < observation.start or old_end > observation.current_end:
                    self.cancel()
                    return None
                observation.current_end += new_end - old_end
                observation.after, observation.changed_at = text, now
                return None
            if observation.changed_at is None or now - observation.changed_at < 3:
                return None
            candidate = correction_candidate(
                observation.before, observation.after, observation.start, observation.end
            )
            self.cancel()
            return self._suggest(candidate) if candidate else None

    def _suggest(self, candidate):
        with _STORE_LOCK:
            trigger = " ".join(candidate["heard"].casefold().split())
            if any(
                trigger == " ".join(alias.casefold().split())
                for entry in Dictionary.load()["replacements"]
                for alias in entry["triggers"]
            ):
                return None
            data = self._load()
            now = self.wall_clock()
            data["pairs"] = {
                key: value
                for key, value in data["pairs"].items()
                if now - value.get("updated", 0) <= 30 * 86400
            }
            key = candidate["heard"].casefold() + "\x1f" + candidate["corrected"].casefold()
            pair = data["pairs"].get(key, {**candidate, "occurrences": [], "dismissals": 0})
            pair["occurrences"] = [
                stamp for stamp in pair["occurrences"] if now - stamp <= 7 * 86400
            ]
            pair["occurrences"] = [*pair["occurrences"], now][-10:]
            pair["updated"] = now
            data["pairs"][key] = pair
            data["pairs"] = dict(
                sorted(data["pairs"].items(), key=lambda item: item[1]["updated"], reverse=True)[
                    :200
                ]
            )
            show = (
                len(pair["occurrences"]) >= getattr(self.config, "learning_min_occurrences", 2)
                and not pair.get("accepted")
                and pair.get("dismissed_until", 0) <= now
                and pair["dismissals"] < 3
                and now - data.get("last_shown", 0) >= 600
            )
            if show:
                data["last_shown"] = now
            self._save(data)
            if not show:
                return None
            proposal = {"id": str(uuid.uuid4()), **candidate, "key": key}
            self.proposals[proposal["id"]] = proposal
            while len(self.proposals) > 200:
                del self.proposals[next(iter(self.proposals))]
            if getattr(self.config, "learning_confirmation", "on") != "on":
                self.accept(proposal["id"])
                return {**proposal, "accepted": True}
            return proposal

    def pending(self):
        with self.lock:
            return list(self.proposals.values())

    def accept(self, proposal_id):
        with self.lock, _STORE_LOCK:
            proposal = self.proposals[proposal_id]
            data = Dictionary.load()
            trigger = proposal["heard"].casefold()
            for entry in data["replacements"]:
                if trigger in {alias.casefold() for alias in entry["triggers"]}:
                    if entry["replacement"] != proposal["corrected"]:
                        raise ValueError("Existing replacement conflicts with learned correction")
                    break
            else:
                data["replacements"].append(
                    {"triggers": [proposal["heard"]], "replacement": proposal["corrected"]}
                )
            # Validate final contents, including preserved vocabulary metadata,
            # before writing any store. Filesystem failures can still be partial.
            entries = VocabularyStore.entries()
            if proposal["corrected"].casefold() not in {
                entry["text"].casefold() for entry in entries
            }:
                entries.append({"text": proposal["corrected"], "weight": 1, "aliases": []})
            data = Dictionary.validate(data)
            entries = VocabularyStore.validate_entries(entries)
            policy = self._load()
            if not isinstance(policy["pairs"].get(proposal["key"]), dict):
                raise ValueError("Learning suggestion no longer exists")
            policy["pairs"][proposal["key"]]["accepted"] = True
            Dictionary.save(data)
            VocabularyStore.save_entries(entries)
            self._save(policy)
            del self.proposals[proposal_id]
            return proposal

    def dismiss(self, proposal_id):
        with self.lock, _STORE_LOCK:
            proposal = self.proposals.pop(proposal_id)
            policy = self._load()
            pair = policy["pairs"][proposal["key"]]
            pair["dismissals"] += 1
            pair["dismissed_until"] = self.wall_clock() + 7 * 86400
            self._save(policy)
