"""Exact, reviewed import-debt baseline comparison."""

from __future__ import annotations

import tomllib
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .policy import Diagnostic, Package, Policy, Role
from .schema import fields, module_name, positive, source_path

if TYPE_CHECKING:
    from pathlib import Path

    from .imports import Violation


@dataclass(frozen=True)
class BaselineEntry:
    importer: str
    importer_role: str
    target_role: str
    rule: str
    kind: str
    count: int
    imports: tuple[str, ...]
    reason: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.importer,
            self.importer_role,
            self.target_role,
            self.rule,
            self.kind,
        )


@dataclass(frozen=True)
class RootModuleInboundImportBaselineEntry:
    path: str
    count: int
    reason: str


@dataclass(frozen=True)
class ArchitectureBaseline:
    violations: tuple[BaselineEntry, ...]
    root_module_inbound_imports: tuple[RootModuleInboundImportBaselineEntry, ...]


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_baseline(path: Path) -> ArchitectureBaseline:
    if not path.exists():
        return ArchitectureBaseline((), ())
    raw = _load_toml(path)
    fields(raw, {"version", "violations", "root_module_inbound_imports"}, {"version"})
    if type(raw.get("version")) is not int or raw["version"] != 2:
        raise ValueError("architecture baseline version must be 2")
    for key in ("violations", "root_module_inbound_imports"):
        if not isinstance(raw.get(key, []), list):
            raise TypeError(f"{key} must be an array of tables")
        for entry in raw.get(key, []):
            expected = (
                {
                    "importer",
                    "importer_role",
                    "target_role",
                    "rule",
                    "kind",
                    "count",
                    "imports",
                    "reason",
                }
                if key == "violations"
                else {"path", "count", "reason"}
            )
            fields(entry, expected, expected)
            positive(entry["count"])
            if not isinstance(entry["reason"], str) or not entry["reason"].strip():
                raise ValueError("baseline entries require a nonempty reason")
            if key == "violations":
                module_name(entry["importer"])
                module_name(entry["importer_role"])
                module_name(entry["target_role"])
                if entry["kind"] not in {
                    "runtime-import",
                    "type-checking-import",
                } or entry["rule"] not in {
                    "visibility",
                    "direction",
                }:
                    raise ValueError("invalid baseline gate or import kind")
                if (
                    not isinstance(entry["imports"], list)
                    or len(entry["imports"]) != entry["count"]
                ):
                    raise ValueError("baseline count must match its exact imports")
                for imported in entry["imports"]:
                    module_name(imported)
            else:
                source_path(entry["path"])
    return ArchitectureBaseline(
        violations=tuple(
            BaselineEntry(
                importer=item["importer"],
                importer_role=item["importer_role"],
                target_role=item["target_role"],
                rule=item["rule"],
                kind=item["kind"],
                count=item["count"],
                imports=tuple(item["imports"]),
                reason=item.get("reason", "").strip(),
            )
            for item in raw.get("violations", ())
        ),
        root_module_inbound_imports=tuple(
            RootModuleInboundImportBaselineEntry(
                path=item["path"],
                count=item["count"],
                reason=item.get("reason", "").strip(),
            )
            for item in raw.get("root_module_inbound_imports", ())
        ),
    )


def _boundary_message(
    violation: Violation,
    importer_package: Package,
    target_package: Package,
    roles: dict[str, Role],
    default_guidance: str,
) -> str:
    importer_role = roles[importer_package.role]
    if violation.rule == "visibility":
        public = ", ".join(sorted(target_package.public_modules)) or "(none)"
        return (
            f"{violation.importer!r} ({importer_package.root}) imports private module "
            f"{violation.imported!r} from package {target_package.root!r} via "
            f"{violation.kind}; declared public modules: {public}. "
            f"To fix: import the package's declared facade. For a multi-module package, "
            f"expose names through {target_package.root}.api instead of making an "
            "implementation module public."
        )
    allowed = ", ".join(sorted(importer_role.allowed)) or "(none)"
    direction = roles[violation.target_role].violation_guidance or default_guidance
    return (
        f"{violation.importer!r} ({importer_role.name}) imports {violation.imported!r} "
        f"({violation.target_role}) via {violation.kind}; allowed roles: {allowed}. "
        f"To fix: {direction}"
    )


def _validate_baseline(
    baseline: tuple[BaselineEntry, ...],
    baseline_path: Path,
) -> tuple[dict[tuple[str, str, str, str, str], BaselineEntry], list[Diagnostic]]:
    entries: dict[tuple[str, str, str, str, str], BaselineEntry] = {}
    diagnostics: list[Diagnostic] = []
    for entry in baseline:
        if entry.key in entries:
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-baseline",
                    f"duplicate baseline entry {entry.key!r}",
                )
            )
        if (
            entry.rule not in {"direction", "visibility"}
            or entry.count < 1
            or entry.count != len(entry.imports)
            or not entry.reason
        ):
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-baseline",
                    (
                        f"entry {entry.key!r} needs a direction or visibility rule, "
                        "a positive count matching imports, and a nonempty reason"
                    ),
                )
            )
        entries[entry.key] = entry
    return entries, diagnostics


def compare_baseline(
    current: list[Violation],
    baseline: tuple[BaselineEntry, ...],
    classified: dict[str, Package],
    policy: Policy,
    baseline_path: Path,
) -> list[Diagnostic]:
    entries, diagnostics = _validate_baseline(baseline, baseline_path)
    grouped: dict[tuple[str, str, str, str, str], list[Violation]] = defaultdict(list)
    for violation in current:
        key = (
            violation.importer,
            classified[violation.importer].role,
            violation.target_role,
            violation.rule,
            violation.kind,
        )
        grouped[key].append(violation)
    packages_by_root = {package.root: package for package in classified.values()}
    roles_by_name = {role.name: role for role in policy.roles}

    for key, violations in grouped.items():
        expected = entries.get(key)
        remaining: defaultdict[str, int] = defaultdict(int)
        if expected:
            for imported in expected.imports:
                remaining[imported] += 1
        new_violations: list[Violation] = []
        for violation in violations:
            if remaining[violation.imported]:
                remaining[violation.imported] -= 1
            else:
                new_violations.append(violation)
        diagnostics.extend(
            Diagnostic(
                violation.path,
                violation.line,
                f"architecture-{violation.rule}",
                _boundary_message(
                    violation,
                    classified[violation.importer],
                    packages_by_root[violation.target_package],
                    roles_by_name,
                    policy.default_violation_guidance,
                ),
            )
            for violation in new_violations
        )
        missing_count = sum(remaining.values())
        if expected and missing_count:
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-baseline-stale",
                    (
                        f"{key!r} has {missing_count} recorded import(s) no longer in source; shrink the entry"
                    ),
                )
            )
    diagnostics.extend(
        Diagnostic(
            baseline_path.as_posix(),
            0,
            "architecture-baseline-stale",
            f"{key!r} no longer matches source; remove it",
        )
        for key in entries.keys() - grouped.keys()
    )
    return diagnostics
