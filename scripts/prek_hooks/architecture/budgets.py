"""Optional distinct-direct-importer budgets for root modules."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .policy import Diagnostic, Policy, SourceModule

if TYPE_CHECKING:
    from pathlib import Path

    from .baseline import RootModuleInboundImportBaselineEntry
    from .imports import Dependency


def root_module_inbound_importer_counts(
    root: Path,
    modules: dict[str, SourceModule],
    dependencies: list[Dependency],
    policy: Policy,
    policy_path: Path,
) -> tuple[dict[str, int], list[Diagnostic]]:
    source_root_paths = {source_root.path for source_root in policy.source_roots}
    root_modules = {
        module.path.relative_to(root).as_posix(): module
        for module in modules.values()
        if not module.is_package and module.path.parent in source_root_paths
    }
    diagnostics = [
        Diagnostic(
            policy_path.as_posix(),
            0,
            "architecture-policy",
            f"file override {source_path!r} must name an importable root-level module",
        )
        for source_path, _override in policy.file_overrides
        if source_path not in root_modules
    ]
    importers: dict[str, set[str]] = defaultdict(set)
    paths_by_module = {module.name: path for path, module in root_modules.items()}
    for dependency in dependencies:
        source_path = paths_by_module.get(dependency.imported)
        if source_path is not None:
            importers[source_path].add(dependency.importer)
    return {path: len(importers[path]) for path in root_modules}, diagnostics


def _validate_root_module_inbound_import_baseline(
    baseline: tuple[RootModuleInboundImportBaselineEntry, ...],
    baseline_path: Path,
) -> tuple[dict[str, RootModuleInboundImportBaselineEntry], list[Diagnostic]]:
    entries: dict[str, RootModuleInboundImportBaselineEntry] = {}
    diagnostics: list[Diagnostic] = []
    for entry in baseline:
        if entry.path in entries:
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-baseline",
                    f"duplicate root-module inbound-import entry {entry.path!r}",
                )
            )
        if entry.count < 1 or not entry.reason:
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-baseline",
                    (
                        f"root-module inbound-import entry {entry.path!r} needs a positive "
                        "count and a nonempty reason"
                    ),
                )
            )
        entries[entry.path] = entry
    return entries, diagnostics


def compare_root_module_inbound_import_baseline(
    current: dict[str, int],
    policy: Policy,
    baseline: tuple[RootModuleInboundImportBaselineEntry, ...],
    baseline_path: Path,
) -> list[Diagnostic]:
    entries, diagnostics = _validate_root_module_inbound_import_baseline(
        baseline,
        baseline_path,
    )
    if policy.root_module_max_inbound_importers is None:
        if entries or policy.file_overrides:
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-policy",
                    "inbound baseline or exemptions require an enabled importer budget",
                )
            )
        return diagnostics
    unbounded_paths = {
        source_path
        for source_path, override in policy.file_overrides
        if override.allow_unbounded_inbound_imports
    }
    current_violations = {
        source_path: count
        for source_path, count in current.items()
        if count > policy.root_module_max_inbound_importers
        and source_path not in unbounded_paths
    }
    for source_path, count in current_violations.items():
        expected = entries.get(source_path)
        if expected is None or count > expected.count:
            diagnostics.append(
                Diagnostic(
                    source_path,
                    0,
                    "architecture-root-module-inbound-importers",
                    (
                        f"root module has {count} direct inbound importers; limit is "
                        f"{policy.root_module_max_inbound_importers}"
                    ),
                )
            )
        elif count < expected.count:
            diagnostics.append(
                Diagnostic(
                    baseline_path.as_posix(),
                    0,
                    "architecture-baseline-stale",
                    (
                        f"root-module inbound-import entry {source_path!r} records "
                        f"{expected.count} importers but source now has {count}; shrink the entry"
                    ),
                )
            )
    for source_path in entries.keys() - current_violations.keys():
        diagnostics.append(
            Diagnostic(
                baseline_path.as_posix(),
                0,
                "architecture-baseline-stale",
                (
                    f"root-module inbound-import entry {source_path!r} no longer exceeds "
                    f"the {policy.root_module_max_inbound_importers}-importer limit; remove it"
                ),
            )
        )
    return diagnostics
