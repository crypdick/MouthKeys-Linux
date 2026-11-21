"""Public API and CLI for the reusable Python architecture checker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .baseline import compare_baseline, load_baseline
from .budgets import (
    compare_root_module_inbound_import_baseline,
    root_module_inbound_importer_counts,
)
from .cycles import check_cycles
from .discovery import discover_modules
from .imports import (
    Violation,
    analyze_modules,
    collect_dependencies,
    dependency_violations,
)
from .policy import (
    Diagnostic,
    classify_modules,
    load_policy,
    policy_cycle_diagnostics,
    resolve_packages,
)
from .standalone import check_scripts


def check_architecture(
    root: Path,
    policy_path: Path,
    baseline_path: Path,
) -> tuple[list[Diagnostic], list[Violation]]:
    policy = load_policy(root, policy_path)
    modules = discover_modules(policy, root)
    packages, diagnostics = resolve_packages(modules, policy, policy_path)
    classified, classification_diagnostics = classify_modules(modules, packages)
    diagnostics.extend(classification_diagnostics)
    diagnostics.extend(policy_cycle_diagnostics(policy, policy_path))
    if diagnostics:
        return sorted(diagnostics), []
    facts, parse_diagnostics = analyze_modules(modules)
    diagnostics.extend(parse_diagnostics)
    diagnostics.extend(check_scripts(modules, facts, policy))
    dependencies, import_diagnostics = collect_dependencies(
        modules,
        facts,
        packages,
    )
    diagnostics.extend(import_diagnostics)
    if policy.check_package_cycles:
        diagnostics.extend(check_cycles(dependencies, classified))
    current = dependency_violations(dependencies, classified, packages, policy)
    baseline = load_baseline(baseline_path)
    diagnostics.extend(
        compare_baseline(
            current,
            baseline.violations,
            classified,
            policy,
            baseline_path,
        )
    )
    root_module_inbound_imports, root_module_diagnostics = (
        root_module_inbound_importer_counts(
            root,
            modules,
            dependencies,
            policy,
            policy_path,
        )
    )
    diagnostics.extend(root_module_diagnostics)
    diagnostics.extend(
        compare_root_module_inbound_import_baseline(
            root_module_inbound_imports,
            policy,
            baseline.root_module_inbound_imports,
            baseline_path,
        )
    )
    normalized = {
        Diagnostic(
            Path(diagnostic.path).relative_to(root).as_posix(),
            diagnostic.line,
            diagnostic.code,
            diagnostic.message,
        )
        if Path(diagnostic.path).is_relative_to(root)
        else diagnostic
        for diagnostic in diagnostics
    }
    return sorted(normalized), current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="architecture.toml")
    parser.add_argument("--baseline", default="architecture-baseline.toml")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        diagnostics, _current = check_architecture(
            root,
            root / args.policy,
            root / args.baseline,
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"architecture configuration error: {exc}")
        return 2
    for diagnostic in diagnostics:
        print(diagnostic.render())
    if diagnostics:
        print(f"\nFound {len(diagnostics)} architecture policy problem(s).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
