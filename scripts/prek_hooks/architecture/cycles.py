"""Optional package dependency cycles, independent of role policy cycles."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .policy import Diagnostic, Package

if TYPE_CHECKING:
    from .imports import Dependency


def check_cycles(
    dependencies: list[Dependency], classified: dict[str, Package]
) -> list[Diagnostic]:
    graph: dict[str, set[str]] = defaultdict(set)
    for dependency in dependencies:
        source = classified[dependency.importer].root
        if source != dependency.target_package:
            graph[source].add(dependency.target_package)
    visiting: list[str] = []
    visited: set[str] = set()
    diagnostics = []

    def visit(name: str) -> None:
        if name in visiting:
            chain = [*visiting[visiting.index(name) :], name]
            diagnostics.append(
                Diagnostic(
                    "architecture.toml",
                    0,
                    "architecture-package-cycle",
                    " -> ".join(chain),
                )
            )
            return
        if name in visited:
            return
        visiting.append(name)
        for target in sorted(graph.get(name, ())):
            visit(target)
        visiting.pop()
        visited.add(name)

    for name in sorted(graph):
        visit(name)
    return diagnostics
