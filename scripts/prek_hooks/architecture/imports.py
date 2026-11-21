"""Whole-tree AST import extraction for the architecture boundary check."""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass

from .policy import (
    Diagnostic,
    Package,
    Policy,
    SourceModule,
    owning_package,
)


@dataclass(frozen=True)
class Dependency:
    importer: str
    imported: str
    target_package: str
    target_role: str
    kind: str
    path: str
    line: int


@dataclass(frozen=True)
class Violation:
    importer: str
    imported: str
    target_package: str
    target_role: str
    rule: str
    kind: str
    path: str
    line: int


_LOADERS = frozenset(
    {
        "__import__",
        "builtins.__import__",
        "importlib.import_module",
        "importlib.util.spec_from_file_location",
        "importlib.machinery.SourceFileLoader",
        "importlib.machinery.SourcelessFileLoader",
        "importlib.machinery.ExtensionFileLoader",
    }
)
_LITERAL_LOADERS = frozenset(
    {"__import__", "builtins.__import__", "importlib.import_module"}
)


@dataclass(frozen=True)
class ImportFacts:
    static: list[tuple[int, str]]
    loaders: list[tuple[int, str]]
    dependencies: list[tuple[str, str, int]]


def _bindings(tree: ast.AST) -> dict[str, set[str]]:
    # Conservatively retain all import bindings, independent of statement order
    # and lexical scope. Rebinding and runtime control flow are not interpreted.
    bindings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                target = alias.name if alias.asname else local
                bindings.setdefault(local, set()).add(target)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                bindings.setdefault(alias.asname or alias.name, set()).add(
                    f"{module}.{alias.name}"
                )
    return bindings


def _names(node: ast.expr, bindings: dict[str, set[str]]) -> set[str]:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, {node.id})
    if isinstance(node, ast.Attribute):
        return {f"{name}.{node.attr}" for name in _names(node.value, bindings)}
    return set()


def _resolve_from(module: SourceModule, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = module.name if module.is_package else module.name.rpartition(".")[0]
    relative_name = f"{'.' * node.level}{node.module or ''}"
    try:
        return importlib.util.resolve_name(relative_name, package)
    except (ImportError, ValueError):
        return None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


class _ImportVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: SourceModule,
        known_modules: set[str],
        first_party_roots: set[str],
        bindings: dict[str, set[str]],
    ) -> None:
        self.module = module
        self.known_modules = known_modules
        self.first_party_roots = first_party_roots
        self.dependencies: list[tuple[str, str, int]] = []
        self._type_checking_depth = 0
        self.bindings = bindings
        self.static: list[tuple[int, str]] = []
        self.loaders: list[tuple[int, str]] = []

    @property
    def _kind(self) -> str:
        return "type-checking-import" if self._type_checking_depth else "runtime-import"

    def _is_known(self, module: str) -> bool:
        return module in self.known_modules or any(
            name.startswith(f"{module}.") for name in self.known_modules
        )

    def _add(self, imported: str | None, line: int) -> None:
        if imported and imported.split(".", 1)[0] in self.first_party_roots:
            self.dependencies.append((imported, self._kind, line))

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking(node.test):
            self.generic_visit(node)
            return
        self.visit(node.test)
        self._type_checking_depth += 1
        for statement in node.body:
            self.visit(statement)
        self._type_checking_depth -= 1
        for statement in node.orelse:
            self.visit(statement)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name, node.lineno)
            self.static.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = _resolve_from(self.module, node)
        self.static.append((node.lineno, "." * node.level + (node.module or "")))
        targets = {base} if base else set()
        if base:
            submodules = {
                f"{base}.{alias.name}"
                for alias in node.names
                if alias.name != "*" and self._is_known(f"{base}.{alias.name}")
            }
            if submodules:
                targets = submodules
                if any(
                    alias.name == "*" or not self._is_known(f"{base}.{alias.name}")
                    for alias in node.names
                ):
                    targets.add(base)
        for target in sorted(targets):
            self._add(target, node.lineno)

    def visit_Call(self, node: ast.Call) -> None:
        loaders = _names(node.func, self.bindings) & _LOADERS
        self.loaders.extend((node.lineno, name) for name in sorted(loaders))
        if loaders & _LITERAL_LOADERS:
            target = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "name"
                    ),
                    None,
                )
            )
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                self._add(target.value, node.lineno)
        self.generic_visit(node)


def analyze_modules(
    modules: dict[str, SourceModule],
) -> tuple[dict[str, ImportFacts], list[Diagnostic]]:
    facts: dict[str, ImportFacts] = {}
    diagnostics: list[Diagnostic] = []
    first_party_roots = {name.split(".", 1)[0] for name in modules}
    for module in modules.values():
        try:
            tree = ast.parse(
                module.path.read_text(encoding="utf-8"), filename=str(module.path)
            )
        except (SyntaxError, UnicodeError, ValueError) as exc:
            diagnostics.append(
                Diagnostic(module.path.as_posix(), 0, "architecture-parse", str(exc))
            )
            continue
        visitor = _ImportVisitor(
            module, set(modules), first_party_roots, _bindings(tree)
        )
        visitor.visit(tree)
        facts[module.name] = ImportFacts(
            visitor.static, visitor.loaders, visitor.dependencies
        )
    return facts, diagnostics


def collect_dependencies(
    modules: dict[str, SourceModule],
    facts: dict[str, ImportFacts],
    packages: tuple[Package, ...],
) -> tuple[list[Dependency], list[Diagnostic]]:
    dependencies: list[Dependency] = []
    diagnostics: list[Diagnostic] = []
    for name, analysis in facts.items():
        module = modules[name]
        for imported, kind, line in analysis.dependencies:
            target_package = owning_package(imported, packages)
            if target_package is None or imported not in modules:
                diagnostics.append(
                    Diagnostic(
                        module.path.as_posix(),
                        line,
                        "architecture-import",
                        f"first-party import {imported!r} resolves to no package",
                    )
                )
                continue
            dependencies.append(
                Dependency(
                    importer=module.name,
                    imported=imported,
                    target_package=target_package.root,
                    target_role=target_package.role,
                    kind=kind,
                    path=module.path.as_posix(),
                    line=line,
                )
            )
    return dependencies, diagnostics


def dependency_violations(
    dependencies: list[Dependency],
    classified: dict[str, Package],
    packages: tuple[Package, ...],
    policy: Policy,
) -> list[Violation]:
    packages_by_root = {package.root: package for package in packages}
    roles = {role.name: role for role in policy.roles}
    violations: list[Violation] = []
    for dependency in dependencies:
        if dependency.importer in policy.composition_roots:
            continue
        importer_package = classified[dependency.importer]
        if dependency.target_package == importer_package.root:
            continue
        target_package = packages_by_root[dependency.target_package]
        if dependency.imported not in target_package.public_modules:
            violations.append(
                Violation(
                    **dependency.__dict__,
                    rule="visibility",
                )
            )
        if dependency.target_role not in roles[importer_package.role].allowed:
            violations.append(
                Violation(
                    **dependency.__dict__,
                    rule="direction",
                )
            )
    return sorted(
        violations,
        key=lambda item: (item.path, item.line, item.imported, item.kind, item.rule),
    )
