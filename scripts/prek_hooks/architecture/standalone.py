"""Optional standalone-script and stdlib-only import constraints."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .policy import Diagnostic, Policy, SourceModule

if TYPE_CHECKING:
    from .imports import ImportFacts


def check_scripts(
    modules: dict[str, SourceModule], facts: dict[str, ImportFacts], policy: Policy
) -> list[Diagnostic]:
    first_party = {name.split(".")[0] for name in modules}
    local_names = first_party | {module.path.stem for module in modules.values()}
    diagnostics: list[Diagnostic] = []
    for name, analysis in facts.items():
        standalone = name in policy.standalone_modules
        allow_dynamic = name in policy.dynamic_loading
        path = modules[name].path.as_posix()
        for line, target in analysis.static:
            top = target.split(".")[0]
            if (
                standalone
                and (top not in sys.stdlib_module_names or top in local_names)
            ) or (
                policy.stdlib_only
                and not target.startswith(".")
                and top not in sys.stdlib_module_names
                and top not in first_party
            ):
                diagnostics.append(
                    Diagnostic(
                        path,
                        line,
                        "architecture-script",
                        f"{target!r} is not an independent stdlib import -- keep scripts standalone; "
                        "test public entry points through the CLI or approved loader",
                    )
                )
        if not allow_dynamic and (standalone or policy.stdlib_only):
            diagnostics.extend(
                Diagnostic(
                    path,
                    line,
                    "architecture-dynamic",
                    f"{loader} loads code "
                    "-- use static stdlib imports; only registered test harnesses may load scripts",
                )
                for line, loader in analysis.loaders
            )
        elif allow_dynamic and not analysis.loaders:
            diagnostics.append(
                Diagnostic(
                    path,
                    0,
                    "architecture-stale",
                    "module no longer loads scripts -- remove its dynamic_loading exception",
                )
            )
    return diagnostics
