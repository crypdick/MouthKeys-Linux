"""Git-scoped source discovery independent of inherited hook context."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

from .policy import Policy, SourceModule

if TYPE_CHECKING:
    from pathlib import Path


def _excluded(relative_path: str, patterns: tuple[str, ...]) -> bool:
    return any(
        relative_path == pattern.removesuffix("/**")
        or (
            pattern.endswith("/**")
            and relative_path.startswith(f"{pattern.removesuffix('/**')}/")
        )
        for pattern in patterns
    )


def discover_modules(policy: Policy, root: Path) -> dict[str, SourceModule]:
    # Git hook context must not redirect a --root scan to the invoking repository.
    env = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
        ],  # noqa: S607 -- Git is required on PATH
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    files = sorted(
        {
            root / name
            for name in result.stdout.split("\0")
            if name and (root / name).exists()
        }
    )
    modules: dict[str, SourceModule] = {}
    for source_root in policy.source_roots:
        if (
            not source_root.path.is_dir()
            or not source_root.path.resolve().is_relative_to(root)
        ):
            raise ValueError(
                f"source root is missing or outside repository: {source_root.path}"
            )
        count = 0
        for path in files:
            if not path.is_relative_to(source_root.path):
                continue
            relative = path.relative_to(source_root.path)
            if _excluded(relative.as_posix(), source_root.exclude):
                continue
            if path.is_symlink() or path.resolve() != path:
                raise ValueError(f"symlinked Python source: {path}")
            is_package = path.name == "__init__.py"
            parts = (
                relative.parent.parts if is_package else relative.with_suffix("").parts
            )
            suffix = ".".join(parts)
            name = ".".join(part for part in (source_root.module, suffix) if part)
            if name in modules:
                raise ValueError(f"duplicate first-party module {name!r}")
            modules[name] = SourceModule(name=name, path=path, is_package=is_package)
            count += 1
        if not count:
            raise ValueError(
                f"source root discovers no Python modules: {source_root.path}"
            )
    return modules
