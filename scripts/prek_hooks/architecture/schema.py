"""Validate the configurable policy before constructing domain records."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def fields(value: object, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(
            f"expected fields {sorted(allowed)}, required {sorted(required)}; got {value!r}"
        )
    return value


def strings(value: object) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("expected a list of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError("duplicate list entries are not allowed")


def module_name(value: object, *, empty: bool = False) -> None:
    if empty and value == "":
        return
    if not isinstance(value, str) or not all(
        part.isidentifier() for part in value.split(".")
    ):
        raise ValueError(f"expected an exact dotted module name: {value!r}")


def source_path(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or Path(value).as_posix() != value
        or any(char in value for char in "*?[]\\")
    ):
        raise ValueError(f"expected an exact repository-relative path: {value!r}")


def positive(value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("expected a positive integer")


def validate_policy(raw: dict[str, Any]) -> None:
    fields(
        raw,
        {
            "version",
            "source_roots",
            "roles",
            "packages",
            "package_families",
            "composition_roots",
            "default_violation_guidance",
            "root_module_max_inbound_importers",
            "file_overrides",
            "check_package_cycles",
            "standalone_modules",
            "dynamic_loading",
            "stdlib_only",
        },
        {"version", "source_roots", "roles", "packages"},
    )
    if type(raw["version"]) is not int or raw["version"] != 2:
        raise ValueError("architecture policy version must be integer 2")
    for name in ("check_package_cycles", "stdlib_only"):
        if name in raw and type(raw[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if "root_module_max_inbound_importers" in raw:
        positive(raw["root_module_max_inbound_importers"])
    if not isinstance(raw.get("default_violation_guidance", ""), str):
        raise TypeError("default_violation_guidance must be a string")
    _validate_roots(raw["source_roots"])
    _validate_roles(raw["roles"])
    _validate_packages(raw["packages"], family=False)
    _validate_packages(raw.get("package_families", []), family=True)
    for key in ("composition_roots", "standalone_modules"):
        strings(raw.get(key, []))
        for name in raw.get(key, []):
            module_name(name)
    dynamic = raw.get("dynamic_loading", {})
    if not isinstance(dynamic, dict):
        raise TypeError("dynamic_loading must map exact module names to reasons")
    for name, reason in dynamic.items():
        module_name(name)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("dynamic_loading entries need a nonempty reason")
    _validate_overrides(raw.get("file_overrides", {}))


def _validate_overrides(overrides: object) -> None:
    if not isinstance(overrides, dict):
        raise TypeError("file_overrides must be a table")
    for path, value in overrides.items():
        source_path(path)
        fields(
            value,
            {"allow_unbounded_inbound_imports", "reason"},
            {"allow_unbounded_inbound_imports", "reason"},
        )
        if (
            value["allow_unbounded_inbound_imports"] is not True
            or not isinstance(value["reason"], str)
            or not value["reason"].strip()
        ):
            raise ValueError("inbound exemption requires true and a nonempty reason")


def _validate_roots(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("source_roots must be a nonempty list")
    for entry in value:
        item = fields(entry, {"path", "module", "exclude"}, {"path", "module"})
        source_path(item["path"])
        module_name(item["module"], empty=True)
        strings(item.get("exclude", []))
        for pattern in item.get("exclude", []):
            source_path(pattern.removesuffix("/**"))


def _validate_roles(value: object) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("roles must be a nonempty table")
    for name, entry in value.items():
        module_name(name)
        item = fields(
            entry,
            {"allowed_dependencies", "violation_guidance"},
            {"allowed_dependencies"},
        )
        strings(item["allowed_dependencies"])
        if not isinstance(item.get("violation_guidance", ""), str):
            raise TypeError("violation_guidance must be a string")


def _validate_packages(value: object, *, family: bool) -> None:
    if not isinstance(value, list):
        raise TypeError("packages and package_families must be arrays of tables")
    root_key = "root_pattern" if family else "root"
    required = {"name", root_key, "role", "public_modules"}
    for entry in value:
        item = fields(
            entry,
            required | ({"include_descendants"} if not family else set()),
            required,
        )
        for key in ("name", "role"):
            module_name(item[key])
        pattern = item[root_key]
        if family:
            if not isinstance(pattern, str) or not pattern.endswith(".*"):
                raise ValueError("family root_pattern must end in one-level .* ")
            module_name(pattern[:-2])
        else:
            module_name(pattern)
        strings(item["public_modules"])
        for public in item["public_modules"]:
            module_name(public.replace("{root}", pattern[:-2]) if family else public)
        if (
            "include_descendants" in item
            and type(item["include_descendants"]) is not bool
        ):
            raise ValueError("include_descendants must be boolean")
