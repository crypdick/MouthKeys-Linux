"""Fail on new Ruff/mypy diagnostics and on growth of existing oversized files."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from .check_file_length import count_logical_lines

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "linux"
BASELINE = ROOT / "scripts/prek_hooks/baselines"
TOOLS = {
    "ruff": ["ruff", "check", "--output-format=json", "src", "tests"],
    "mypy": ["mypy", "--output=json", "src/mouthkeys"],
}


def diagnostics(tool: str) -> tuple[Counter[str], dict[str, str]]:
    run = subprocess.run(
        TOOLS[tool], cwd=PROJECT, check=False, capture_output=True, text=True
    )
    if run.returncode not in (0, 1) or (run.returncode and not run.stdout):
        raise SystemExit(run.stderr.strip() or f"{tool} failed ({run.returncode})")
    measured: Counter[str] = Counter()
    locations: dict[str, str] = {}
    rows = (
        json.loads(run.stdout)
        if tool == "ruff"
        else [json.loads(line) for line in run.stdout.splitlines()]
    )
    for row in rows:
        if tool == "mypy" and row["severity"] == "note":
            continue
        path = Path(row["filename"] if tool == "ruff" else row["file"])
        if path.is_absolute():
            path = path.relative_to(PROJECT)
        line = row["location"]["row"] if tool == "ruff" else row["line"]
        key = f"{path}\t{row['code']}\t{row['message']}"
        measured[key] += 1
        locations[key] = f"linux/{path}:{line}: {row['code']} {row['message']}"
    if run.returncode and not measured:
        raise SystemExit(f"{tool} failed without diagnostics")
    return measured, locations


def file_lengths() -> tuple[Counter[str], dict[str, str]]:
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "linux/src",
            "linux/tests",
            "scripts/prek_hooks",
        ],
        cwd=ROOT,
        text=True,
    ).split("\0")
    measured: Counter[str] = Counter()
    locations: dict[str, str] = {}
    for name in sorted(set(paths)):
        path = ROOT / name
        if path.suffix != ".py" or not path.is_file():
            continue
        size = count_logical_lines(path)
        if size > 400:
            measured[name] = size
            locations[name] = (
                f"{name}:1: {size} lines of code (new files limited to 400)"
            )
    return measured, locations


def enforce(tool: str, *, prune: bool) -> int:
    path = BASELINE / f"{tool}.txt"
    allowed: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        count, key = line.split("\t", 1)
        allowed[key] = int(count)
    measured, locations = file_lengths() if tool == "file-length" else diagnostics(tool)
    new = measured - allowed
    for key, count in new.items():
        detail = f" (+{count} beyond baseline)" if key in allowed else ""
        print(f"{locations[key]}{detail}")
    if new:
        return 1
    if allowed != measured:
        if prune:
            path.write_text(
                "".join(f"{count}\t{key}\n" for key, count in sorted(measured.items())),
                encoding="utf-8",
            )
        else:
            print(f"{tool} debt decreased: rerun check_quality_budget {tool} --prune")
            return 1
    print(f"{tool}: no growth; {len(measured)} baseline entries")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=[*TOOLS, "file-length"])
    parser.add_argument(
        "--prune", action="store_true", help="Remove resolved debt; reject growth"
    )
    args = parser.parse_args()
    return enforce(args.tool, prune=args.prune)


if __name__ == "__main__":
    raise SystemExit(main())
