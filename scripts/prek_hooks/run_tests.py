"""Run headless tests with coverage isolated from concurrent runs."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[2] / "linux"


def main() -> int:
    with TemporaryDirectory(prefix="mouthkeys-coverage-") as temporary:
        return subprocess.run(
            [sys.executable, "-m", "pytest", *sys.argv[1:]],
            cwd=PROJECT,
            env=os.environ
            | {
                "QT_QPA_PLATFORM": "offscreen",
                "COVERAGE_FILE": str(Path(temporary) / ".coverage"),
            },
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
