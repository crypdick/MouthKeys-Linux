"""Launch installed Python modules without importing from the caller's directory."""

import sys


def python_command(module: str = "mouthkeys", *args: str) -> list[str]:
    return [sys.executable, "-P", "-m", module, *args]
