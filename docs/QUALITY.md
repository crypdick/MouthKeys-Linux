# Quality gates

`./linux/check.sh` runs the full prek suite and builds the Linux distributions. CI uses this same
command. The package manager locks development tools, and applies a three-day cooldown when
resolving new dependencies. `new-feature create NAME --no-agent` syncs a private worktree venv
from the lockfile. Tests use per-test XDG paths, offscreen Qt, and bounded waits; setup does not
start a daemon. CI installs FFmpeg for the capture tests that read synthetic PCM files without
opening a microphone. If manually running the app from a feature worktree, point all XDG homes and
`XDG_RUNTIME_DIR` at private directories to avoid sharing the installed app's daemon and state.

Run `prek install` from the control checkout to enable commit checks. Use a prek executable
available outside disposable feature worktrees. To run a single gate:
`uv run --project linux --locked prek run HOOK --all-files`.

## Required checks

- **Types and lint:** Strict mypy checks production code. Ruff checks production and tests;
  cyclomatic complexity is limited to 15. The tracked [baselines](../scripts/prek_hooks/baselines)
  are empty: fix new diagnostics rather than adding exceptions to the baselines.
- **Size:** Python files are limited to 400 code-containing lines, with no grandfathered files.
  Extract complete workflows without changing locking, cancellation, or delivery order.
- **Coverage:** The headless suite requires 100% line and branch coverage, including Python
  subprocesses started by the tests. Desktop and optional pretrained-model checks remain separate; see
  [the parity ledger](../linux/PARITY.md). Don't count manual checks as automated coverage.
- **Runtime:** Beartype checks annotated parameters and returns. Keep plain JSON/IPC validation:
  runtime annotations are not complete container checks. System GI and model SDKs not installed
  in the headless environment are scoped mypy exceptions, not blanket import suppression.

## Conventions

Parse external values at their boundary and pass typed results inward. Distinguish values that
are easy to confuse in code (e.g. `CharacterOffset` vs. seconds in `word_alignment`); do not add
wrapper types for their own sake. Keep work with its owner, and prefer a composed collaborator
over an inheritance or naming workaround. Keep the [architecture map](ARCHITECTURE.md) current
when dependency direction changes.

Prefer structured logging for operational diagnostics. Preserve CLI and subprocess-protocol stdout.
If a boundary must catch broadly to return a user-facing error or fail safely, explain the
exception locally. Never duplicate transcript, clipboard, credential, or provider payloads in logs.
