# MouthKeys for Linux

This checkout is `crypdick/MouthKeys-Linux`, Ricardo Decal's Linux fork. Runtime, tests, packaging,
and the lockfile live in `linux/`. Read `linux/README.md` for architecture, delivery limits,
installation, and manual desktop validation. `AGENTS.md` links to this file.

## Validation

Run `./linux/check.sh`, which performs:

```sh
uv sync --project linux --python 3.13 --extra gui --locked
uv run --project linux ruff check linux
uv run --project linux ruff format --check linux
QT_QPA_PLATFORM=offscreen uv run --project linux pytest linux/tests
uv build linux
```

Tests must stay headless: fake capture/clipboard/key adapters, isolated XDG directories, and Qt's
offscreen backend. Never use a live microphone, touch the real clipboard, synthesize keys, show
windows, take focus, or play sounds in tests. Give waits bounded timeouts. Real compositor focus,
microphone capture, and paste behavior require the manual checklist in `linux/README.md`.

## Product rules

Target upstream MouthKeys feature parity, adapting platform integrations for Linux. Keep local
dictation available, telemetry off, and update checks off. Downloads happen only on explicit
request. Cloud AI, Spoken Send, media controls, and other optional behavior must be configurable;
do not invent prohibitions on these features. CLI/i3 shortcut bindings are sufficient; integrated
shortcut management is lower priority. Clipboard delivery is the default. Automatic delivery and
Spoken Send require explicit configuration and destination/focus checks supported by the desktop.
Never silently activate another window. Preserve Ghostty/terminal clipboard paste behavior. Cancel before delivery
must suppress late delivery. Keep recovery audio bounded and private, and persist transcripts before
desktop delivery unless the user explicitly disables history (then use memory and temporary audio). Expose configuration in App Settings as well as the CLI.

## Git and delivery

`origin` is the personal Linux fork. `mouthkeys-upstream` is `BenevolentFutures/MouthKeys`.
PRs, when requested, **must** target `crypdick/MouthKeys-Linux --base main`. Never publish personal
changes to the parent project or its website. Upstream changes are adapted by hand; see `UPSTREAM.md`.

Carry authorized changes through branch, build, tests, fresh-context review, merge to `main`,
and push. The user prefers direct merges without PRs. Ask only for decisions or desktop checks
that require the user.
macOS code and tooling are maintained upstream.
