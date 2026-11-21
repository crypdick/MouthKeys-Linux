# Linux architecture map

[MouthKeys Linux](../linux/README.md#recording-and-architecture) is a local dictation app. The
installed entry point, `mouthkeys.cli.main`, parses arguments with `cli_options.build_parser`,
then dispatches local work or uses `control` to contact the private Unix socket served by
`daemon.Server`. The socket host owns lifecycle, not the recording queue.

- **Recording:** `service.Controller` owns capture and a FIFO of `DictationJob`s. `audio.CaptureSession`
  wraps the process owned by `capture`. `transcribe.Transcriber` serializes inference, with optional
  engines in `engines`, `native_asr`, and `native_worker`.
- **Delivery:** `formatting` and `spoken_send` transform text without desktop effects. `storage.History`
  persists text before `delivery.Delivery` attempts insertion. `accessibility` calls the isolated
  `atspi_helper`; `clipboard` can own a temporary paste selection in a separate Qt process.
  Cancellation before delivery wins; unverified destination identity falls back to copy.
- **UI and optional integrations:** `gui`, `overlay`, `tray`, and `hotkeys` call the service.
  `file_transcription_ui` owns the file worker and its history; Settings and the tray check whether
  it is busy before closing.
  `local_api` is opt-in loopback access. `ai_providers` and `ai_stream` handle configured AI
  requests only; imports, status, and Settings opening must not contact providers or download models.
- **State:** `config.Config.from_mapping` loads settings at the JSON boundary, including old files.
  `operations` prepares and applies backups. `upstream_backup` converts upstream dictionaries and
  prompt drafts; `upstream_backup_sections` parses settings and transcript records without writing
  state. `word_alignment`
  parses timed tokens into character-aware spans; character offsets are not time values.

[Import policy](../architecture.toml) keeps primitives independent and prevents the desktop
adapters it owns from importing application orchestration or sibling adapters. Application code
does not import development tooling or tests. Listed interfaces are internal to this repo, not
a promise of a public Python SDK. The remaining flat application owner intentionally permits its
existing internal imports. Avoid adding reverse edges when extracting a cohesive owner. The
whole-tree AST check also checks literal dynamic imports it recognizes; it does not infer
computed plugin names, subprocess behavior, or runtime library dependencies.
