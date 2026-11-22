# Linux architecture map

[MouthKeys Linux](../linux/README.md#recording-and-architecture) is a local dictation app. The
installed entry point, `mouthkeys.cli.main`, parses arguments with `cli_options.build_parser`,
then dispatches local work or uses `control` to contact the private Unix socket served by
`daemon.Server`. The socket host owns lifecycle, not the recording queue. It sends the shutdown
reply before stopping the server.

- **Recording:** `service.Controller` accepts commands and keeps a FIFO of `DictationJob`s.
  `recording_capture` owns microphone start, switch, and stop, and hands stopped jobs to the
  finalizer. `audio.CaptureSession` preserves the PCM prefix across source switches. `capture_recorder` owns the FFmpeg lifeline,
  and `capture_readiness` checks when a Bluetooth loopback has a physical source.
  `microphone` owns session source preferences, retry
  budgets, and device quarantine. Capture transitions use the shared recording state lock.
  `recording_setup` freezes per-recording settings and prompt rules before the controller
  supersedes any pending desktop action. A rejected start leaves the current job in place.
  `recording_destination` samples focus and formatting context before capture stops and looks up
  metadata afterward, without activating another window. Automatic stop does not wait on focus
  in the daemon loop; an expired focus snapshot falls back to copy. `recording_preview` meters
  live audio and tracks pauses; stale decodes cannot publish after a microphone switch.
  `recording_inference` validates capture and serializes final decoding; the controller
  commits results only if cancellation or logout has not won. `transcribe.Transcriber`
  uses engines in `engines` and `native_worker`. `transcript_window` commits stable preview
  segments only after the whole decode succeeds and leaves the unresolved tail for finalization.
  `native_asr` owns local artifact checks and audio helpers. `native_worker` manages the offline
  process; `native_protocol` bounds its framed JSON and PCM transfers, including startup.
  `transformers_asr` and `nemo_asr` load and decode inside that process.
- **Benchmarks:** `benchmark` measures explicit WAV fixtures in an offline worker.
  `benchmark_reports` validates and retains bounded local reports for the model catalog.
- **Delivery:** `formatting` and `spoken_send` transform text without desktop effects. `storage.History`
  persists text before `delivery.Delivery` attempts insertion. `accessibility` calls the isolated
  `atspi_helper`, which checks the field role before reading text and revalidates text, selection,
  and focus before returning a snapshot. `clipboard` can own a temporary paste selection in a
  separate Qt process.
  `desktop_input` probes compositor window identity and runs clipboard and keyboard commands.
  `delivery` decides whether to use those commands and rechecks the destination before sending.
  Cancellation before delivery wins; unverified destination identity falls back to copy.
  `recording_transcript` prepares dictated text for delivery: pronunciation, Spoken Send parsing,
  and local formatting. `spoken_punctuation` matches prefixed dictionary phrases and renders
  generated symbols without rewriting unrelated ASR punctuation. `pronunciation_spans` bounds
  acoustic search and applies only completed,
  disjoint matches to observed word timings. `recording_cleanup` runs configured AI cleanup with
  the recording's cancellation event.
  `recording_finalizer` runs stopped jobs through recognition, review or dictation, and retirement.
  It and the controller share one `recording_state` lock for job and UI transitions.
  `recording_delivery` receives committed text and owns the cancellable Spoken Send review
  window. The key reservation uses the same lock as cancel and new-recording commands.
  Manual replay checks ownership again before requesting a paste; it cannot update a newer
  recording's status.
  `recording_edit` validates captured selection evidence and reserves replacement once for
  reviewed and automatic Edit. A newer recording cancels a replacement still in progress.
  `recording_learning` observes confirmed insertions and coordinates review without holding
  the controller state lock during AT-SPI reads or dictionary writes.
- **Command line:** `cli_options` defines argument grammar; `cli` dispatches commands and keeps
  results on stdout and diagnostics on stderr. `desktop_entry` owns XDG launcher and login files
  shared by CLI, Settings, and backup restore.
- **UI and optional integrations:** `gui`, `overlay`, `tray`, and `hotkeys` call the service.
  `tray_status` interprets service replies and deduplicates alerts independently of Qt.
  `lark_monitor` polls the opted-in input and discards stale results; only the isolated
  `lark` helper opens the receiver and sends its fixed status heartbeat.
  `x11_shortcuts` owns the native X11 connection and passive grabs. `hotkeys` serializes
  their events and releases an owned held recording when the listener stops. The Wayland
  listener (`portal_shortcuts`) reads events from a separate GI helper (`portal_helper`),
  which owns portal consent, the granted bindings, and session teardown.
  `overlay_process` owns the passive panel's lifetime and wake socket; `overlay_rendering` draws
  its current frame without controlling focus, placement, or IPC. `qt_control` sends their
  control requests without importing Settings or blocking the Qt event loop.
  `settings_form` owns Settings editors, their categories, and live hints. `settings_choices`
  populates model and microphone selectors without loading models or opening capture. The main
  window owns persistence and recording-service requests; `gui_launch` starts it and the
  configured desktop integrations. `provider_settings_ui` owns explicit provider and keyring
  actions. `regional_ui` shares the same save path between the inline banner and Settings prompt.
  `assistant_chats` handles conversation selection and history, including an opt-out during a
  pending request. `assistant_session` validates and bounds messages and execution receipts for
  both private memory and history. `assistant_history` commits saved conversations in SQLite
  transactions. `assistant_editor` displays proposals and forwards explicit review actions.
  `recording_assistant` binds voice jobs to their starting conversation, retains execution
  receipts, and publishes bounded status snapshots. Only the active request round can publish
  incremental output. `command_preview` decodes only the root explanation for display;
  `assistance` validates the final proposal separately before it can be acted on.
  Transcript history (`history_ui`), activity (`stats_view`), backup review (`backup_ui`), and
  diagnostics (`help_ui`) own their Settings pages. `dictionary_ui` owns editing and voice training
  drafts; saving compares the editor's snapshot under the same lock used to write the dictionary.
  `learning_ui` sends take controls for its acknowledged recording only. Vocabulary writes compare
  the editor snapshot under the same lock used by automatic learning.
  Settings waits for backup previews and reviews before closing. `prompt_profiles_ui` owns the
  profile editor, application overrides, and draft prompt tests. Recorded tests keep the draft
  that was active when recording started and never enter the desktop delivery path.
  `model_downloads_ui` owns the model download process, cache actions, and progress display.
  It keeps child-process output bounded and requests model warmup only after a successful download.
  `model_cache` resolves local revisions and restricts deletion to the selected app-owned cache.
  `file_speakers` stages explicit model downloads and validates attributed intervals against the
  decoded file.
  `file_transcription_ui` owns the file worker and its history; Settings and the tray check whether
  it is busy before closing.
  `local_api` owns authenticated loopback routing and releases inference slots before sending
  responses. `api_dictionary` commits dictionary updates under the shared lock.
  `ai_provider_settings` validates provider routes without network access.
  `provider_tools` discovers models only on request, paging on the same origin and keeping
  deployment query parameters. `ai_providers` and `ai_stream` handle configured AI requests only.
  `ai_stream_events` normalizes provider frames before callbacks; the transport
  discards frames read after cancellation or timeout. Imports, status, and Settings opening must
  not contact providers or download models.
- **State:** `config.Config.from_mapping` loads settings at the JSON boundary, including old files.
  `operations` dispatches history and backup actions; `statistics` summarizes history without
  writing to it. `backup` validates a restore plan before applying it. `upstream_backup` converts
  upstream dictionaries and prompt drafts; `upstream_backup_sections` parses settings and
  transcript records without writing state. `private_files` atomically replaces private files.
  `forced_alignment` adds optional cached Whisper timings only to validated ASR segments.
  `word_alignment` anchors timed tokens to character spans; character offsets are not time values.

[Import policy](../architecture.toml) keeps primitives independent and prevents the desktop
adapters it owns from importing application orchestration or sibling adapters. Application code
does not import development tooling or tests. Listed interfaces are internal to this repo, not
a promise of a public Python SDK. The remaining flat application owner intentionally permits its
existing internal imports. Avoid adding reverse edges when extracting a cohesive owner. The
whole-tree AST check also checks literal dynamic imports it recognizes; it does not infer
computed plugin names, subprocess behavior, or runtime library dependencies.
