# MouthKeys for Linux

Local dictation for Linux in [crypdick/MouthKeys-Linux](https://github.com/crypdick/MouthKeys-Linux).
Record from a shortcut, transcribe on CPU or NVIDIA GPU, and paste into an editor or terminal.
Local dictation needs no account. Cloud/local AI cleanup, Spoken Send, learning, media controls,
and automatic insertion are optional. Telemetry and update checks stay off.

The runtime is Python/Qt. Linux-native speech and desktop adapters replace upstream Apple APIs;
the macOS application remains upstream. [PARITY.md](PARITY.md) is the canonical acceptance ledger:
implementation and headless checks are distinct from real desktop, hardware, and pretrained-model
validation. This guide describes supported configuration, not a declaration of complete parity.

## Install

Python 3.11–3.14, FFmpeg with PulseAudio input, and PulseAudio or PipeWire's PulseAudio compatibility
service are required. Python 3.13 is the tested version; optional native engines have additional
package/hardware requirements.

Ubuntu/Debian desktop dependencies:

```sh
sudo apt install ffmpeg wl-clipboard libxcb-cursor0 libxkbcommon-x11-0
# X11 clipboard/key delivery:
sudo apt install xclip xdotool x11-utils
# Optional AT-SPI field integration / portal helpers:
sudo apt install python3-gi gir1.2-atspi-2.0
```

Native Transformers/NeMo isolation also requires `libseccomp2`. Optional media/sounds use `playerctl`
and `paplay` (`pulseaudio-utils`). Fedora/Arch package names differ; `mouthkeys doctor` reports missing
command-line tools. Native Wayland paste needs an explicitly configured backend; see [Text delivery](#text-delivery).

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/crypdick/MouthKeys-Linux.git
cd MouthKeys-Linux
uv tool install --python 3.13 './linux[gui]'
mouthkeys install-desktop
mouthkeys download
mouthkeys doctor
mouthkeys gui
```

Put `~/.local/bin` on PATH (`uv tool update-shell`). Omit `[gui]` for terminal-only use. Setup guides
model/microphone selection and provides a non-delivering dictation tryout. Choosing a model or opening
settings never downloads its weights: use **Download selected model** or `mouthkeys download`.

### Update and check your version

App Settings shows application and running-service versions; a mismatch requires a manual restart.
Updating the package does not replace a running daemon. Finish dictation and close the GUI before:

```sh
mouthkeys shutdown
git pull --ff-only
uv tool install --force --python 3.13 './linux[gui]'
mouthkeys gui
```

Preserve the extras you actually use when reinstalling. Configuration, history, and models survive.
No code update or test installs over or restarts the running application automatically.

To uninstall, stop the service yourself, remove the MouthKeys autostart/desktop entries, and run
`uv tool uninstall mouthkeys-linux`. This leaves configuration, models, history and learned data
intact. Delete only the MouthKeys XDG directories separately if you want to erase that data.

## Set up your shortcut

Desktop/i3 bindings are the default; no global grab or privileged input service is required.
Bind an absolute executable path from `command -v mouthkeys`:

| Action | Command | Example |
| --- | --- | --- |
| Start/stop | `mouthkeys toggle` | Super+Alt+Space |
| Cancel | `mouthkeys cancel` | Super+Alt+Escape |
| Paste latest now | `mouthkeys paste` | Separate shortcut |
| Retry retained audio | `mouthkeys reprocess` | Separate shortcut |

Sway/i3 examples, replacing the path where necessary:

```text
bindsym $mod+Mod1+space exec ~/.local/bin/mouthkeys toggle
bindsym $mod+Mod1+Escape exec ~/.local/bin/mouthkeys cancel
```

Separate press/release bindings may invoke `mouthkeys start` / `mouthkeys stop`. CLI commands alone
cannot know whether a shortcut key remains physically held. Prompt-specific bindings can invoke
`mouthkeys toggle --profile PROFILE_UUID`; Command/Edit use `--mode command` or `--mode edit`.

Optional integrated shortcuts are explicit: set `shortcut_backend` to `x11` or `portal`, then use
**Start configured shortcuts** or `mouthkeys hotkeys`. X11 uses passive keyboard/mouse grabs;
`button4`, for example, names a mouse button. Comma-separated chords allow multiple bindings.
`hotkey_mode` is `toggle`, `hold`, or `automatic` (short tap toggles; holding at least 250 ms stops
on release). Conflicting bindings fail visibly. Shortcut capture is a deliberate Settings action.
The GlobalShortcuts portal supports keyboard chords and compositor-managed consent; unsupported
or denied portals do not pretend to work. Return to desktop-owned bindings with `shortcut_backend=desktop`.

The first recording command starts a private daemon. `mouthkeys warmup` loads the selected model.
Closing Settings leaves recording available. `launch_at_startup=on` creates a user autostart entry;
`show_window_at_login` chooses Settings versus tray/daemon. Changing the session type or service
environment requires a manual restart.

## Recording and architecture

1. Focus the destination and invoke the recording shortcut.
2. Speak when the panel shows Listening, then stop with the shortcut.
3. Clipboard mode copies the result; paste normally with Ctrl+V or terminal Ctrl+Shift+V.

The service owns capture, a serialized model worker, immutable per-recording settings, and FIFO
jobs. Capture may continue for a newer recording while older jobs transcribe/deliver. `queue_limit`
bounds pending work; cancellation of a newer job does not cancel an earlier dictation. Live words
and energy belong to the current recording, not the previous queued job. Cancel before delivery
suppresses late clipboard/key actions; once delivery starts, cancellation cannot withdraw inserted
text. The separate half-second Spoken Send window remains cancellable.

Capture uses FFmpeg/PulseAudio. The timer starts after the first audio buffer, and Bluetooth
PipeWire inputs additionally wait for their physical capture node (`pw-dump` when available).
Readiness must complete within five seconds; listening does not wait for speech. Metering uses
actual 20 ms energy buffers, independent of inference. Default recording limit is five minutes;
a duration-limit stop copies without selecting a new automatic-paste destination.

`preview_interval_ms` defaults to 2000. Whisper/provisional-window engines retain completed segments
ending at least two seconds before the snapshot and decode forward from the uncommitted boundary.
Unresolved windows grow rather than skip speech. Stop reuses a useful in-flight/cached tail, without
re-decoding committed text. Boundary accuracy can differ from a whole-recording retry; Reprocess
starts fresh. NeMo's cache-aware path instead advances native encoder/decoder caches; its frontend
still recomputes the observed prefix for correct STFT boundaries. Native inference cannot always be
interrupted immediately; cancelled work must finish or time out, with delivery suppressed and caches
reset before reuse. `native_timeout_seconds` bounds the isolated native worker.

The overlay is a separate optional Qt process, intended not to take focus. It shows provisional text,
waveform, timer, words/WPM, microphone, and optional battery. Regular/compact, top/bottom, size,
offset, preview length, border, sensitivity, and theme are configurable. History offers inline
copy/reinsert/reprocess; failure recovery offers copy/retry. Drag to move; double-click to reset.
Compositors control placement/stacking. Disable with `recording_panel=off` if it interferes with focus.

## Text delivery

**Clipboard is the default.** Dictation text replaces clipboard contents and remains available for
manual paste. Transcripts are saved before delivery unless history is explicitly disabled. Terminal
text always uses clipboard paste, never simulated character typing.

Automatic delivery requires explicit `delivery=x11` or `delivery=wayland`. `target_policy=stop`
captures the destination at explicit stop; `target_policy=start` captures at recording start.
`return_to_start=on` also captures the starting destination and explicitly permits returning to it.
Without that setting, the application never activates another window. Changed or unidentifiable
destinations fall back to copy.

| Available desktop evidence | Automatic behavior and limit |
| --- | --- |
| AT-SPI editable field | Stable D-Bus object identity, foreground state, text, caret, and selection checks; verified insertion where the toolkit exposes them. Password fields are excluded. |
| X11 without usable AT-SPI | Focused X-window continuity; cannot distinguish tabs/panes/fields within that window. |
| Sway/Hyprland without usable AT-SPI | Compositor container/window identity; cannot distinguish fields inside it. |
| Other Wayland without usable AT-SPI | Clipboard-only fallback; no generic fabricated destination. |

AT-SPI depends on toolkit/accessibility support. Helpers run out of process with bounded timeouts
and may use distro Python/GI when uv Python lacks GI. Foreground field checks are not a universal
atomic compare-and-swap: validate the actual destination application. Large fields use bounded
snapshot windows; native direct replacement and correction learning require complete snapshots.

`insertion_mode=direct` prefers clipboard-free AT-SPI insertion into an unchanged nonterminal field;
X11 can also use guarded direct typing when window class positively identifies a nonterminal.
Control characters, terminals, unsupported fields, or failed checks fall back to clipboard delivery.
`copy_to_clipboard` controls the additional copy following direct insertion. `edit_delivery=replace`
permits replacement only of the exact captured selection, otherwise output remains reviewable/copied.

X11 example:

```sh
mouthkeys config delivery x11
mouthkeys config paste_shortcut ctrl+v   # ctrl+shift+v remains the terminal-oriented default
```

Wayland clipboard paste needs an existing compatible virtual-key backend:

```sh
mouthkeys config paste_backend wtype    # e.g. Sway; not supported by every compositor
# Or an already configured, user-accessible ydotool daemon:
mouthkeys config paste_backend ydotool
mouthkeys config delivery wayland
```

MouthKeys does not install a privileged ydotool service or change device permissions. The daemon
must inherit `YDOTOOL_SOCKET` when required. `mouthkeys paste` is an explicit focus-now shortcut:
it copies the latest result and sends the paste key to current focus, without reviving an old target.
Do not invoke that action from Settings while expecting another window to receive it.

`verify_paste=on` compares actual field contents after insertion where available. `pasted` means
that field accepted the expected text; `paste_requested`/`insert_requested` mean a request only.
With `restore_clipboard=on`, a separate Qt owner snapshots up to 32 MiB of MIME formats. Restoration
requires retained ownership, so a newer user copy wins. Field-aware paste restores only after
verified consumption; window-only fallback uses the configured delay (`clipboard_restore_delay_ms`).
Unsupported snapshots fall back to ordinary copy. `clipboard_exclude_history=on` supplies an advisory
secret/history-exclusion hint on supported temporary clipboard paths; clipboard managers may ignore it.
Neither hint nor restoration guarantees removal from a manager's existing history.

### Spoken Send

Off by default. Enable `spoken_send`, choose `spoken_send_phrase` (default `send it`), and
`spoken_send_key` (`return`, `shift+return`, `ctrl+return`). A trailing command phrase is removed;
sentence-owned phrases such as “I will send it” are not commands. Use the literal prefix to dictate
the phrase. After supported insertion, a half-second countdown permits cancelling Send without
withdrawing text already inserted. Terminals always receive plain Return.

`send_after_pause=on` allows a trailing phrase plus half a second of quiet to stop and request Send.
More speech or the panel control disarms it. Integrated hold state suppresses automatic quiet-stop
while the activating gesture remains held. Starting a newer recording suppresses an older pending
Send. Supported field/window/compositor continuity is checked again before the key; unsupported
Wayland destinations, clipboard-only delivery, and duration-limit stops do not send.

## Models, microphones, and recovery

App Settings exposes every Config option in categories; `mouthkeys config` lists current values.
Model/language/device/precision/microphone controls also accept supported custom paths/names. A model
choice does not download or warm it automatically. The model browser filters/sorts installed state,
language, and catalog metadata, and offers explicit download/activate/delete. Its resident-model
label/row indicator uses service-reported engine/model/revision/device/precision, not the current
Settings selection. Unknown RAM, speed, and accuracy are not invented; weight bytes are not runtime memory. Progress/cancel use the Hub
transfer callback where supported. A partial download is not advertised as a ready native/ONNX model.

### Speech engines

| `engine` / extra | Available integration | Important limits |
| --- | --- | --- |
| `whisper` / default | faster-whisper, CPU/CUDA, incremental provisional text | Whisper word timings enabled for pronunciation matching; actual checkpoint quality varies. |
| `onnx` / `[onnx]` | Parakeet TDT v2/v3, Canary 1B v2 / 180M Flash | Offline VAD chunks, not native streaming; no onnx-asr hotword decoder. |
| `transformers` / `[transformers]` | Nemotron 3.5/English, Qwen3-ASR 0.6B/1.7B `-hf`, Cohere Transcribe | Builtin Transformers classes, provisional inference; no downloaded Python model code. Cohere needs explicit language and gated access where required. |
| `nemo` / `[nemo]` | Parakeet Realtime EOU 120M (Flash equivalent), Nemotron 3.5 | Native cache-aware streaming; model-specific language/latency limits. Realtime EOU English output is uncased/unpunctuated. |

Install only needed extras, after closing the application/service yourself. CPU native example
with a uv version supporting `--torch-backend` (verified with uv 0.13.5):

```sh
uv tool install --python 3.13 --torch-backend cpu './linux[gui,transformers,nemo]'
# ONNX and optional speaker labels instead:
uv tool install --python 3.13 './linux[gui,onnx,diarization]'
```

For CUDA, use compatible PyTorch/CTranslate2/ONNX Runtime builds and NVIDIA libraries; selecting
`device=cuda` cannot supply those system components. Native engines support default/auto/float32/
float16/bfloat16; cache-aware NeMo streaming requires float32. ONNX supports default/auto/float32/int8.
Whisper `compute_type=default` follows stored precision with hardware fallback; `auto` chooses a fast
supported type, and `int8` may reduce CPU memory. Existing precision choices are preserved.

```sh
mouthkeys config engine onnx
mouthkeys config model nemo-parakeet-tdt-0.6b-v3
mouthkeys config device cpu
mouthkeys config compute_type default
mouthkeys download
mouthkeys warmup
mouthkeys models
```

Native examples use `engine=nemo` with `nvidia/parakeet_realtime_eou_120m-v1`, or
`engine=transformers` with `Qwen/Qwen3-ASR-0.6B-hf`. Select the model before explicitly downloading.
`model_revision` pins Hub artifacts; native caches distinguish revisions. Cohere's catalog default
uses compatibility revision `refs/pr/6`; changing that choice requires a compatible checkpoint.
Gated downloads require accepting the publisher's terms and supplying `HF_TOKEN` to the process
performing the explicit download. Missing extras/artifacts fail with setup instructions, not hidden
fetches. Native runtime uses private pipe workers with Internet sockets denied by libseccomp, plus
offline library settings; local Unix IPC remains available. No account is needed for cached local use.

`native_streaming=on` selects supported NeMo cache-aware inference. `streaming_latency` is `model`
or a supported 80/160/320/560/1120 ms context preset, not a guaranteed end-to-end latency.
`asr_max_tokens` bounds native generation; `native_timeout_seconds` bounds worker operations.
`mouthkeys delete-model MODEL` removes only app-owned cache artifacts for the selected engine/revision,
never a user-entered local directory. Deleting cache does not unload a resident model; warm another
model or restart the service yourself.

### Microphones, media, and appearance

The Microphone page edits ordered Preferred, fallback, and Never groups (drag or move buttons),
shows capture's actual meter, and selects sound output. CLI equivalents are comma-separated source
names in `microphone_preferred`, `microphone_fallback`, and `microphone_never`. Explicit source
selection never overrides Never. Choose source names from `pactl list short sources`; `.monitor`
records system output rather than your microphone. Device selection is resolved before capture;
an unplugged active device is reported, not silently spliced to a different microphone mid-recording.

`pause_media=on` uses `playerctl`/MPRIS, resuming only players this session paused and only if still
paused. `sound_preset=none` is silent; `mouthkeys` uses bundled upstream start/stop cues; `custom`
uses `sound_file`/`sound_stop_file`. Both custom paths default empty, so there are no sounds by default.
Preview buttons are explicit playback actions. `sound_volume`/`sound_output` choose volume/output.

`mic_battery=on` polls supported Hollyland Lark A1 receiver status only when it is the selected
capture source. The Linux adapter checks vendor/product and HID status-report descriptor, sends
only the fixed heartbeat, and reports unavailable/permission-denied rather than changing permissions.
Receiver access, firmware behavior, and actual battery readings require hardware validation.

Tray hosting varies by desktop (GNOME may need an AppIndicator extension). `tray=on` enables it;
Quit tray exits the UI, not the recording daemon. Active GUI workers/file transcription guard quit.
`show_microphone_alerts`/`show_paste_alerts` control notifications. Theme is system/light/dark.
Help includes a bounded diagnostic-log viewer and a prefilled issue draft; it never sends feedback
without the user submitting it.

### History and private recovery

History is in `$XDG_STATE_HOME/mouthkeys` (normally `~/.local/state/mouthkeys`), configuration in
`$XDG_CONFIG_HOME/mouthkeys`, models in `$XDG_CACHE_HOME/mouthkeys`. The user-private control socket
uses `$XDG_RUNTIME_DIR/mouthkeys`, falling back to the state directory. State/config directories
are private; transcript/audio/export files are private. Diagnostic logs do not intentionally contain
transcripts. Clipboard managers and destination applications have their own retention policies.

History includes searchable text/app/window metadata, duration/counts, and raw/final text. Raw text
is persisted before optional AI and final text before desktop delivery. Recovery retains the latest
successful audio plus failed/interrupted attempts; `save_audio=on` retains earlier successes too.
`audio_budget_mb` defaults to 4000 decimal MB, pruning oldest inactive audio after work and while idle.
Transcripts survive audio pruning. A recording larger than the budget can lose recovery audio immediately.

`save_history=off` instead keeps the latest dictation only in memory and records in private runtime
temporary files, removed on completion/cancel/shutdown and cleaned after a crash on next service start.
No raw/final transcript row is saved for that attempt. Copy/paste latest uses this memory result,
not an older stored transcript. Restart clears it. Explicit dictionary learning/enrollment remains
separately configured persistent user data. Disabling history also suppresses automatic file-history
saving. `save_file_history` and `file_history_limit` independently bound the file browser.

Shutdown/SIGTERM stops capture without starting delivery and retains interrupted audio when history
is enabled. Cancel discards the ongoing attempt and clears any already-saved raw cancelled text.

```sh
mouthkeys history --query phrase --limit 100 --offset 0
mouthkeys copy --id ENTRY_UUID
mouthkeys reprocess --id ENTRY_UUID
mouthkeys clear-history --yes
mouthkeys delete-audio --yes
mouthkeys shutdown
```

## Dictionary, learning, and pronunciation

Custom Dictionary supports multiple triggers, enable switches, punctuation/action aliases, private
JSON import/export, and spoken-trigger training. **Train by voice** records the recognizer's actual
output without delivery, then lets you review a replacement. This differs from acoustic enrollment.
Formatting supports editable fillers, regional keep-word suggestions based only on local locale/timezone,
reset controls, literal-prefix punctuation/actions, slash/@ shortcuts, QQ/trailing Q, casing, trailing
period removal, and spacing. `smart_capitalization`/`space_between_dictations` use actual AT-SPI context;
unknown context is not guessed. Punctuation/action rows are independently enabled.

**Learning & Vocabulary** offers three separate controls:

- `automatic_learning=on`: after a **verified actual insertion**, observe the same foreground editable
  field for `learning_observe_seconds` (default 30). Only stable edits within the anchored delivered
  range count; no guessed spelling correction, terminals, passwords, or partial field snapshots.
  `learning_min_occurrences` defaults to 2. Review Accept/Dismiss suggestions; `learning_confirmation=off`
  explicitly permits automatic acceptance. Accepted pairs become replacements/custom vocabulary.
  Full field contents stay in memory, not the learned-pair file. New recording/cancel stops observation.
- Pronunciation enrollment: supply a 1–5-word spelling and record at least two consistent isolated
  0.25–10 s takes. `pronunciation_matching=on` uses local cepstral/DTW templates, not Apple embeddings.
  With actual Whisper/NeMo word timings, only matched word spans are replaced inside a longer utterance;
  outside text, punctuation, and observed intervals are preserved. Without actual alignment, only
  short whole-phrase matching is available. Ambiguous labels/distractors abstain. The search budget
  (`pronunciation_span_budget`, default 2000 comparisons), distance-cell/time caps, and 1–5-source-word
  candidate limit bound work; incomplete coverage is reported rather than claimed exhaustive.
  These scores are heuristic similarity, **not** calibrated recognition accuracy or speaker identity.
- `vocabulary_boosting=on`: custom terms have weights 1–5 and explicit aliases. Whisper/Qwen use bounded
  repeated context hints; supported NeMo uses a real weighted decoder context graph. ONNX has no native
  hotword facility. Aliases are explicit literal normalization, not approximate acoustic guesses.

Acoustic features, learned pairs, and vocabulary are private configuration data. Delete unwanted
pronunciation profiles/vocabulary through their page. Dictionary export covers replacement/formatting
rules; a complete backup also includes vocabulary and pronunciation features. Generalizing enrollment
quality to real speakers/noise requires actual voice tests.

## Optional AI cleanup, prompts, Command and Edit

`ai_enabled=on` sends transcript **text**, not audio, to the selected local/remote endpoint.
Remote endpoints require HTTPS; only loopback permits HTTP. Redirects are rejected. API keys live in
the environment variable named by `ai_key_env` (default `MOUTHKEYS_AI_API_KEY`), never URL/profile fields.
An empty AI key-variable name is supported for unauthenticated local servers. Restart the service
manually after changing its environment. Configure model, prompt, output limit, and timeout explicitly.

| `ai_provider` | Example full endpoint |
| --- | --- |
| `compatible` | `https://api.openai.com/v1/chat/completions` |
| `anthropic` | `https://api.anthropic.com/v1/messages` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` |
| `ollama` | `http://localhost:11434/api/chat` |

**Verify connection** sends a fixed small test, not your saved prompt/transcript. Compatible verification
uses `max_tokens=32`; a gateway requiring another token-limit convention may reject that test even
if normal cleanup works. **Discover models** uses recognized protocol routes or an explicit same-origin
`ai_models_endpoint`. Neither action runs when merely opening Settings. Reasoning controls use
`ai_reasoning_parameter`/`ai_reasoning_value`: supported dotted thinking/reasoning fields with scalar
JSON/string values, never model/messages/auth overrides. Support varies by provider/model.

`ai_streaming=on` uses native compatible/Anthropic/Gemini SSE or Ollama NDJSON. By default only output
text appears. `ai_show_thinking=on` separately displays reasoning content the external provider
actually returns; it does not request unavailable internal reasoning or enable provider thinking by
itself. Provider reasoning configuration remains explicit. Cancellation closes the owned stream;
initial connection remains timeout-bound.
Truncated/premature/error streams never deliver partial text. Cleanup failure visibly falls back to
formatted raw text; provider error bodies and keys are not echoed into diagnostics.

AI Prompts stores named profiles, provider/model settings, and optional shortcuts. Routing priority:
explicit `--profile` > matching application/mode route > mode default. `ai_scope=selected` restricts
normal dictation cleanup to routed apps. `edit_profile`/`command_profile` independently choose
`default`, `off`, or a profile UUID; default shares global provider but uses the built-in mode prompt
or `edit_prompt`/`command_prompt`, while a selected profile supplies its own provider and prompt.
Typed/voice draft-prompt tests show results without desktop insertion. Selected text is data, not an
instruction override.

### Command/Edit conversations

Command and Edit support persistent conversations with New/Switch/Rename/Delete controls. The private
`assistant.sqlite3` store keeps at most **50 chats total across both modes**, with 200 messages/1 MiB
of message text per chat. Command execution receipts are separately capped at 100 events/512 KiB
per session. Provider context uses only recent complete turns, bounded to 32 messages/240 KiB; older
saved turns are not necessarily sent again. Chat deletion removes its saved session, not text already
copied to another application or data retained by an external provider.

With `save_history=off`, typed and voice follow-ups share bounded in-memory private context rather
than writing session messages to SQLite. Previously saved chats are not implicitly deleted. Complete
backups merge saved assistant sessions by UUID; legacy backups without assistant-session records
leave current sessions intact.

Command Mode proposes an editable shell command, explanation, and optional absolute working
directory. Execution requires a separate confirmation showing the exact command/directory; a blank
directory resolves to the GUI process's absolute directory and is shown in confirmation.
Execution is bounded to 30 seconds/64 KiB output,
and cancellation stops its child process group. Observed command, working directory, exit status,
output, timeout/truncation/cancellation outcome are retained as bounded receipts even when automatic
continuation is off or a later provider request fails. They use private session storage when history
is on and memory only when it is off; explicit follow-ups can still see the last execution outcome.
`command_continue=on` sends those results, including command output, to the selected AI provider for
the next proposal, up to a 20-step review boundary. **Every command still requires its own concrete
review/confirmation**; continuation never authorizes an execution loop. Turning continuation off
stops automatic next-proposal requests, not explicit user follow-up instructions.

Edit supports typed/captured selected text, generated content, and follow-up instructions applied
to the previous assistant result unless a new selection is supplied. Default `edit_delivery=review`
shows output for Copy; opt-in `replace` requires an unchanged captured selection. **Cancel request**
is separate from **Cancel recording** and suppresses late results/deltas and stale-chat output.
Provider-returned thinking display is optional, display-only rather than saved assistant-message
content, and does not constitute a command authorization.

```sh
mouthkeys toggle --profile PROFILE_UUID
mouthkeys start --mode command
mouthkeys start --mode edit --selected-text 'Original text'
mouthkeys assistant-history --mode command
mouthkeys assistant-history --mode edit --new
mouthkeys assistant-history --id CHAT_UUID --select
mouthkeys assistant-history --id CHAT_UUID --rename 'Project follow-up'
mouthkeys assistant-history --id CHAT_UUID --delete --yes
```

## File transcription, stats, backup, and API

File Transcription accepts picker/drop input, separate persistent search/copy/export/delete history,
TXT/JSON export, duration and processing time. Available segment log probabilities produce an explicitly
labeled confidence estimate; unsupported engines leave confidence unknown, not fabricated. Files use
no microphone/clipboard/key delivery. CLI is stdout/export only unless `--save-history` is requested.
GUI saving also respects `save_history` and `save_file_history`.

```sh
mouthkeys transcribe recording.wav --json --output transcript.json
mouthkeys transcribe recording.wav --save-history
mouthkeys file-history --query meeting
mouthkeys delete-file-history --id RESULT_UUID --yes
# Omit --id to clear file-transcription history, after reviewing the action.
```

Optional `[diarization]` uses sherpa-onnx CPU segmentation/embedding models, downloaded explicitly:

```sh
mouthkeys download-speakers
mouthkeys config file_speaker_labels on
mouthkeys config file_speaker_count 0   # 0 automatic; 1–20 explicit
```

Default `file_diarization_mode=turns` identifies speaker turns first and transcribes those audio slices;
uncovered audio is also decoded so missed turns do not silently drop words. `overlap` retains the
older whole-file/majority-overlap segment attribution as an explicit alternative. Diarization is
speaker labeling, not identification; speaker accuracy, overlap, and short turns require real recordings.

Stats includes time windows, saved-time estimate against `typing_wpm`, totals, current/best streaks,
weekend policy, daily charts, app/peak/record summaries, and milestones. It derives from local history;
disabling history also removes those attempts from durable stats.

Backup & Audio exports private settings/profiles/mode routes, dictionary/vocabulary/pronunciation
features, dictation/file-history/assistant-session records (including saved command output receipts),
and separate audio ZIPs. Restore validates before applying,
merges history by UUID, and does not delete other recordings. Environment secrets are not exported.
Learning-policy observation counts/dismissal cooldowns and in-memory pending suggestions are not the
same as accepted dictionary data; backups preserve accepted data, not an active field observation.

```sh
mouthkeys backup-export backup.json
mouthkeys backup-import backup.json --yes
mouthkeys export-audio recordings.zip
mouthkeys dictionary-export dictionary.json
mouthkeys dictionary-import dictionary.json --yes
```

Optional loopback API: set `api_enabled=on`, port, and a service environment token named by
`api_token_env` (default `MOUTHKEYS_API_TOKEN`, at least 24 characters), then restart yourself.
Every request requires `Authorization: Bearer TOKEN`. Browser-origin/rebinding requests are rejected;
no CORS, remote binding, or arbitrary input file paths. Routes include health/history/stats,
`GET/POST /v1/dictionary/replacements`, `GET/POST /v1/dictionary/custom-words`,
`POST /v1/postprocess` (`text`), and `POST /v1/transcribe` (`audio_base64`). Custom-word entries contain
`text`, integer `weight`, and `aliases`; append/replace writes validate before saving. Requests cap at
25 MiB, uploaded decoded audio at ten minutes, concurrent connections at eight, and reads at 15 seconds;
API inference is serialized and shares the recording model lock.

## Parity and limits

See [PARITY.md](PARITY.md) for the full upstream-baseline acceptance ledger and exact evidence gaps.
Major limits are desktop/toolkit capabilities, optional engine dependencies/checkpoint access,
model/hardware measurements, and actual compositor behavior—not blanket bans on optional features.
Apple-only APIs/notch rendering are replaced by Linux equivalents, not shipped as pretend adapters.
A mock test, tiny random-weight smoke, or working protocol parser does not prove pretrained accuracy,
real clipboard timing, foreground behavior, or receiver battery readings.

## Development and validation

Run the repository-required headless checks and build:

```sh
./linux/check.sh
```

It performs locked Python 3.13 GUI dependency sync, Ruff lint/format checks, offscreen pytest, and
`uv build linux`. Tests use fake capture/clipboard/key adapters, isolated XDG directories, bounded
waits, and no live microphone, clipboard, synthesized keys, visible windows, focus changes, or sounds.

Optional existing-fixture checks, separate from the default headless suite:

```sh
# Explicit small pretrained-model download; never records or pastes:
uv run --project linux mouthkeys download tiny.en
uv run --project linux python linux/tests/smoke_transcription.py
```

For actual native library inference without downloading a pretrained checkpoint, use an isolated CPU
environment rather than modifying the installed app:

```sh
uv venv --python 3.13 /tmp/mouthkeys-native-smoke-env
uv pip install --python /tmp/mouthkeys-native-smoke-env/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchaudio==2.11.0
uv pip install --python /tmp/mouthkeys-native-smoke-env/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -e './linux[transformers,nemo]'
/tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
  --build-tiny qwen --artifact /tmp/mouthkeys-tiny-qwen --report /tmp/mouthkeys-qwen-smoke.json
/tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
  --build-tiny cohere --artifact /tmp/mouthkeys-tiny-cohere --report /tmp/mouthkeys-cohere-smoke.json
/tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
  --build-tiny nemo --artifact /tmp/mouthkeys-tiny-nemo --stream --word-times \
  --report /tmp/mouthkeys-nemo-smoke.json
```

Artifact destinations must not already exist. These smoke builders use real installed model classes,
synthetic tokenizers/random tiny weights, and the existing WAV fixture. They exercise loading,
inference, streaming/caches, timings, and worker isolation—not pretrained recognition quality.
`smoke_native.py --engine nemo --artifact /absolute/path/model.nemo --stream` instead uses an already
available checkpoint. Never download a large/gated model implicitly to turn smoke evidence into an
accuracy claim. Record package versions, model revision, device, language, and observed errors.

After a push to `main` passes CI, GitHub Actions bumps the patch version with `uv version`, validates,
and commits `linux/pyproject.toml`/`linux/uv.lock` together. Pull that version commit before reinstalling.
The application performs no update polling. Linux changes target this personal fork, not upstream macOS.

### Manual desktop and hardware checklist

Headless tests cannot complete these checks. Record compositor/toolkit/backend/version and relevant
configuration before marking a setup verified:

- Cold microphone/Bluetooth start: speak when Listening appears; check first word, immediate meter,
  silence readiness, stop/cancel/duration-limit microphone release, and interrupted-audio recovery.
- Editor and Ghostty/other terminal: Unicode/newlines, normal clipboard paste, direct nonterminal
  insertion, and no simulated character typing into terminals.
- Focus safety: switch windows, fields, tabs, terminal panes, and virtual desktops while processing.
  Test active-window changes where a toolkit leaves its old child locally focused. Stale, protected,
  or unidentifiable destinations must not receive automatic text/Send.
- Start target and explicit return: verify `target_policy` both values and `return_to_start` off/on;
  no cross-window activation without the latter. Edit replacement must use exactly the captured
  selection; modify selection/text before completion and confirm review/copy fallback.
- Queue and cancel: record twice while inference runs; verify FIFO delivery, responsive status,
  queued × cancellation, stale preview suppression, and no late output after cancel/new recording.
- Clipboard restoration: text/image/file MIME formats, slow consumers, rich formats, and user copy
  during delivery. Confirm user-copy wins and observe whether your clipboard manager honors hints.
- X11, Sway, Hyprland, GNOME/KDE Wayland: validate the actual AT-SPI/window backend and virtual-key
  tool in use. Missing field/compositor support must copy instead of inventing a destination.
- Spoken Send: phrase/literal/sentence cases, configured modifier, terminal Return, half-second cancel,
  continued-speech disarm, newer-job suppression, and held-key release/overlapping shortcuts.
- Panel/tray: no focus theft while typing; light/dark/compact/placement/drag/reset; inline reinsertion,
  failure recovery, tray close/reopen/quit guards, startup/window preference, and notifications.
- Integrated shortcuts: explicit X11 keyboard/mouse grabs or portal consent/denial; multiple chords,
  conflict errors, hold/toggle/automatic, autorepeat, overlapping primary shortcuts, profile binding,
  and terminating/reconfiguring the listener while a gesture remains held.
- Microphones/media/sounds: reorder/exclude/disconnect devices, output selection, actual meter,
  pause/resume only players the app paused, user playback changes, start/stop preset/custom cues.
- Lark A1: confirm receiver-source gating, report-descriptor matching, permissions/unplug/replug,
  actual transmitter/battery values, and no settings/firmware writes beyond the documented heartbeat.
- Learning: edit a verified insertion, including selecting then replacing a word; test repeat/cooldown,
  Accept/Dismiss/confirmation-off, focus/new-record/cancel/opt-out, password/terminal/large-field refusal.
  Confirm only accepted words/pairs—not full surrounding field contents—persist as dictionary data.
- Pronunciation/vocabulary: independent real takes, homophones/distractors/noise, embedded spans,
  timing/punctuation preservation, budget warnings, ambiguity abstention, and each supported engine's
  native vocabulary mechanism. Similarity scores are not proof of recognition accuracy.
- AI: explicit Verify/Discover only, local then configured cloud provider, reasoning/streaming,
  per-mode/app/shortcut routes, draft tests, typed/voice cancel, failed/truncated response fallback,
  no desktop output from prompt tryout, and exact shell-command/directory confirmation before execution.
- Conversations: new/switch/rename/delete in both modes, typed ↔ voice follow-ups, history-off memory,
  backup/legacy restore, long-context trimming, visible returned-thinking opt-in, cancellation while
  switching sessions, reviewed command-result continuation, nonzero exits/timeouts, 20-step boundary,
  and child-process cancellation. Verify no next command executes without another confirmation.
- Files: mixed speakers/overlap/brief turns/missed turns, labels/confidence unknowns, TXT/JSON, persistent
  browser copy/delete/restart, private backup roundtrip, and history-off behavior.
- Pretrained engines: after authorized download/access, evaluate chosen language/model/revision,
  warmup, long dictation boundaries, native cache-aware finalization, resource use, and real accuracy.

The macOS application, Xcode tooling, and tests remain maintained upstream.
