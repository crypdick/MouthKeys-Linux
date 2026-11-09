# MouthKeys for Linux

Local dictation for Linux, in the personal [crypdick/MouthKeys-Linux](https://github.com/crypdick/MouthKeys-Linux)
repository. Record with a desktop shortcut, transcribe on your CPU or NVIDIA GPU, and paste into your
editor or terminal. Local transcription works without an account. Optional AI cleanup supports local or cloud providers; telemetry and update checks stay off.

The Linux runtime is Python/Qt with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
The macOS app is maintained in the upstream MouthKeys repository. This Linux implementation
targets upstream feature parity using Linux integrations. It includes optional Spoken Send, local/cloud
AI cleanup and prompt profiles, dictionary/formatting, media controls, a system tray, searchable
history and audio exports, stats, and file transcription with optional speaker labels. Its floating recording panel follows the
macOS Datasheet Mono design: a square 340 × 130 panel, side controls, three-line live transcript,
real microphone trace, timer, and system light/dark appearance. The main window uses the same
sidebar, typography hierarchy, and ruled surfaces. Linux uses system sans/monospace fonts in
place of Apple's fonts and adapts macOS-only integrations where Linux alternatives exist. See [Parity and limits](#parity-and-limits).

## Install

Python 3.11–3.14, FFmpeg with PulseAudio input, and a PulseAudio server (including PipeWire's
PulseAudio compatibility service) are required. Python 3.13 is the tested version.

On Ubuntu/Debian, install the desktop dependencies:

```sh
sudo apt install ffmpeg wl-clipboard libxcb-cursor0 libxkbcommon-x11-0
# On an X11 session, also install:
sudo apt install xclip xdotool
```

On Fedora the corresponding packages include `ffmpeg-free`, `wl-clipboard`, `xcb-util-cursor`,
`libxkbcommon-x11` and, for X11, `xclip` and `xdotool`. On Arch use `ffmpeg`, `wl-clipboard`,
`xcb-util-cursor`, `libxkbcommon-x11`, and the same X11 tools. Package availability depends on
your distribution's repositories. `mouthkeys doctor` reports missing command-line tools.

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

Ensure `~/.local/bin` is on your PATH (`uv tool update-shell` can configure it). The desktop
launcher contains an absolute executable path. For a terminal-only install, omit `[gui]`.
Models download **only** when you run `mouthkeys download` or press Download selected model.
The default is `base.en` (English), CPU, and the model's stored precision with hardware
fallbacks (`compute_type=default`). Downloads come
from Hugging Face; dictation itself uses cached models with `local_files_only=True`.

### Update and check your version

**App Settings** shows the app version and the running recording service version. A mismatch
means an older process is still running. `mouthkeys --version` reports the command's version;
`mouthkeys status` includes the service's version. Restart the app and service after updating;
older services without version reporting are unsupported.

To update from this checkout, finish any dictation and close the settings window, then run:

```sh
mouthkeys shutdown
git pull --ff-only
uv tool install --force --python 3.13 './linux[gui]'
mouthkeys gui
```

Check that both versions in App Settings match the version in `linux/pyproject.toml`.
Updating the package alone does not restart an existing recording service.

## Set up your shortcut

In GNOME **Settings → Keyboard → View and Customize Shortcuts → Custom Shortcuts**, add:

| Name | Command | Example shortcut |
| --- | --- | --- |
| MouthKeys toggle | `/home/YOU/.local/bin/mouthkeys toggle` | Super+Alt+Space |
| MouthKeys cancel | `/home/YOU/.local/bin/mouthkeys cancel` | Super+Alt+Escape |

Replace `/home/YOU` with your actual home directory. Use the output of `command -v mouthkeys`
if installed elsewhere. KDE can bind the same commands under Custom Shortcuts. For Sway:

```text
bindsym $mod+Mod1+space exec ~/.local/bin/mouthkeys toggle
bindsym $mod+Mod1+Escape exec ~/.local/bin/mouthkeys cancel
```

The desktop owns the shortcut, so MouthKeys never grabs global keys or needs root. The first
command starts a private background daemon. `mouthkeys warmup` loads the model ahead of your
first recording; subsequent dictations reuse it. Closing the settings window leaves the daemon
running. Optional login startup: `mouthkeys install-desktop --autostart`. Restart the daemon
after changing desktop session types; its display environment is inherited at launch.

1. Focus your editor or terminal and press the toggle shortcut.
2. Speak, then press it again. The controller shows recording/transcribing/ready status;
   `mouthkeys status` also reports it for a status bar. With the GUI extra installed, a floating
   panel appears automatically, including when Settings is closed. It shows provisional words
   while you speak, then the final delivery status. It never intentionally takes keyboard focus.
   Loading and failed shortcut commands also appear in the panel. Sounds are off unless you choose a local cue file in App Settings.
3. When transcription finishes, paste with Ctrl+V, or Ctrl+Shift+V in most terminals.

The recording timer and “Listening” status start only after the first audio buffer is saved.
On PipeWire, Bluetooth virtual inputs can supply silence while the headset is still switching
from playback to microphone mode. MouthKeys also waits for that headset's physical capture node
to run before showing Listening. This check uses `pw-dump` (from PipeWire tools); it neither waits
for speech nor keeps the microphone open between recordings. Capture limits FFmpeg input probing
so waveform data is available promptly; a source that produces no audio or cannot become ready
within five seconds fails instead of showing a running timer. Speak when the recording panel
appears. Live transcript words still depend on the preview interval below.

The panel opens near the bottom of the screen containing the pointer. Drag it to reposition;
double-click its surface to reset. History opens the main window, Copy copies the finished
transcript, × cancels/dismisses, and Reprocess retries retained audio. Linux compositors control
placement and stacking: Wayland may ignore requested positioning or keep other windows above
it. Focus behavior requires a real desktop check; offscreen Qt tests cannot verify it.
If your compositor tiles the panel or gives it focus, disable it in Settings or with
`mouthkeys config recording_panel off` and use `mouthkeys status` for feedback. Restore it
with `mouthkeys config recording_panel on`. Verify focus behavior on your desktop before
enabling X11 automatic paste.

Live transcription runs locally on the selected model, roughly two seconds after the previous
update finishes. Completed Whisper segments ending at least two seconds before the audio snapshot
are retained; newer text remains provisional and may change. Updates read forward from the first
uncommitted frame, normally up to 30 seconds at a time, so slow inference never skips earlier speech.
If no completed boundary is available, that window grows with incoming audio rather than freezing.
Stopping waits for useful in-flight inference, then decodes only the uncommitted audio through the
end of the recording. If the last update already covers the finalized audio, its result is reused
without another decode. There is no second pass over committed text. Segment boundaries can affect
accuracy compared with a full-recording pass; Reprocess always starts fresh from retained audio.
If a preview fails, finalization retries its uncommitted audio. A segment with no boundary before
the two-second margin stays provisional until finalization, which can mean a longer stop wait.
The engine cannot interrupt a native inference call; cancel suppresses delivery and skips queued
final inference. One model worker is shared throughout, without loading a second model.
`mouthkeys warmup` avoids first-use model loading. Live text is not saved or delivered until stop.
The waveform uses actual recorded energy with 20 ms capture buffers and a separate 60 Hz
animation clock. Bars scroll at 12 per second of speech, ease to their measured heights, and
hold during silence after a 250 ms hangover, following the native macOS motion. Only the newest
20 ms of audio is metered; preview inference does not drive animation. Cancel discards the preview and suppresses final delivery.

Recordings stop automatically after five minutes by default. Automatic stops only copy;
they never select the currently focused window for automatic paste. You can start another recording while earlier dictations transcribe; final inference and delivery
remain ordered. Queue capacity is configurable (`queue_limit`, default 8); full queues reject new
recordings without dropping existing work. Cancel suppresses
delivery even if inference has already started. Cancellation applies to the latest visible recording; earlier queued work stays intact. Starting
a new recording suppresses any pending Spoken Send from an earlier job. Only one model worker runs at a time. Capture shutdown and text delivery run
outside the controller state lock so status remains responsive. Once delivery has started,
cancellation reports that it is too late to withdraw the clipboard/paste operation.

## Text delivery

**Clipboard mode is the default on every desktop.** It writes finished text to the clipboard and
leaves it there. Clipboard managers may retain dictated text. With history enabled, raw and final
transcripts are saved before delivery, even if desktop delivery fails. With history disabled,
transcripts stay in memory and recording audio uses a private runtime temporary directory.

Optional X11 `restore_clipboard=on` snapshots all formats advertised through Qt (up to 32 MiB),
then restores them only while MouthKeys still owns the clipboard. A user copy wins. Restoration
waits `clipboard_restore_delay_ms` (default 500 ms, range 100–5000); this is not proof a slow app
consumed the paste. Snapshot failure falls back to normal clipboard delivery. Wayland restoration
and hiding temporary text from clipboard managers are not implemented. These limitations are
reported rather than claiming macOS-equivalent paste verification.

X11 `insertion_mode=direct` requests clipboard-free typing for printable single-line text into
identified non-terminal windows. Unknown window classes, multiline/control characters, and terminals
(including Ghostty/c11) always use clipboard paste. `copy_to_clipboard=off` keeps direct insertion
clipboard-free; clipboard-mode delivery still necessarily changes the clipboard.

**X11 optional automatic paste:**

```sh
mouthkeys config delivery x11
# Ctrl+Shift+V is the terminal-oriented default; for ordinary editor/browser fields:
mouthkeys config paste_shortcut ctrl+v
```

The daemon snapshots the focused X window on an explicit stop, before acknowledging it,
and rechecks before sending the paste shortcut. If the window changed or cannot be identified, it only copies. It never activates a
window. It sends Return only when Spoken Send is explicitly enabled and its checks pass. This check cannot distinguish tabs, panes, or fields within the same
X window and is not equivalent to macOS Accessibility validation. Use clipboard mode if you
switch panes while waiting. A `paste_requested` status means the shortcut was sent, not proof
the destination accepted it. Terminal text always uses clipboard paste, never simulated typing.

**Wayland:** native apps cannot be safely targeted with X11 tools. `delivery=x11` never injects
keys into native Wayland apps. For explicitly opted-in automatic paste, set `delivery=wayland`
and configure `paste_backend=wtype` or `ydotool` as below. Sway (`swaymsg`) and Hyprland
(`hyprctl`) supply window identity, rechecked before paste. Unsupported compositors and changed
focus fall back to copy; this cannot distinguish fields/panes inside one window. Duration-limit
stops still only copy. Clipboard mode remains the default. GNOME and KDE work with `wl-copy` plus your
normal paste key. For an explicit “paste last dictation now” desktop shortcut:

```sh
# Sway and other compositors implementing the virtual-keyboard protocol:
mouthkeys config paste_backend wtype
# Or GNOME/KDE, ONLY if you already have a working, user-accessible ydotool daemon:
mouthkeys config paste_backend ydotool
# Bind this to a second desktop shortcut, invoked once transcription is complete:
mouthkeys paste
```

`wtype` does not work on all compositors. `ydotool` requires a separately configured daemon and
access to `/dev/uinput`; MouthKeys neither installs a privileged service nor changes device
permissions. The daemon must inherit `YDOTOOL_SOCKET` if your setup uses a non-default socket.
`paste` copies the last transcript and sends only the configured paste shortcut to the current
focus. Release your shortcut modifiers promptly. Do not invoke it from the GUI while expecting
another app to receive it. Standard manual paste needs neither tool nor extra privileges.

## Models, microphones, and recovery

The GUI's **Settings** tab has dropdowns for model, language, compute type, microphone,
device, and delivery settings. Languages show names and save the engine's language codes;
**Detect automatically** selects automatic language detection. Model and microphone fields
also accept custom model paths/IDs and PulseAudio source names. Microphone discovery uses
`pactl` when available; otherwise you can enter a source name. Recording duration uses a
bounded seconds control. Selecting a model does not download it: use **Download selected model**.
Models appear in approximate transcription accuracy order, strongest first, for short English
dictation. Accuracy varies by language and audio; `.en` and distilled models are English-only.
Duplicate aliases use their specific names: `large-v3` and `large-v3-turbo`.

Compute type `default` follows the model's stored precision with CTranslate2 hardware
fallbacks; `auto` chooses the fastest supported type. `int8` is an optional way to reduce
memory use and improve CPU speed. Model precision can use more memory and run slower than
INT8 on CPU. Existing saved precision settings are preserved. To switch an existing
configuration to model precision, run `mouthkeys config compute_type default`.

```sh
# More accurate English model (larger and slower on CPU):
mouthkeys config model small.en
mouthkeys download
mouthkeys warmup
# Multilingual:
mouthkeys config model small
mouthkeys config language auto
mouthkeys download
# CUDA requires compatible NVIDIA/CTranslate2 libraries; see faster-whisper's documentation:
mouthkeys config device cuda
mouthkeys config compute_type default
# Return to CPU (optionally use int8 for lower memory use and faster inference):
mouthkeys config device cpu
mouthkeys config compute_type default
# Pick a PulseAudio/PipeWire source (pactl is provided by pulseaudio-utils on Ubuntu):
pactl list short sources
mouthkeys config audio_source YOUR_SOURCE_NAME
```

`default` follows your desktop's default input device. Do not select a `.monitor` source unless
you intentionally want system audio. Change settings only between recordings; a recording
keeps the settings it started with. Use a local CTranslate2 model directory as `model` to avoid
downloads entirely.

```sh
mouthkeys history                     # Latest 100 entries, JSON
mouthkeys copy                        # Latest nonempty transcript
mouthkeys reprocess                   # Latest entry's retained audio
mouthkeys reprocess --id ENTRY_UUID
mouthkeys transcribe recording.wav    # File → stdout; no mic, clipboard, or history writes
mouthkeys clear-history --yes         # Delete all local transcripts and retained recordings
```

History and audio are private files in `$XDG_STATE_HOME/mouthkeys` (normally
`~/.local/state/mouthkeys`). Configuration is in `$XDG_CONFIG_HOME/mouthkeys/config.json`;
models are in `$XDG_CACHE_HOME/mouthkeys/models`. The control socket is user-private under
`$XDG_RUNTIME_DIR/mouthkeys`, falling back to the state directory. Audio for the latest success
and failed/interrupted attempts is retained for recovery, including after a crash; older
successful audio is removed after a successful dictation. Each reprocess creates a separate attempt, preserving
the original transcript if the retry fails. Retained audio (including failures) has a configurable **4,000 MB** budget, inspired by
upstream macOS's budget-based retention (whose default is 4 GiB). Oldest inactive audio is deleted first; transcripts remain.
The budget is enforced after transcription and once a minute while idle, including after
restarting. An active recording is excluded until it finishes. Adjust it in Settings or with
`mouthkeys config audio_budget_mb 1000` (decimal MB). A recording larger than the budget
can lose its recovery audio immediately. Clear history deletes transcripts and audio.

`save_audio=on` keeps successful history audio within the storage budget; off retains only latest
success and failure/interruption recovery audio. `save_history=off` writes no transcript rows and
uses private runtime temporary recording files, removed on completion/cancel/shutdown and cleaned
on next service startup after a crash. This explicitly opts out of crash-recovery history and the
persist-before-delivery guarantee. Clipboard managers and explicitly configured cloud providers
have their own retention behavior. No transcript is intentionally written to the daemon diagnostic log.

To stop the background service, use `mouthkeys shutdown`. Shutdown, SIGTERM, and SIGINT
stop capture and preserve interrupted audio for reprocessing, without starting transcription
or delivery. Use `mouthkeys cancel` to explicitly discard an ongoing recording.
Remove `~/.config/autostart/mouthkeys.desktop` to
disable login startup. Uninstall with `uv tool uninstall mouthkeys-linux` and remove
`~/.local/share/applications/mouthkeys.desktop`. These steps leave history/models intact;
delete the MouthKeys XDG directories separately if you want to erase them.

## Parity features

All configuration fields are exposed in **App Settings** and `mouthkeys config KEY VALUE`.
Opening settings does not rewrite existing configuration. Per-recording settings are captured
when recording starts; changes affect later recordings. Destructive history actions wait until
pending recordings finish. No installation/restart is performed by changing application code.

### Tray, microphones, media, and appearance

`mouthkeys tray` starts a system-tray controller. `mouthkeys gui` also shows a tray icon when
supported, enabled with `tray=on`. Closing Settings leaves that tray running; Quit tray exits
its UI without stopping the recording service. The menu contains record/stop, cancel, copy,
reprocess, microphone selection, Settings, and service shutdown. Desktop tray support varies
(GNOME may require an AppIndicator extension); use the GUI or i3 commands when unavailable.

Set `launch_at_startup=on` to create the user autostart entry. `show_window_at_login=off` starts
the tray (or daemon when tray disabled); on opens Settings. Updating these preferences rewrites
only MouthKeys' desktop/autostart entries, not the installed package or running service.

`microphone_preferred`, `microphone_fallback`, and `microphone_never` contain comma-separated
PulseAudio source names. Preferred order wins, then a permitted default, then fallback order.
Explicit source selection never overrides Never. Selection is resolved before each recording;
an unplugged active capture is reported, not silently spliced to another device mid-recording.

`pause_media=on` uses optional `playerctl` to pause playing MPRIS players during capture. Only
players MouthKeys actually paused are resumed, and only if still paused. `sound_file` selects
an existing local cue, `sound_volume` controls volume, and `sound_output` selects the PulseAudio
output for optional `paplay`. No cue plays unless configured. Optional tray notifications are
controlled by `show_microphone_alerts` and `show_paste_alerts`.

Overlay options: `overlay_position`, `overlay_size` (percent), `overlay_bottom_offset`,
`preview_chars`, `visualizer_sensitivity`, `overlay_border`, `live_preview`, and `theme`
(system/light/dark). The overlay includes live word count/WPM, microphone selection, inline
history copy/reprocess, and existing controls. Requested positions/focus remain compositor-dependent.

### Dictionary and spoken formatting

**Custom Dictionary** offers replacement, punctuation, and formatting-action tables. Separate
spoken aliases with commas; enable/disable rules individually. Punctuation/action replacement
cells use `\n` for newline and `\t` for tab. Import replaces the dictionary after confirmation;
export writes private JSON. **Train by voice** captures a recognized phrase without delivery:
type correct replacement, record it, stop, review the recognized trigger, then Save. This is
trigger training, not speaker-embedding pronunciation matching.

Formatting preferences include `remove_filler_words` and editable `filler_words`,
`lowercase_first`, `remove_trailing_period`, `literal_formatting` (slash commands and explicit
@ mentions), `question_mark_shortcut` (QQ/trailing Q), and `spoken_formatting` with customizable
`literal_prefix`. Formatting is deterministic and independent of AI cleanup.

`smart_capitalization` and `space_between_dictations` use actual text before the caret when
Linux AT-SPI exposes it. No guessed context is used when unavailable. Optional system packages
`python3-gi` and `gir1.2-atspi-2.0` provide this integration; uv-managed Python can use the bounded
system-Python helper. Password fields are excluded. AT-SPI availability varies by toolkit and
accessibility configuration; this is not reliable terminal-pane identity verification.

### Spoken Send

Spoken Send is off until `spoken_send=on`. Configure `spoken_send_phrase` (default `send it`),
`spoken_send_key` (`return`, `shift+return`, `ctrl+return`), and `send_after_pause` (default off).
A recognized trailing command phrase is removed and the key follows successful requested X11
insertion after a half-second cancel window. Sentence-owned phrases such as “I will send it”
are not commands; say the literal prefix before the phrase to dictate it. Terminals always use
plain Return. Cancel can suppress a pending send after text insertion, but cannot withdraw text
already delivered. Starting another recording suppresses an earlier job's pending send.

With `send_after_pause=on`, a trailing send phrase plus half a second of quiet stops recording
and requests the configured send. Continue talking or use the overlay send control to disarm.
CLI/i3 start/stop bindings do not convey key-hold state; this differs from macOS hold-to-talk.
Spoken Send requires an unchanged X11 window and explicit automatic-delivery configuration;
clipboard-only, duration-limit, and generic Wayland paths do not send. X11 focus cannot distinguish
fields/panes inside a window: validate this opt-in on your desktop before relying on it.

### Optional AI cleanup, prompts, Command and Edit

`ai_enabled=on` enables cleanup using a configured local or remote provider. This sends transcript
text to that endpoint; no audio is sent. `ai_provider` selects protocol, not a destination:

| Protocol | Example full endpoint |
| --- | --- |
| `compatible` (OpenAI, LM Studio, other compatible providers) | `https://api.openai.com/v1/chat/completions` |
| `anthropic` | `https://api.anthropic.com/v1/messages` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` |
| `ollama` | `http://localhost:11434/api/chat` |

Set `ai_endpoint`, `ai_model`, `ai_prompt`, `ai_max_tokens`, and `ai_timeout` explicitly. Remote
endpoints require HTTPS; HTTP is allowed only for loopback. Redirects are rejected. Set the API
key in the recording service's environment under the variable named by `ai_key_env`, default
`MOUTHKEYS_AI_API_KEY`; never paste a key into endpoint URLs or profile files. Restart the service
manually after changing environment variables. Provider errors fall back to formatted raw text
with visible error feedback; provider error bodies are not put in diagnostic logs.

**AI Prompts** saves named profiles with independent prompt/provider/model/endpoint settings;
select one using `ai_profile` in App Settings. Backups include profiles, never environment secrets.
**Command / Edit** accepts typed or recorded instructions. Command Mode proposes an editable shell
command and explanation, and runs it only after a separate confirmation showing that exact command.
Execution is bounded to 30 seconds and 64 KiB output. Edit Mode uses text pasted into its selection
box and returns reviewable output with Copy; it never silently replaces another application's text.
CLI/i3 may bind `mouthkeys start --mode command`, `--mode edit --selected-text TEXT`, or normal
dictation. Recorded proposals appear in Command / Edit after stopping; nothing is automatically
executed or inserted for these modes.

### Linux ONNX engines and speaker-labeled files

Default Whisper installation is unchanged. Optional [onnx-asr](https://istupakov.github.io/onnx-asr/usage/)
provides Linux CPU Parakeet TDT v2/v3 and Canary models:

```sh
uv tool install --python 3.13 './linux[gui,onnx,diarization]'
mouthkeys config engine onnx
mouthkeys config model nemo-parakeet-tdt-0.6b-v3
mouthkeys config device cpu
mouthkeys config compute_type default
mouthkeys download
mouthkeys warmup
mouthkeys models
```

ONNX supports `default`, `auto`, `float32`, and `int8` precision. CPU is supplied by the optional
extra; CUDA needs separately installed `onnxruntime-gpu` and compatible NVIDIA libraries. Runtime
loads offline-only; explicit download also fetches required Silero VAD. ONNX uses bounded VAD
chunks, not Apple's/CoreML's streaming implementation. `mouthkeys delete-model MODEL` removes
only app-owned cached artifacts, never a user-entered local directory; an already loaded model
remains resident until changed or service restarted. GUI offers catalog, download cancellation,
and cached-model deletion. A cancelled download is not marked installed.

**File Transcription** accepts chosen/dropped audio and exports TXT or timestamped JSON. For
optional CPU [speaker diarization](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html):

```sh
mouthkeys download-speakers
mouthkeys config file_speaker_labels on
mouthkeys config file_speaker_count 0  # Automatic; 1–20 specifies count
mouthkeys transcribe recording.wav --json --output transcript.json
```

Speaker models download only on explicit request. Labels are segment-level greatest-overlap
assignments, not word-aligned identification; JSON preserves full speaker turns. File transcription
uses no live microphone/clipboard and does not silently add files to dictation history.

### History, stats, backups, and API

History supports case-insensitive Unicode search, pagination, individual deletion, audio opening,
and transcript/audio pair export. **Stats** shows time-window counts, time saved against configurable
`typing_wpm`, daily usage, streaks (`weekends_break_streak`), records, and available destination-app
metadata. **Backup & Audio** provides usage, audio-only deletion, ZIP export, backup/restore. Restore
validates before applying and merges history by UUID without deleting other recordings.

```sh
mouthkeys history --query phrase --limit 100 --offset 0
mouthkeys stats
mouthkeys audio-usage
mouthkeys export-audio recordings.zip
mouthkeys backup-export backup.json
mouthkeys backup-import backup.json --yes
mouthkeys delete --id ENTRY_UUID
mouthkeys delete-audio --yes
mouthkeys dictionary-export dictionary.json
mouthkeys dictionary-import dictionary.json --yes
```

Optional loopback API: set `api_enabled=on`, `api_port`, and a service environment variable named
by `api_token_env` (default `MOUTHKEYS_API_TOKEN`, at least 24 characters). Restart service yourself.
Every request requires `Authorization: Bearer TOKEN`. Browser-origin/rebinding requests are rejected;
no CORS is enabled. Routes: `GET /v1/health`, `/v1/history`, `/v1/stats`,
`GET/POST /v1/dictionary/replacements`, `POST /v1/postprocess` with JSON `text`, and
`POST /v1/transcribe` with JSON `audio_base64`. Requests are limited to 25 MiB and API audio to 10 decoded minutes. Eight connection slots,
15-second read deadlines, and serialized inference bound API resource use; arbitrary file-path
requests and remote binding are not supported. API shares the recording model lock.

## Parity and limits

Upstream baseline: BenevolentFutures/MouthKeys `d50ad008` (2026-11-01). Defaults remain compatible
with saved Linux settings. Configurable cloud AI and Spoken Send are supported—not forbidden by
Linux policy. Remaining gaps are implementation/platform limits, not invented product decisions:

- No proven paste-consumption or pane/field verification; X11/Sway/Hyprland use window focus,
  other Wayland compositors fall back to copy. No “Return to Starting Field” activation across applications.
- Clipboard restoration is X11-only and timed; clipboard-manager exclusion is not implemented.
- Native Apple ASR/CoreML engines cannot run on Linux. Parakeet/Canary are alternatives; Nemotron,
  Qwen3 ASR, Cohere Transcribe, and Parakeet Flash-specific backends are not yet integrated.
- No automatic correction-learning or advanced pronunciation embeddings/vocabulary boosting.
  Manual/voice trigger training is supported.
- No automatic selected-text replacement, per-app prompt routing, or integrated global shortcut
  capture. CLI/i3 bindings and reviewable Command/Edit output are supported.
- No macOS notch-specific layout, Hollyland battery telemetry, or usage milestone badges.
- No dedicated persistent file-transcription history browser; export results to keep them.

These are outstanding parity work, not claims of completion. Real microphone, compositor focus,
paste timing, optional model accuracy, sounds/media, and tray hosting require desktop checks below.

## Development and validation

After a push to `main` passes CI, GitHub Actions bumps the patch version with `uv version`,
validates the updated package, and commits `linux/pyproject.toml` and `linux/uv.lock` together.
The version commit does not trigger another bump. Pull the version commit before installing
an update. PRs and manual check runs do not bump versions; stale runs and retries skip a bump
when `main` has already advanced. This changes package versions without publishing packages
or adding automatic update checks to the app.

```sh
./linux/check.sh                     # Full headless checks and packaging
# Or run the steps individually:
uv sync --project linux --python 3.13 --extra gui --locked
uv run --project linux pytest
uv run --project linux ruff check linux
uv build linux
# Real model smoke check; downloads explicitly, never records or pastes:
uv run --project linux mouthkeys download tiny.en
uv run --project linux python linux/tests/smoke_transcription.py
```

Unit/integration tests use fake microphones and delivery adapters, temporary XDG directories,
and bounded waits. Qt smoke checks use the offscreen backend. Tests never capture live audio,
change the clipboard, synthesize keys, show windows, or play sounds.

Before calling a desktop setup verified, run this manual check:

- Download a model, warm it, choose your actual microphone, and dictate two sentences.
- Start from an idle microphone and speak as soon as the recording panel appears; confirm
  the waveform responds promptly and the final transcript includes the first word. Repeat once.
- Paste into an editor and a terminal (Ghostty/c11 where available); confirm Unicode and newlines.
- Cancel a recording; confirm there is no clipboard change or late paste.
- Switch windows while X11 inference runs; confirm it falls back to copy.
- Reprocess retained audio, copy from history, and verify recovery after restarting the daemon.
- Confirm the microphone indicator turns off on stop/cancel and at the maximum duration.
- With X11 automatic paste enabled, let the duration limit expire after switching windows:
  confirm text is copied without any paste key being sent.
- Terminate the daemon during recording and restart it; confirm audio can be reprocessed.
- Invoke toggle during model warmup and with an unavailable microphone; confirm visible
  error feedback, then confirm the next successful recording clears it.
- Close Settings and start with the shortcut: confirm the floating panel appears, shows words
  while speaking, and leaves your editor focused. Check both light and dark system appearances.
- Dictate for more than 30 seconds with several sentences and pauses; confirm earlier text stays
  present and the final words arrive once, including when stopping mid-sentence. Compare Reprocess
  for boundary omissions, repetition, and punctuation; time stop-to-delivery for both paths.
- Stop, cancel, and start again while preview inference is running; check that old words never
  appear in the new recording and cancellation never changes the clipboard.
- On GNOME Wayland, KDE Wayland, and Sway, start from a focused editor and keep typing;
  confirm panel appearance leaves keyboard focus and manual paste in the editor. If it fails,
  disable the panel and record the compositor/backend version; these setups are not yet verified.
- Check the panel's positioning, drag/reset, history, copy, reprocess, and dismissal on your
  compositor; final delivery should still target the editor when using X11 automatic paste.

The macOS application, Xcode build, and tests are maintained in the upstream MouthKeys repository.

Additional opt-in desktop checks:

- Enable Spoken Send on X11; verify phrase removal, configured key, sentence-owned phrases,
  cancel countdown, and no Return after beginning a second recording or switching windows.
- Dictate two recordings while first inference is pending; confirm FIFO delivery and queued × cancel.
- Enable clipboard restoration; verify image/file formats survive, and copying new content during
  delivery wins. Tune restore delay on slow applications; no paste-consumption guarantee is made.
- On Sway/Hyprland with opted-in automatic paste, switch windows during inference: only copy.
  Unsupported Wayland compositors must never inject an automatic paste key.
- Check tray start/stop, close/reopen Settings, audio-process quit guard, and optional notifications.
- Verify microphone priority/Never, playerctl pause/resume, and selected local sound/output.
- Verify optional Parakeet/Canary and speaker models against actual recordings after explicit download.
- Disable history, dictate, then Copy last; confirm latest in-memory text and no transcript/audio
  history remains. Restart service and confirm that private last transcript is unavailable.
- Configure a local AI endpoint first; inspect cleanup, prompt selection, Command proposal confirmation,
  and Edit output. Cloud requests occur only after configuring/enabling that provider.
