# MouthKeys for Linux

Local dictation for Linux, in the personal [crypdick/MouthKeys-Linux](https://github.com/crypdick/MouthKeys-Linux)
repository. Record with a desktop shortcut, transcribe on your CPU or NVIDIA GPU, and paste into your
editor or terminal. No account, cloud transcription, telemetry, or update checks.

The Linux runtime is Python/Qt with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
The macOS app is maintained in the upstream MouthKeys repository. This Linux implementation
does not provide full macOS feature parity: no Apple/Parakeet models, media controls, Spoken Send,
or automatic clipboard restoration. The Linux UI provides recording controls, history, copy,
audio reprocessing, model downloads, and settings. Its floating recording panel follows the
macOS Datasheet Mono design: a square 340 × 130 panel, side controls, three-line live transcript,
real microphone trace, timer, and system light/dark appearance. The main window uses the same
sidebar, typography hierarchy, and ruled surfaces. Linux uses system sans/monospace fonts in
place of Apple's fonts and does not expose macOS-only settings.

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
   Loading and failed shortcut commands also appear in the panel. There are no sounds.
3. When transcription finishes, paste with Ctrl+V, or Ctrl+Shift+V in most terminals.

The recording timer and “Listening” status start only after the first audio buffer is saved.
Capture limits FFmpeg input probing so waveform data is available promptly; a source that
produces no audio within five seconds fails instead of showing a running timer. Speak when
the recording panel appears. Live transcript words still depend on the preview interval below.

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
they never select the currently focused window for automatic paste. A busy model rejects additional
recording commands explicitly; it never silently replaces pending work. Cancel suppresses
delivery even if inference has already started. Inference must finish before another recording
can start. Only one model worker runs at a time. Capture shutdown and text delivery run
outside the controller state lock so status remains responsive. Once delivery has started,
cancellation reports that it is too late to withdraw the clipboard/paste operation.

## Text delivery

**Clipboard mode is the default on every desktop.** It writes the finished text to the clipboard
and leaves it there. It replaces your previous clipboard contents; Linux does not yet provide
the macOS version's multi-format clipboard snapshot/restore. Clipboard managers may retain
dictated text. History is saved before delivery, even if the clipboard tool fails.

**X11 optional automatic paste:**

```sh
mouthkeys config delivery x11
# Ctrl+Shift+V is the terminal-oriented default; for ordinary editor/browser fields:
mouthkeys config paste_shortcut ctrl+v
```

The daemon snapshots the focused X window on an explicit stop, before acknowledging it,
and rechecks before sending the paste shortcut. If the window changed or cannot be identified, it only copies. It never activates a
window or sends Return. This check cannot distinguish tabs, panes, or fields within the same
X window and is not equivalent to macOS Accessibility validation. Use clipboard mode if you
switch panes while waiting. A `paste_requested` status means the shortcut was sent, not proof
the destination accepted it. Terminal text always uses clipboard paste, never simulated typing.

**Wayland:** native apps cannot be safely targeted with X11 tools. Automatic paste is therefore
disabled, even when `delivery` is set to `x11`. GNOME and KDE work with `wl-copy` plus your
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

Upstream macOS saves ordinary history audio only when enabled (off by default); Linux
retains recovery audio by default. The storage budget bounds that Linux behavior, rather
than implying parity with upstream's audio-saving preference. No transcript is written to the daemon diagnostic log.

To stop the background service, use `mouthkeys shutdown`. Shutdown, SIGTERM, and SIGINT
stop capture and preserve interrupted audio for reprocessing, without starting transcription
or delivery. Use `mouthkeys cancel` to explicitly discard an ongoing recording.
Remove `~/.config/autostart/mouthkeys.desktop` to
disable login startup. Uninstall with `uv tool uninstall mouthkeys-linux` and remove
`~/.local/share/applications/mouthkeys.desktop`. These steps leave history/models intact;
delete the MouthKeys XDG directories separately if you want to erase them.

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
