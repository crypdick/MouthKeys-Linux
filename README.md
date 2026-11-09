# MouthKeys-Linux

Local dictation for Linux: record with a desktop shortcut, transcribe on your CPU or NVIDIA
GPU, and paste into your editor or terminal. Optional Linux ONNX engines, Spoken Send,
local/cloud AI cleanup, dictionary/formatting, system tray, searchable history, and file
transcription. Local dictation needs no account. No telemetry or update checks.

This repository maintains the Linux Python/Qt implementation of
[MouthKeys](https://github.com/BenevolentFutures/MouthKeys). The macOS app is maintained upstream.
See the
[Linux guide](linux/README.md) for installation, shortcuts, models, settings, text delivery,
privacy, recovery, and desktop validation.

## Install

Install Python/FFmpeg and desktop dependencies as described in the [guide](linux/README.md#install),
then:

```sh
git clone https://github.com/crypdick/MouthKeys-Linux.git
cd MouthKeys-Linux
uv tool install --python 3.13 './linux[gui]'
mouthkeys install-desktop
mouthkeys download
mouthkeys doctor
mouthkeys gui
```

Bind `mouthkeys toggle` to a desktop shortcut. Clipboard delivery is the default; paste with
your normal paste key after transcription. All configuration options are available in App Settings.

## Update an existing installation

Pull the latest code in your clone, then reinstall:

```sh
git pull --ff-only
uv tool install --force --python 3.13 './linux[gui]'
mouthkeys install-desktop
```

Stop the daemon with `mouthkeys shutdown` before updating if it is running, and reopen with
`mouthkeys gui` afterward. Reinstallation preserves your settings, history, and downloaded models.
Existing clones can update their remote with
`git remote set-url origin https://github.com/crypdick/MouthKeys-Linux.git`.

## Development

The runtime, tests, lockfile, and packaging live in `linux/`:

```sh
./linux/check.sh
```

Checks use fake capture and delivery adapters, isolated XDG directories, and Qt's offscreen
backend. They never open a live microphone, change the clipboard, or synthesize keys.
Desktop focus and actual dictation still require the manual checks in the Linux guide.

## Project lineage

MouthKeys for Linux is derived from [MouthKeys](https://github.com/BenevolentFutures/MouthKeys)
by Benevolent Futures, originally forked from [FluidVoice](https://github.com/altic-dev/FluidVoice)
by altic-dev. The macOS application remains in the upstream MouthKeys repository; this checkout
contains Linux code only. See [UPSTREAM.md](UPSTREAM.md) for attribution and porting policy.

Licensed under [GPL-3.0](LICENSE).
