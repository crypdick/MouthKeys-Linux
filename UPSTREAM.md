# Upstream and attribution

This Linux implementation is maintained in [crypdick/MouthKeys-Linux](https://github.com/crypdick/MouthKeys-Linux).
It derives from [BenevolentFutures/MouthKeys](https://github.com/BenevolentFutures/MouthKeys),
the macOS dictation application by Benevolent Futures, originally forked from
[altic-dev/FluidVoice](https://github.com/altic-dev/FluidVoice) by altic-dev.

MouthKeys' visual language, grin artwork, and interaction design informed the Linux Qt interface.
The upstream macOS sources, artwork-generation scripts, design prototypes, and historical port
ledger remain available in this repository's Git history and in the upstream repository.
Removing macOS build products and sources does not change the GPL-3.0 license or attribution.

## Porting policy

- Keep `origin` pointing to the Linux fork and `mouthkeys-upstream` to BenevolentFutures/MouthKeys.
- Review upstream changes and adapt relevant behavior by hand. Do not merge the macOS tree back in.
- Credit a port with its upstream repository, commit SHA, subject, and original author in the commit.
- Target upstream feature parity with Linux-native affordances. Local transcription stays available;
  cloud AI and Spoken Send are optional, configurable features, not prohibited. Keep telemetry and
  upstream auto-updaters off. CLI/i3 shortcuts remain supported; integrated shortcuts are lower priority.
- Open Linux PRs against `crypdick/MouthKeys-Linux --base main`, never the parent repository or its site.

## Linux parity adaptation, 2026-11-09

Baseline: `BenevolentFutures/MouthKeys@d50ad008ecd6b734e8fb3e9ab7ee942e3f6e388d`.
Linux implementations adapt upstream behavior rather than its Swift/platform APIs:

- Atin, `a40080cd9275ae68a69efd52784d579a5ad6d185`, “Spoken Send everywhere, a quicker
  stop, and Q for a question mark”: configurable send parsing and spoken shortcuts.
- Atin, `3ebcd840d6a7de1a7dd622ba997ae21928e44147`, “Settings split into Hotkeys,
  Microphone and App Settings; microphone groups”: microphone priorities and configurable settings.
- Atin, `8ab26a8f1c6041367050daaabd9eccfbd364ac62`, “Typing: queue concurrent dictations
  instead of dropping, bound AX stalls”: independent queued jobs and ordered delivery.

Dictionary/formatting, history, profiles, file tools, and desktop affordances follow the same
baseline's user-facing contracts with Linux-native alternatives. Optional inference integrations
use onnx-asr and sherpa-onnx rather than Apple's frameworks. Current support and outstanding
gaps have one canonical home: [Linux guide — parity and limits](linux/README.md#parity-and-limits).
