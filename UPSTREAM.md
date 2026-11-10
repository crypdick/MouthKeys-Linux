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

Dictionary/formatting, history, profiles, file tools, and desktop affordances follow that baseline's
user-facing contracts with Linux-native alternatives. Further hand-adapted behavior includes:

- `AutomaticDictionaryCorrectionTracker` / suggestion policy: anchored actual corrections,
  repeat/cooldown rules, explicit review, and opt-in automatic acceptance. Linux observes bounded
  AT-SPI snapshots rather than macOS AX notifications; full field contents are not persisted.
- `PronunciationDictionaryStore` / FluidAudio acoustic corrections: private multiple-take enrollment
  and matched word-span replacement. Linux uses cepstral/DTW templates with actual Whisper/NeMo word
  timings, ambiguity/edge/overlap guards and bounded work, not Apple's model-specific embeddings.
- `ParakeetVocabularyStore`: editable terms, weights and aliases. NeMo supports real weighted context
  graphs; Whisper/Qwen use explicitly described context hints, not claimed FST equivalence.
- `LarkA1Protocol`: Hollyland receiver status/heartbeat adaptation. Original protocol decoding credit
  also belongs to `pedrozimmermannsoares/vorssaint-utils#2510`; the Linux port checks the status-report
  descriptor and does not expose firmware/settings writes.
- `MeetingTranscriptionService`, prompt routing, media/sounds, setup, shortcuts and overlay/history:
  turn-first speaker transcription, per-mode/application/shortcut profiles, explicit provider tests,
  cancellable streaming, bounded persistent/private-memory multi-chat follow-ups, reviewed
  working-directory/tool-result continuation with execution receipts, Linux-native guarded delivery,
  and private portable data tools.

Native speech integrations use faster-whisper, onnx-asr, builtin Transformers and NeMo classes;
speaker labeling uses sherpa-onnx. Apple/CoreML APIs remain upstream. All five recording-sound
choices are converted from the same upstream assets, with separate recording-enable/preview controls;
GPL attribution and license remain in effect.

The fresh settings/reachable-flow audit additionally adapts these baseline contracts explicitly:

- `SettingsStore` default punctuation dictionary: all 40 symbol rows and aliases, generated-symbol
  spacing/quote state, generated-comma/action-adjacent ASR punctuation cleanup, configurable literal
  prefix, and existing-user-dictionary preservation.
- `ASRService+DictationLiteralFormatting`: guarded slash command/path handling, explicit and real
  chat-app contextual mentions, and app-specific standalone autocomplete spacing. Continuous
  dictation appends trailing spacing and reads sentence boundaries through closing wrappers.
- `RegionalFillerOffer` and `VoiceEngineLanguageCatalog`: ordered passive region/timezone detection,
  one-time inline keep/remove answers in Settings/tryout, Minnesota wording informed by actual
  tryout text, and explicit language→compatible Linux route selection without automatic downloads.
- `NotificationService`: configurable AI-failure notifications independent of recoverable raw text.
- Custom-only dictation prompts: the exact upstream base asset is combined with the effective
  global/profile body by default, or omitted explicitly with `ai_custom_prompt_only`; composition
  is shared by normal cleanup and typed/voice playgrounds, without changing Command/Edit prompts.
- Hide-from-Dock/App-Switcher: Linux exposes `hide_from_taskbar` as the main window's Qt Tool hint,
  independent of tray visibility; actual taskbar/switcher treatment belongs to the compositor.
- `CommandModeService` confirmation behavior: upstream tests a destructive-command text heuristic
  when its confirmation switch is enabled, not a blanket always-confirm restriction. Linux keeps
  every-command review as default and exposes heuristic-only/automatic policies explicitly, applies
  policy to typed and voice proposals, bounds continuation to 20 steps, and retains execution receipts
  independently of next-proposal requests. The heuristic is not a shell parser or sandbox.
- Command/Edit notch interaction is adapted to an explicitly opened, focusable assistant surface
  beside the passive Linux panel, with conversations/results/review controls. Reviewed Edit accepts
  only the unchanged captured selection, supports explicit focus return, and consumes each proposal
  once before delivery; unsupported/stale destinations remain copyable.
- File transcription reports actual stage/chunk progress (or indeterminate state), offers explicit
  result copy, and preserves clean CLI transcript stdout when structured stderr progress is enabled.

[linux/PARITY.md](linux/PARITY.md) is the canonical acceptance ledger; the
[Linux guide](linux/README.md) documents operation and limits. The objective is to bridge all
bridgeable gaps, not to label optional features forbidden or to turn mock/tiny-model tests into
claims of real desktop behavior or pretrained accuracy. Actual tiny native-library smoke summaries
are retained in [linux/tests/native_smoke_evidence.json](linux/tests/native_smoke_evidence.json).
Separate [pretrained Whisper fixture evidence](linux/tests/whisper_smoke_evidence.json) uses already
available large-v3/turbo snapshots with Internet sockets denied: greeting recognition, real word
offsets and committed-preview continuation; turbo also exercises private file-history and TXT/JSON
export. This is a scoped public-fixture regression, not a general accuracy/resource/language claim.
