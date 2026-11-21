"""Synthetic fixture with upstream backup schema-1 keys."""

PROFILE_ID = "a90dbe73-4ab7-42dc-acfc-f51d3293ab13"
HISTORY_ID = "c66ae138-d0a2-47c1-aeab-6bd1f23f81aa"


def upstream_document():
    # Shapes/keys from upstream BackupService.swift and TranscriptionHistoryEntry.
    return {
        "schemaVersion": {"major": 1, "minor": 0},
        "appVersion": "0.1.5",
        "exportedAt": "2026-11-01T19:40:09Z",
        "settings": {
            "selectedProviderID": "ollama",
            "selectedModelByProvider": {"ollama": "existing-local-model"},
            "savedProviders": [],
            "selectedSpeechModel": "whisper-medium",
            "customDictionaryEntries": [
                {
                    "id": "df59fc30-6fe5-4578-804a-137e052c3bf8",
                    "triggers": ["mouth keys"],
                    "replacement": "MouthKeys",
                },
            ],
            "fillerWords": ["um", "uh"],
            "removeFillerWordsEnabled": True,
            "overlayBottomOffset": 50.0,
            "overlayPosition": "top",
            "transcriptionSoundVolume": 0.25,
            "transcriptionStartSound": "fluid_sfx_2",
            "weekendsDontBreakStreak": True,
            "saveTranscriptionHistory": True,
            "defaultDictationPromptOverride": "Clean only; do not answer.",
            "selectedDictationPromptID": PROFILE_ID.upper(),
            "selectedEditPromptID": PROFILE_ID,
            "dictationPromptRoutingScope": "selectedAppsOnly",
            "hotkeyShortcut": {
                "kind": "keyboard",
                "keyCode": 49,
                "modifierFlagsRawValue": 1048576,
                "modifierKeyCodes": [],
            },
            "preferredInputDeviceUID": "AppleUSBAudioEngine:some-device",
            "shareAnonymousAnalytics": True,
            "autoUpdateCheckEnabled": True,
            "commandModeConfirmBeforeExecute": False,
            "textInsertionMode": "reliablePaste",
            "pronunciationMatchingEnabled": False,
        },
        "promptProfiles": [
            {
                "id": PROFILE_ID.upper(),
                "name": "Concise",
                "prompt": "Shorten selected text.",
                "mode": "edit",
                "includeContext": True,
                "createdAt": "2026-10-01T01:02:03Z",
                "updatedAt": "2026-10-02T01:02:03Z",
            }
        ],
        "appPromptBindings": [
            {
                "id": "e0d5d6aa-bb9a-4c2d-aecc-857ec57e93fa",
                "mode": "edit",
                "appBundleID": "com.example.Editor",
                "appName": "Editor",
                "promptID": PROFILE_ID,
            }
        ],
        "transcriptionHistory": [
            {
                "id": HISTORY_ID.upper(),
                "timestamp": "2026-11-01T12:00:00-08:00",
                "rawText": "raw private words",
                "processedText": "Final private words.",
                "appName": "Editor",
                "windowTitle": "notes",
                "characterCount": 20,
                "wordCount": 3,
                "wasAIProcessed": True,
                "processingModel": "actual-cleanup-model",
                "aiProcessingError": None,
                "audio": {
                    "fileName": "not-present.wav",
                    "durationMilliseconds": 2500,
                    "byteCount": 80000,
                    "sampleRate": 16000,
                    "channels": 1,
                    "model": "whisper-medium",
                },
            }
        ],
        "pronunciationProfiles": [{"modelKey": "parakeet-v3", "hiddenSize": 1024}],
    }
