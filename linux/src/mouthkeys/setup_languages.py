"""Language-first Linux routes, adapted from upstream VoiceEngineLanguageCatalog.

Only documented language sets are offered. Whisper uses the locked faster-whisper
language inventory (including Hebrew's Linux code `he`); Apple routes are replaced
by local Linux engines. Routes never load a model or contact its provider.
"""

LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Myanmar",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Mandarin Chinese",
    "yue": "Cantonese",
}
LANGUAGE_ALIASES = {
    "ar": ["Arab"],
    "bn": ["Bangla"],
    "de": ["Deutsch"],
    "es": ["Castilian"],
    "fa": ["Farsi"],
    "my": ["Burmese"],
    "no": ["Norwegian Bokmal"],
    "ro": ["Moldavian", "Moldovan"],
    "si": ["Sinhalese"],
    "tl": ["Filipino"],
    "zh": ["Chinese", "Mandarin"],
}
PARAKEET_V3 = set(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)
COHERE = set("ar de el en es fr it ja ko nl pl pt vi zh".split())
NEMOTRON = {
    code.split("-", 1)[0] if code != "nb-NO" else "no": code
    for code in (
        "es it pt hi ko en de-DE fr ru tr vi-VN nl ja-JP ar uk pl nb-NO fi zh-CN cs bg "
        "sk sv hr ro et da hu el-GR he-IL lt-LT sl-SI lv-LV mt-MT th-TH nn-NO"
    ).split()
}


def languages():
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    return [(code, LANGUAGE_NAMES[code]) for code in _LANGUAGE_CODES]


def routes(language):
    """Return explicit engine/model/language/revision bindings, not quality ratings."""
    if language not in dict(languages()):
        return []
    choices = []

    def add(label, engine, model, language_code=language, revision="main"):
        choices.append(
            {
                "label": label,
                "engine": engine,
                "model": model,
                "language": language_code,
                "model_revision": revision,
            }
        )

    if language == "en":
        add("Parakeet TDT v2 · ONNX · English", "onnx", "nemo-parakeet-tdt-0.6b-v2")
        add("Parakeet Flash · NeMo · English", "nemo", "nvidia/parakeet_realtime_eou_120m-v1")
        add(
            "Parakeet TDT v2 · NeMo · native vocabulary / word times",
            "nemo",
            "nvidia/parakeet-tdt-0.6b-v2",
        )
    if language in PARAKEET_V3:
        add("Parakeet TDT v3 · ONNX · multilingual", "onnx", "nemo-parakeet-tdt-0.6b-v3")
        add(
            "Parakeet TDT v3 · NeMo · native vocabulary / word times",
            "nemo",
            "nvidia/parakeet-tdt-0.6b-v3",
        )
    if language in COHERE:
        add(
            "Cohere Transcribe · Transformers · gated download",
            "transformers",
            "CohereLabs/cohere-transcribe-03-2026",
            revision="refs/pr/6",
        )
    if language in NEMOTRON:
        code = NEMOTRON[language]
        add(
            "Nemotron 3.5 · NeMo · streaming/offline",
            "nemo",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            code,
        )
        add(
            "Nemotron 3.5 · Transformers · offline",
            "transformers",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            code,
        )
    # Cantonese has its own language token only in the v3 vocabulary.
    for model in ("large-v3-turbo",) if language == "yue" else ("small", "large-v3-turbo"):
        add(f"Whisper {model} · multilingual", "whisper", model)
    return choices
