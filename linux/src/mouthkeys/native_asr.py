"""Linux native speech backends using installed packages and explicit local artifacts."""

import importlib
import json
import math
import os
import re
import tarfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from .engines import READY, managed_path

_NEMOTRON_LANGUAGE_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")

# These names/classes are documented by stable Transformers 5.22.2 and NVIDIA NeMo 3.
# No remote Python model code is imported from a downloaded model repository.
TRANSFORMERS_MODELS = {
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "name": "Nemotron 3.5 ASR (offline)",
        "languages": "40 language-locales",
        "family": "nemotron3_5_asr",
        "revision": "main",
        "weight_bytes": 2552062944,
    },
    "nvidia/nemotron-speech-streaming-en-0.6b": {
        "name": "Nemotron Speech English",
        "languages": "English",
        "family": "nemotron_asr_streaming",
        "revision": "main",
        "weight_bytes": 2472413604,
    },
    "Qwen/Qwen3-ASR-0.6B-hf": {
        "name": "Qwen3 ASR 0.6B",
        "languages": "30 languages + Chinese dialects",
        "family": "qwen3_asr",
        "revision": "main",
        "weight_bytes": 1564928088,
    },
    "Qwen/Qwen3-ASR-1.7B-hf": {
        "name": "Qwen3 ASR 1.7B",
        "languages": "30 languages + Chinese dialects",
        "family": "qwen3_asr",
        "revision": "main",
        "weight_bytes": 4076193080,
    },
    "CohereLabs/cohere-transcribe-03-2026": {
        "name": "Cohere Transcribe (gated)",
        "languages": "14 languages; select explicitly",
        "family": "cohere_asr",
        "revision": "refs/pr/6",
        "weight_bytes": 4131862976,
    },
}
NEMO_MODELS = {
    "nvidia/parakeet_realtime_eou_120m-v1": {
        "name": "Parakeet Flash / Realtime EOU 120M",
        "languages": "English",
        "filename": "parakeet_realtime_eou_120m-v1.nemo",
        "revision": "main",
        "weight_bytes": 460062720,
    },
    "nvidia/nemotron-3.5-asr-streaming-0.6b": {
        "name": "Nemotron 3.5 ASR (NeMo)",
        "languages": "40 language-locales",
        "filename": "nemotron-3.5-asr-streaming-0.6b.nemo",
        "revision": "main",
        "weight_bytes": 2368284501,
    },
}
FAMILIES = {
    "nemotron3_5_asr": "AutoModelForRNNT",
    "nemotron_asr_streaming": "AutoModelForRNNT",
    "qwen3_asr": "AutoModelForMultimodalLM",
    "cohere_asr": "CohereAsrForConditionalGeneration",
}


def effective_revision(model, revision="main", engine="transformers"):
    catalog = TRANSFORMERS_MODELS if engine == "transformers" else NEMO_MODELS
    # Mainline Cohere weights predate its builtin Transformers integration. The
    # documented compatibility revision is displayed in the model catalog.
    return catalog.get(model, {}).get("revision", "main") if revision == "main" else revision


def artifact_path(model, engine, revision="main"):
    return managed_path(f"{model}@{effective_revision(model, revision, engine)}", engine)


def download_native(model, engine, revision="main", progress=None):
    from huggingface_hub import snapshot_download

    if engine not in ("transformers", "nemo"):
        raise ValueError("Unknown native speech engine")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    revision = effective_revision(model, revision, engine)
    path = artifact_path(model, engine, revision)
    (path / READY).unlink(missing_ok=True)
    patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]
    if engine == "nemo":
        patterns = [NEMO_MODELS.get(model, {}).get("filename", "*.nemo")]
    options = {}
    if progress:
        from .model_progress import progress_tqdm

        options["tqdm_class"] = progress_tqdm(progress)
    try:
        snapshot_download(
            model, revision=revision, local_dir=path, allow_patterns=patterns, **options
        )
    except Exception as exc:
        if type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
            raise RuntimeError(
                "Model access unavailable. Accept its Hugging Face conditions and configure "
                "HF_TOKEN "
                "before requesting the download again."
            ) from None
        raise
    if not any(path.glob("*.nemo" if engine == "nemo" else "*.safetensors")):
        raise ValueError("Downloaded repository has no supported model weights")
    if progress:
        progress(
            {"phase": "verifying model artifacts", "completed": 0, "total": 1, "unit": "stage"}
        )
    validate_artifacts(path, engine)
    (path / READY).touch(mode=0o600)
    if progress:
        progress({"phase": "ready", "completed": 1, "total": 1, "unit": "stage"})
    return str(path)


def validate_artifacts(path, engine):
    """Check companion files before advertising an explicit download as installed."""
    if engine == "nemo":
        import yaml

        artifacts = [path] if path.is_file() else list(path.glob("*.nemo"))
        if len(artifacts) != 1:
            raise ValueError("NeMo repository must contain exactly one model archive")
        with tarfile.open(artifacts[0], "r:*") as archive:
            members = {member.name.removeprefix("./"): member for member in archive}
            member = members.get("model_config.yaml")
            if member is None or not member.isfile() or member.size > 1024 * 1024:
                raise ValueError("NeMo model configuration missing or invalid")
            with archive.extractfile(member) as source:
                config = yaml.safe_load(source)
            if not any(name.endswith("model_weights.ckpt") for name in members):
                raise ValueError("NeMo model weights missing")
            seen = set()

            def check(value):
                if isinstance(value, dict | list):
                    if id(value) in seen:
                        return
                    seen.add(id(value))
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if key in ("target", "_target_") and (
                                not isinstance(child, str)
                                or not child.startswith(("nemo.", "torch."))
                            ):
                                raise ValueError(
                                    "NeMo archive requests a non-framework Python target"
                                )
                            check(child)
                    else:
                        for child in value:
                            check(child)
                elif isinstance(value, str) and value.startswith("nemo:"):
                    name = value[5:]
                    if name not in members or not members[name].isfile():
                        raise ValueError("NeMo tokenizer or auxiliary artifact is missing")

            check(config)
        return
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError("Native model config.json missing")
    if json.loads(config_path.read_text()).get("model_type") not in FAMILIES:
        raise ValueError("Downloaded model requires an unsupported or remote-code architecture")
    if not any(
        (path / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")
    ):
        raise ValueError("Native model audio processor configuration missing")
    if not any(
        (path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise ValueError("Native model tokenizer missing")
    for index in path.glob("*.safetensors.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map", {})
        if not mapping or any(
            not isinstance(name, str) or Path(name).name != name or not (path / name).is_file()
            for name in mapping.values()
        ):
            raise ValueError("Native model weight shard missing or invalid")


def local_artifact(config):
    path = Path(config.model).expanduser()
    if path.is_dir() or (config.engine == "nemo" and path.is_file()):
        return path
    path = artifact_path(config.model, config.engine, getattr(config, "model_revision", "main"))
    if not (path / READY).is_file():
        raise RuntimeError("Speech model missing; explicitly download it with mouthkeys download")
    return path


def torch_runtime(config):
    try:
        import torch
    except ImportError:
        raise RuntimeError(f"Install the {config.engine} optional dependency extra") from None
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; choose CPU or install a CUDA-enabled PyTorch build"
        )
    precision = config.compute_type
    if precision not in ("default", "auto", "float32", "float16", "bfloat16"):
        raise ValueError(
            "Native speech compute type must be default, auto, float32, float16, or bfloat16"
        )
    dtype = (
        "auto"
        if precision == "default"
        else getattr(
            torch,
            ("float16" if device == "cuda" else "float32") if precision == "auto" else precision,
        )
    )
    return torch, device, dtype


def speech_spans(samples):
    """Reuse packaged offline VAD to bound model input, preserving speech timestamps."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return get_speech_timestamps(
        samples,
        VadOptions(max_speech_duration_s=25, min_silence_duration_ms=500, speech_pad_ms=200),
    )


def _text(value):
    if isinstance(value, list):
        return " ".join(_text(item) for item in value).strip()
    if not isinstance(value, str):
        raise ValueError("Speech engine returned no text")
    return value.strip()


def _strip_decoder_prompt(generated, prompt):
    """Discard verified decoder conditioning, including an optional prepended BOS."""
    count = prompt.shape[1]
    expected = prompt[0].tolist()
    for offset in (0, 1):
        if generated[0, offset : offset + count].tolist() == expected:
            return generated[:, offset + count :]
    raise ValueError("Speech decoder returned an unexpected conditioning prefix")


class TransformersBackend:
    def __init__(self, config):
        path = local_artifact(config)
        torch, device, dtype = torch_runtime(config)
        try:
            module = importlib.import_module("transformers")
        except ImportError:
            raise RuntimeError("Install the transformers optional dependency extra") from None
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        details = json.loads((path / "config.json").read_text())
        self.family = details.get("model_type")
        if self.family not in FAMILIES:
            raise ValueError("Use a builtin Nemotron, Qwen3-ASR -hf, or Cohere ASR model")
        loader = getattr(module, FAMILIES[self.family])
        self.processor = module.AutoProcessor.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = (
            loader.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.torch = torch
        if self.processor.feature_extractor.sampling_rate != 16000:
            raise ValueError("Speech model must accept 16 kHz audio")

    def segments(self, samples, config, *, presegmented=False):
        if self.family == "cohere_asr" and config.language == "auto":
            raise ValueError("Cohere Transcribe requires an explicit language")
        output = []
        spans = [{"start": 0, "end": len(samples)}] if presegmented else speech_spans(samples)
        for span in spans:
            audio = samples[span["start"] : span["end"]]
            options = {"sampling_rate": 16000, "return_tensors": "pt"}
            language = None if config.language == "auto" else config.language
            if self.family == "qwen3_asr":
                from .vocabulary import context_prompt

                inputs = self.processor.apply_transcription_request(
                    audio=audio,
                    language=language,
                    prompt=context_prompt(config) or None,
                    **options,
                )
            else:
                # Nemotron 3.5 processor takes language locales; use the model's
                # language table rather than guessing the language's regional variant.
                if self.family == "nemotron3_5_asr":
                    language = nemotron_language(self.processor, config.language)
                elif self.family == "nemotron_asr_streaming":
                    if config.language not in ("en", "auto"):
                        raise ValueError("Nemotron Speech English supports English only")
                    language = None
                if language is not None:
                    options["language"] = language
                inputs = self.processor(audio, **options)
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=config.asr_max_tokens)
            generated = getattr(generated, "sequences", generated)
            if self.family == "qwen3_asr":
                generated = generated[:, inputs["input_ids"].shape[1] :]
                text = self.processor.decode(generated, return_format="transcription_only")
            else:
                if self.family == "cohere_asr" and "decoder_input_ids" in inputs:
                    generated = _strip_decoder_prompt(generated, inputs["decoder_input_ids"])
                text = self.processor.decode(generated, skip_special_tokens=True)
            text = _text(text)
            if text:
                output.append(
                    SimpleNamespace(
                        start=span["start"] / 16000,
                        end=span["end"] / 16000,
                        text=text,
                    )
                )
        return output


def nemotron_language(processor, language):
    if language == "auto":
        return "auto"
    # Processor validates locale keys. Stable release exposes this map in its
    # feature extractor configuration; fall back only to documented defaults.
    locales = getattr(processor, "language_to_id", None)
    if locales is None:
        locales = getattr(processor, "lang_to_id", None)
    if locales:
        if language in locales:
            return language
        matches = [
            locale for locale in locales if locale.lower().startswith(language.lower() + "-")
        ]
        if matches:
            return matches[0]
    default_locales = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "pt": "pt-PT",
        "de": "de-DE",
        "it": "it-IT",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh": "zh-CN",
        "ru": "ru-RU",
        "ar": "ar-AR",
        "hi": "hi-IN",
        "vi": "vi-VN",
        "uk": "uk-UA",
        "nl": "nl-NL",
        "tr": "tr-TR",
        "pl": "pl-PL",
        "sv": "sv-SE",
        "cs": "cs-CZ",
        "nb": "nb-NO",
        "da": "da-DK",
        "bg": "bg-BG",
        "fi": "fi-FI",
        "hr": "hr-HR",
        "sk": "sk-SK",
        "hu": "hu-HU",
        "ro": "ro-RO",
        "et": "et-EE",
        "el": "el-GR",
        "lt": "lt-LT",
        "lv": "lv-LV",
        "mt": "mt-MT",
        "sl": "sl-SI",
        "he": "he-IL",
        "th": "th-TH",
        "nn": "nn-NO",
    }
    return default_locales.get(language, language)


class NemoBackend:
    def __init__(self, config):
        path = local_artifact(config)
        if path.is_dir():
            candidates = list(path.glob("*.nemo"))
            if len(candidates) != 1:
                raise ValueError("NeMo model directory must contain exactly one .nemo artifact")
            path = candidates[0]
        validate_artifacts(path, "nemo")
        self.torch, device, dtype = torch_runtime(config)
        try:
            module = importlib.import_module("nemo.collections.asr.models")
        except ImportError:
            raise RuntimeError("Install the nemo optional dependency extra") from None
        self.model = module.ASRModel.restore_from(str(path), map_location=device).to(device).eval()
        if dtype != "auto":
            self.model = self.model.to(dtype=dtype)
        if getattr(config, "native_streaming", "on") == "on":
            if config.compute_type not in ("default", "auto", "float32"):
                raise ValueError("NeMo cache-aware streaming requires float32 compute type")
            self.model = self.model.to(dtype=self.torch.float32)
        latency = getattr(config, "streaming_latency", "model")
        if latency != "model":
            if not hasattr(self.model.encoder, "set_default_att_context_size"):
                raise ValueError("This model does not support configurable streaming latency")
            right = {"80": 0, "160": 1, "320": 3, "560": 6, "1120": 13}[latency]
            left = getattr(self.model.encoder, "att_context_size", [56])[0]
            self.model.encoder.set_default_att_context_size(att_context_size=[left, right])
        self._base_decoding = None
        self._vocabulary_state = None
        self.reset_stream()

    def _vocabulary(self, config):
        """Use NeMo's weighted context graph, not textual prompt emulation."""
        from .vocabulary import VocabularyStore

        entries = VocabularyStore.entries() if config.vocabulary_boosting == "on" else []
        timed = config.pronunciation_matching == "on"
        state = (tuple((entry["text"], entry["weight"]) for entry in entries), timed)
        if state == self._vocabulary_state or (
            not entries and self._base_decoding is None and not timed
        ):
            return
        from omegaconf import OmegaConf

        if self._base_decoding is None:
            self._base_decoding = OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
        settings = deepcopy(self._base_decoding)
        if timed:
            settings.update(
                compute_timestamps=True, preserve_alignments=False, rnnt_timestamp_type="word"
            )
        if entries:
            from nemo.collections.asr.parts.context_biasing import GPUBoostingTreeModel
            from nemo.collections.asr.parts.context_biasing.context_graph_universal import (
                ContextGraph,
            )

            directory = os.environ.get("MOUTHKEYS_NATIVE_STATE")
            if not directory:
                raise RuntimeError("Native vocabulary boosting requires the isolated model worker")
            graph = ContextGraph(context_score=1.0, depth_scaling=2.0)
            tokens = [self.model.tokenizer.text_to_ids(entry["text"]) for entry in entries]
            if any(not token_ids for token_ids in tokens):
                raise ValueError("Native vocabulary contains a phrase the tokenizer cannot encode")
            graph.build(
                token_ids=tokens,
                scores=[float(entry["weight"]) for entry in entries],
                phrases=[entry["text"] for entry in entries],
                uniform_weights=False,
            )
            tree = GPUBoostingTreeModel.from_context_graph(
                context_graph=graph,
                vocab_size=self.model.tokenizer.vocab_size,
                use_triton=False,
                uniform_weights=False,
            )
            path = Path(directory) / "vocabulary.nemo"
            tree.save_to(str(path))
            path.chmod(0o600)
            settings["strategy"] = "greedy_batch"
            settings.setdefault("greedy", {}).update(
                boosting_tree={"model_path": str(path), "use_triton": False},
                boosting_tree_alpha=1.0,
            )
        self.model.change_decoding_strategy(OmegaConf.create(settings))
        self._vocabulary_state = state

    def reset_stream(self):
        """Called only while the controller holds its model lock."""
        self.stream_key = None
        self.buffer = None
        self.caches = None
        self.previous_hypotheses = None
        self.predictions = None
        self.stream_text = ""
        self.stream_samples = 0
        self.words = []

    def _word_times(self, hypothesis, offset=0, duration=None):
        timestamps = getattr(hypothesis, "timestamp", None)
        if not isinstance(timestamps, dict):
            return []
        scale = float(self.model.encoder.subsampling_factor) * float(
            self.model.cfg.preprocessor.window_stride
        )
        words = []
        for item in timestamps.get("word", []):
            # Native RNNT offsets are encoder frames, not waveform samples. Read
            # them without mutating partial hypotheses reused by cache-aware decoding.
            start, end = float(item["start_offset"]) * scale, float(item["end_offset"]) * scale
            if duration is not None:
                # Encoder frames have quantized right edges; the final frame
                # may extend beyond the actual (unpadded) recording boundary.
                end = min(end, duration)
            text = self._transcript_text(str(item["word"]))
            if text and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end:
                words.append({"word": text, "start": start + offset, "end": end + offset})
        return words

    def _transcript_text(self, text):
        if hasattr(self.model, "set_inference_prompt"):
            # NeMo strips language tags from decoded text, but its independent
            # timestamp renderer can still emit them as word records.
            text = _NEMOTRON_LANGUAGE_TAG.sub("", text)
        return text.replace("<EOU>", "").strip()

    def _language(self, config):
        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(nemotron_language(self.model, config.language))
            self.model.decoding.set_strip_lang_tags(True)
        elif "parakeet_realtime_eou" in config.model and config.language not in ("en", "auto"):
            raise ValueError("Parakeet Flash / Realtime EOU supports English only")

    def stream(self, samples, config, session_key, *, final=False):
        """Advance native encoder/decoder caches; never re-encode committed chunks.

        NeMo's own frontend is recomputed on the current prefix so STFT boundary
        handling remains its responsibility. Only new complete feature chunks
        enter conformer_stream_step. Retain a short right margin until finalization.
        """
        if self.stream_key != session_key:
            self.reset_stream()
            self._vocabulary(config)
            self._language(config)
            module = importlib.import_module("nemo.collections.asr.parts.utils.streaming_utils")
            preprocessing = getattr(getattr(self.model, "cfg", None), "preprocessor", None)
            normalization = getattr(preprocessing, "normalize", None)
            self.buffer = module.CacheAwareStreamingAudioBuffer(
                model=self.model,
                online_normalization=normalization in ("per_feature", "all_features"),
            )
            self.caches = self.model.encoder.get_initial_cache_state(batch_size=1)
            self.stream_key = session_key
        if len(samples) < self.stream_samples:
            raise ValueError("Recording shrank during native streaming")
        self.stream_samples = len(samples)
        processed, lengths = self.buffer.preprocess_audio(samples)
        self.buffer.buffer = processed
        self.buffer.streams_length = lengths
        # Final partial chunks need enough feature frames for the pre-encoder.
        minimum = getattr(self.buffer, "sampling_frames", None)
        if isinstance(minimum, list):
            minimum = minimum[0 if self.buffer.buffer_idx == 0 else 1]
        remaining = processed.size(-1) - self.buffer.buffer_idx
        if final and minimum and 0 < remaining < minimum:
            count = minimum - remaining
            self.buffer.buffer = self.torch.nn.functional.pad(processed, (0, count))
            self.buffer.streams_length = lengths + count
        iterator = iter(self.buffer)
        while self.buffer.buffer_idx < self.buffer.buffer.size(-1):
            cfg = self.buffer.streaming_cfg
            chunk_size = cfg.chunk_size
            if isinstance(chunk_size, list):
                chunk_size = chunk_size[0 if self.buffer.buffer_idx == 0 else 1]
            remaining = self.buffer.buffer.size(-1) - self.buffer.buffer_idx
            if not final and remaining < chunk_size + 4:
                break  # Future audio may revise the frontend's rightmost STFT frames.
            last = final and remaining <= chunk_size
            try:
                chunk, chunk_lengths = next(iterator)
            except StopIteration:
                break
            drop = 0 if self.buffer.step == 1 else cfg.drop_extra_pre_encoded
            with self.torch.inference_mode():
                result = self.model.conformer_stream_step(
                    processed_signal=chunk.to(self.model.dtype),
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self.caches[0],
                    cache_last_time=self.caches[1],
                    cache_last_channel_len=self.caches[2],
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.predictions,
                    keep_all_outputs=last,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            (self.predictions, texts, channel, time_cache, cache_len, self.previous_hypotheses) = (
                result
            )
            self.caches = channel, time_cache, cache_len
            values = [getattr(text, "text", text) for text in texts]
            self.stream_text = self._transcript_text(_text(values))
            if config.pronunciation_matching == "on" and texts:
                self.words = self._word_times(texts[0], duration=len(samples) / 16000)
                # NeMo timestamp rendering replaces the token timestep list by
                # a dictionary. Its next partial-hypothesis merge expects the
                # original list; restore only that representation after reading
                # word offsets, preserving the encoder/decoder caches unchanged.
                for hypothesis in self.previous_hypotheses or []:
                    timestamp = getattr(hypothesis, "timestamp", None)
                    if isinstance(timestamp, dict):
                        hypothesis.timestamp = timestamp["timestep"]
            if last:
                break
        text = self.stream_text
        if final:
            words = self.words
            self.reset_stream()
            self.words = words
        return text

    def segments(self, samples, config, *, presegmented=False):
        if self.stream_key is not None:
            raise RuntimeError("Native recording is active; finish it before file transcription")
        self._vocabulary(config)
        self._language(config)
        output = []
        self.words = []
        spans = [{"start": 0, "end": len(samples)}] if presegmented else speech_spans(samples)
        for span in spans:
            audio = samples[span["start"] : span["end"]]
            with self.torch.inference_mode():
                results = self.model.transcribe(
                    audio=[audio],
                    batch_size=1,
                    return_hypotheses=True,
                    verbose=False,
                )
            if isinstance(results, tuple):
                results = results[0]
            text = self._transcript_text(_text(getattr(results[0], "text", results[0])))
            words = (
                self._word_times(results[0], span["start"] / 16000, len(audio) / 16000)
                if config.pronunciation_matching == "on"
                else []
            )
            self.words.extend(words)
            if text:
                output.append(
                    SimpleNamespace(
                        start=span["start"] / 16000,
                        end=span["end"] / 16000,
                        text=text,
                        words=words,
                    )
                )
        return output
