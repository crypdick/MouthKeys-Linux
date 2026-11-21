"""Isolated local NeMo inference and cache-aware streaming."""

import importlib
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import native_asr
from .config import Config
from .native_asr import NativeWord
from .vocabulary import VocabularyStore

_LANGUAGE_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")


class NemoBackend:
    def __init__(self, config: Config) -> None:
        path = native_asr.local_artifact(config)
        if path.is_dir():
            candidates = list(path.glob("*.nemo"))
            if len(candidates) != 1:
                raise ValueError("NeMo model directory must contain exactly one .nemo artifact")
            path = candidates[0]
        native_asr.validate_artifacts(path, "nemo")
        self.torch, device, dtype = native_asr.torch_runtime(config)
        try:
            from nemo.collections.asr.models import ASRModel  # noqa: PLC0415 - optional SDK
        except ImportError:
            raise RuntimeError("Install the nemo optional dependency extra") from None
        self.model = ASRModel.restore_from(str(path), map_location=device).to(device).eval()
        if dtype != "auto":
            self.model = self.model.to(dtype=dtype)
        self.supports_native_streaming = native_asr.supports_native_streaming(self.model)
        streaming = native_asr.uses_native_streaming(config) and self.supports_native_streaming
        if streaming:
            if config.compute_type not in ("default", "auto", "float32"):
                raise ValueError("NeMo cache-aware streaming requires float32 compute type")
            self.model = self.model.to(dtype=self.torch.float32)
        latency = getattr(config, "streaming_latency", "model")
        if latency != "model" and streaming:
            if not hasattr(self.model.encoder, "set_default_att_context_size"):
                raise ValueError("This model does not support configurable streaming latency")
            right = {"80": 0, "160": 1, "320": 3, "560": 6, "1120": 13}[latency]
            left = getattr(self.model.encoder, "att_context_size", [56])[0]
            self.model.encoder.set_default_att_context_size(att_context_size=[left, right])
        self._base_decoding: dict[str, object] | None = None
        self._vocabulary_state: tuple[tuple[tuple[str, float | None], ...], bool] | None = None
        self.reset_stream()

    def _vocabulary(self, config: Config) -> None:
        """Use NeMo's weighted context graph, not textual prompt emulation."""
        entries = VocabularyStore.effective_entries() if config.vocabulary_boosting == "on" else []
        timed = config.pronunciation_matching == "on"
        state = (tuple((entry["text"], entry["weight"]) for entry in entries), timed)
        if state == self._vocabulary_state or (
            not entries and self._base_decoding is None and not timed
        ):
            return
        from omegaconf import OmegaConf  # noqa: PLC0415 - optional SDK

        if self._base_decoding is None:
            base = OmegaConf.to_container(self.model.cfg.decoding, resolve=True)
            if not isinstance(base, dict):
                raise ValueError("NeMo decoding configuration must be a mapping")
            self._base_decoding = base
        settings = deepcopy(self._base_decoding)
        if timed:
            settings.update(
                compute_timestamps=True, preserve_alignments=False, rnnt_timestamp_type="word"
            )
        if entries:
            greedy = settings.setdefault("greedy", {})
            if not isinstance(greedy, dict):
                raise ValueError("NeMo greedy decoding configuration must be a mapping")
            directory = os.environ.get("MOUTHKEYS_NATIVE_STATE")
            if not directory:
                raise RuntimeError("Native vocabulary boosting requires the isolated model worker")
            biasing = importlib.import_module("nemo.collections.asr.parts.context_biasing")
            graph_module = importlib.import_module(
                "nemo.collections.asr.parts.context_biasing.context_graph_universal"
            )
            graph = graph_module.ContextGraph(context_score=1.0, depth_scaling=2.0)
            tokens = [self.model.tokenizer.text_to_ids(entry["text"]) for entry in entries]
            if any(not token_ids for token_ids in tokens):
                raise ValueError("Native vocabulary contains a phrase the tokenizer cannot encode")
            graph.build(
                token_ids=tokens,
                scores=[float(entry["weight"] or 1) for entry in entries],
                phrases=[entry["text"] for entry in entries],
                uniform_weights=False,
            )
            tree = biasing.GPUBoostingTreeModel.from_context_graph(
                context_graph=graph,
                vocab_size=self.model.tokenizer.vocab_size,
                use_triton=False,
                uniform_weights=False,
            )
            path = Path(directory) / "vocabulary.nemo"
            tree.save_to(str(path))
            path.chmod(0o600)
            settings["strategy"] = "greedy_batch"
            greedy.update(
                boosting_tree={"model_path": str(path), "use_triton": False},
                boosting_tree_alpha=1.0,
            )
        self.model.change_decoding_strategy(OmegaConf.create(settings))
        self._vocabulary_state = state

    def reset_stream(self) -> None:
        """Called only while the controller holds its model lock."""
        self.stream_key: object = None
        # NeMo owns the toolkit-specific tensor/cache representations.
        self.buffer: Any = None
        self.caches: Any = None
        self.previous_hypotheses: Any = None
        self.predictions: Any = None
        self.stream_text = ""
        self.stream_samples = 0
        self.words: list[NativeWord] = []

    def _word_times(
        self, hypothesis: object, offset: float = 0, duration: float | None = None
    ) -> list[NativeWord]:
        timestamps = getattr(hypothesis, "timestamp", None)
        if not isinstance(timestamps, dict):
            return []
        scale = float(self.model.encoder.subsampling_factor) * float(
            self.model.cfg.preprocessor.window_stride
        )
        words: list[NativeWord] = []
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

    def _transcript_text(self, text: str) -> str:
        if hasattr(self.model, "set_inference_prompt"):
            # NeMo strips language tags from decoded text, but its independent
            # timestamp renderer can still emit them as word records.
            text = _LANGUAGE_TAG.sub("", text)
        return text.replace("<EOU>", "").strip()

    def _language(self, config: Config) -> None:
        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(
                native_asr.nemotron_language(self.model, config.language)
            )
            self.model.decoding.set_strip_lang_tags(True)
        elif (
            "parakeet_realtime_eou" in config.model or config.model == "nvidia/parakeet-tdt-0.6b-v2"
        ) and config.language not in ("en", "auto"):
            raise ValueError("Selected English Parakeet model supports English only")

    def _start_stream(self, config: Config, session_key: object) -> None:
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

    def _prepare_features(self, samples: NDArray[np.generic], *, final: bool) -> None:
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

    def stream(
        self,
        samples: NDArray[np.generic],
        config: Config,
        session_key: object,
        *,
        final: bool = False,
    ) -> str:
        """Advance native encoder/decoder caches; never re-encode committed chunks.

        NeMo's own frontend is recomputed on the current prefix so STFT boundary
        handling remains its responsibility. Only new complete feature chunks
        enter conformer_stream_step. Retain a short right margin until finalization.
        """
        if self.stream_key != session_key:
            self._start_stream(config, session_key)
        if len(samples) < self.stream_samples:
            raise ValueError("Recording shrank during native streaming")
        self.stream_samples = len(samples)
        self._prepare_features(samples, final=final)
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
            self.stream_text = self._transcript_text(native_asr.transcript_text(values))
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

    def segments(
        self, samples: NDArray[np.generic], config: Config, *, presegmented: bool = False
    ) -> list[SimpleNamespace]:
        if self.stream_key is not None:
            raise RuntimeError("Native recording is active; finish it before file transcription")
        self._vocabulary(config)
        self._language(config)
        output = []
        self.words = []
        spans = (
            [{"start": 0, "end": len(samples)}]
            if presegmented
            else native_asr.speech_spans(samples)
        )
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
            text = self._transcript_text(
                native_asr.transcript_text(getattr(results[0], "text", results[0]))
            )
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
