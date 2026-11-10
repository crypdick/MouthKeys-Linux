"""Opt-in, offline native ASR functional smoke (never an accuracy benchmark).

CPU-only development environment, without modifying the installed application::

    uv venv --python 3.13 /tmp/mouthkeys-native-smoke-env
    uv pip install --python /tmp/mouthkeys-native-smoke-env/bin/python \
        --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchaudio==2.11.0
    uv pip install --python /tmp/mouthkeys-native-smoke-env/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu -e './linux[transformers,nemo]'
    /tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
        --build-tiny qwen --artifact /tmp/mouthkeys-tiny-qwen
    /tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
        --build-tiny cohere --artifact /tmp/mouthkeys-tiny-cohere
    /tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
        --build-tiny nemo --artifact /tmp/mouthkeys-tiny-nemo --stream --word-times
    /tmp/mouthkeys-native-smoke-env/bin/python linux/tests/smoke_native.py \
        --engine nemo --artifact /absolute/path/model.nemo --stream --stream-repeat 3 \
        --word-times --expect-greeting --report /tmp/mouthkeys-native-acceptance.json

Use --artifact with an existing local model to test a real checkpoint. Tiny
builders use installed library classes and synthetic tokenizers/random weights;
no Hub access, metadata downloads, microphones, or desktop integration. Their
output is deliberately NOT evidence of pretrained transcription accuracy.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path


def build_tiny(family, path):
    """Build real builtin models/processors, entirely offline and small enough for CPUs."""
    if path.exists():
        raise ValueError("Tiny artifact destination already exists; choose an unused directory")
    if family == "nemo":
        return build_tiny_nemo(path)
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(1)
    tokens = [
        "<pad>",
        "<unk>",
        "<eos>",
        "<bos>",
        "<|audio_pad|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<timestamp>",
        "<asr_text>",
        "language",
        "English",
        "hello",
        "world",
        "▁",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        "<|en|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|noitn|>",
        "<|notimestamp|>",
        "<|nodiarize|>",
    ]
    backend = Tokenizer(
        models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        extra_special_tokens={
            "audio_token": "<|audio_pad|>",
            "audio_bos_token": "<|audio_start|>",
            "audio_eos_token": "<|audio_end|>",
        },
    )
    if family == "qwen":
        from transformers import (
            Qwen3ASRConfig,
            Qwen3ASREncoderConfig,
            Qwen3ASRFeatureExtractor,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
            Qwen3Config,
        )

        template = (
            "{% for message in messages %}{% for item in message['content'] %}"
            "{% if item['type'] == 'audio' %}"
            "<|audio_start|><|audio_pad|><|audio_end|>"
            "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
        )
        processor = Qwen3ASRProcessor(Qwen3ASRFeatureExtractor(), tokenizer, chat_template=template)
        config = Qwen3ASRConfig(
            audio_config=Qwen3ASREncoderConfig(
                d_model=16,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=32,
                output_dim=16,
                downsample_hidden_size=8,
            ),
            text_config=Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                pad_token_id=0,
                eos_token_id=2,
            ),
            audio_token_id=4,
            timestamp_token_id=7,
            pad_token_id=0,
            eos_token_id=2,
        )
        model = Qwen3ASRForConditionalGeneration(config)
    else:
        from transformers import (
            CohereAsrConfig,
            CohereAsrFeatureExtractor,
            CohereAsrForConditionalGeneration,
            CohereAsrProcessor,
            ParakeetEncoderConfig,
        )

        processor = CohereAsrProcessor(CohereAsrFeatureExtractor(), tokenizer)
        config = CohereAsrConfig(
            encoder_config=ParakeetEncoderConfig(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                subsampling_conv_channels=8,
                num_mel_bins=128,
            ),
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=128,
            pad_token_id=0,
            eos_token_id=2,
            bos_token_id=3,
            decoder_start_token_id=3,
        )
        model = CohereAsrForConditionalGeneration(config)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    (path / "mouthkeys-smoke-artifact.json").write_text(json.dumps({"random_tiny_model": family}))


def build_tiny_nemo(path):
    import sentencepiece as spm
    import torch
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf

    torch.manual_seed(1)
    path.mkdir()
    with tempfile.TemporaryDirectory(prefix="mouthkeys-tokenizer-") as temporary:
        corpus = Path(temporary) / "corpus.txt"
        corpus.write_text(
            "hello world mouth keys offline speech small random model smoke test\n" * 20
        )
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=f"{temporary}/tokenizer",
            model_type="bpe",
            vocab_size=32,
            bos_id=-1,
            eos_id=-1,
            minloglevel=2,
        )
        tokenizer = spm.SentencePieceProcessor(model_file=f"{temporary}/tokenizer.model")
        (Path(temporary) / "vocab.txt").write_text(
            "\n".join(tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())) + "\n"
        )
        config = {
            "sample_rate": 16000,
            "model_defaults": {"enc_hidden": 16, "pred_hidden": 16, "joint_hidden": 16},
            "tokenizer": {"dir": temporary, "type": "bpe"},
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "features": 80,
                "n_fft": 512,
                "dither": 0.0,
                "pad_to": 0,
            },
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 80,
                "feat_out": -1,
                "n_layers": 1,
                "d_model": 16,
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "subsampling_conv_channels": 8,
                "ff_expansion_factor": 2,
                "self_attention_model": "rel_pos",
                "n_heads": 2,
                "att_context_size": [8, 1],
                "att_context_style": "chunked_limited",
                "conv_kernel_size": 5,
                "conv_context_size": "causal",
                "dropout": 0.0,
                "dropout_pre_encoder": 0.0,
                "dropout_emb": 0.0,
                "dropout_att": 0.0,
            },
            "decoder": {
                "_target_": "nemo.collections.asr.modules.RNNTDecoder",
                "normalization_mode": None,
                "random_state_sampling": False,
                "blank_as_pad": True,
                "prednet": {"pred_hidden": 16, "pred_rnn_layers": 1, "t_max": None, "dropout": 0.0},
            },
            "joint": {
                "_target_": "nemo.collections.asr.modules.RNNTJoint",
                "log_softmax": None,
                "preserve_memory": False,
                "fuse_loss_wer": False,
                "jointnet": {
                    "encoder_hidden": 16,
                    "pred_hidden": 16,
                    "joint_hidden": 16,
                    "activation": "relu",
                    "dropout": 0.0,
                },
            },
            "decoding": {
                "strategy": "greedy_batch",
                "greedy": {"max_symbols": 1},
                "preserve_alignments": False,
                "compute_timestamps": False,
            },
            "loss": {"loss_name": "default"},
        }
        model = EncDecRNNTBPEModel(OmegaConf.create(config))
        model.save_to(str(path / "tiny.nemo"))
    (path / "mouthkeys-smoke-artifact.json").write_text(json.dumps({"random_tiny_model": "nemo"}))


def greeting_count(text):
    assert not re.search(r"<EOU>|<[a-z]{2}-[A-Z]{2}>", text), "Decoder marker leaked"
    return len(re.findall(r"\bhello\b", text.casefold()))


def check_word_times(words, samples, expect_greeting):
    assert words, "Decoder supplied no word alignment"
    assert all(0 <= word["start"] <= word["end"] <= samples / 16000 for word in words)
    if expect_greeting:
        starts = [word["start"] for word in words if greeting_count(word["word"])]
        assert starts and starts == sorted(starts), "Expected ordered greeting word timings"
        assert all(word["start"] < word["end"] for word in words)
        return len(set(starts))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--engine", choices=("transformers", "nemo"), default="transformers")
    parser.add_argument("--build-tiny", choices=("qwen", "cohere", "nemo"))
    parser.add_argument(
        "--fixture", type=Path, default=Path(__file__).parent / "resources/dictation_fixture.wav"
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-repeat", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--expect-greeting", action="store_true", help="Require a recognized hello")
    parser.add_argument("--native-timeout", type=int, help="Override the configured worker timeout")
    parser.add_argument("--revision", default="main", help="Record the local artifact revision")
    parser.add_argument(
        "--word-times", action="store_true", help="Request authoritative NeMo word alignment"
    )
    parser.add_argument("--report", type=Path, help="Write a summary without recognized text")
    args = parser.parse_args()
    # Import/build and worker execution are both offline, even if an installed
    # processor happens to attempt an auxiliary network request.
    from mouthkeys.native_worker import isolate_network

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        WANDB_MODE="disabled",
    )
    isolate_network()
    if args.build_tiny == "nemo":
        args.engine = "nemo"
    if args.word_times and args.engine != "nemo":
        raise ValueError("Native word alignment smoke requires --engine nemo")
    if args.stream and args.engine != "nemo":
        raise ValueError("Native cache-aware streaming smoke requires --engine nemo")
    if args.stream_repeat > 1 and not args.stream:
        raise ValueError("Repeated input requires --stream")
    from mouthkeys.audio import read_audio_from
    from mouthkeys.config import Config
    from mouthkeys.native_asr import speech_spans
    from mouthkeys.native_worker import NativeWorkerBackend

    audio = read_audio_from(args.fixture)
    if not speech_spans(audio):
        raise ValueError(
            "Smoke fixture must contain speech; empty VAD would not exercise inference"
        )

    with tempfile.TemporaryDirectory(prefix="mouthkeys-smoke-state-") as state:
        for key in (
            "XDG_CONFIG_HOME",
            "XDG_STATE_HOME",
            "XDG_RUNTIME_DIR",
            "XDG_CACHE_HOME",
            "HF_HOME",
        ):
            directory = Path(state) / key
            directory.mkdir(mode=0o700)
            os.environ[key] = str(directory)
        if args.build_tiny:
            build_tiny(args.build_tiny, args.artifact)
        defaults = Config()
        is_tiny = (args.artifact / "mouthkeys-smoke-artifact.json").is_file()
        config = replace(
            defaults,
            engine=args.engine,
            model=str(args.artifact.resolve()),
            model_revision=args.revision,
            device="cpu",
            compute_type="float32",
            asr_max_tokens=8 if is_tiny else defaults.asr_max_tokens,
            native_timeout_seconds=(
                args.native_timeout
                if args.native_timeout is not None
                else defaults.native_timeout_seconds
            ),
            pronunciation_matching="on" if args.word_times else "off",
        ).validate()
        load_started = time.monotonic()
        worker = NativeWorkerBackend(config)
        timings = {"load_seconds": round(time.monotonic() - load_started, 3)}
        operations = []
        word_counts = {}
        greetings = {}
        try:
            if args.stream:
                import numpy as np

                stream_audio = audio
                cuts = [len(audio) // 2]
                if args.stream_repeat > 1:
                    padded = np.concatenate([audio, np.zeros(8000, dtype=audio.dtype)])
                    stream_audio = np.tile(padded, args.stream_repeat)
                    cuts = [len(padded) * count for count in range(1, args.stream_repeat + 1)]
                stream_started = time.monotonic()
                previews = [worker.stream(stream_audio[:cut], config, "smoke") for cut in cuts]
                text = worker.stream(stream_audio, config, "smoke", final=True)
                timings["stream_seconds"] = round(time.monotonic() - stream_started, 3)
                expected = min(2, args.stream_repeat)
                if args.expect_greeting:
                    greetings["stream"] = greeting_count(text)
                    assert greetings["stream"] >= expected, "Repeated greetings were lost"
                    if args.stream_repeat > 1:
                        assert any(greeting_count(preview) for preview in previews), (
                            "No greeting was recognized before finalization"
                        )
                if args.word_times:
                    timed_greetings = check_word_times(
                        worker.words, len(stream_audio), args.expect_greeting
                    )
                    assert not args.expect_greeting or timed_greetings >= expected
                    word_counts["stream"] = len(worker.words)
                    print(json.dumps({"operation": "stream words", "words": worker.words}))
                operations.append("stream")
                worker.reset_stream()
                assert not worker.words, "Reset retained previous word alignment"
                if args.expect_greeting:
                    fresh_audio = padded if args.stream_repeat > 1 else audio
                    fresh = worker.stream(fresh_audio, config, "fresh", final=True)
                    assert greeting_count(fresh) == 1, "Fresh session did not isolate its greeting"
                    operations.append("reset")
                print(json.dumps({"operation": "stream", "text": text}))
            file_started = time.monotonic()
            segments = worker.segments(audio, config)
            timings["file_seconds"] = round(time.monotonic() - file_started, 3)
            assert all(0 <= s.start <= s.end <= len(audio) / 16000 for s in segments)
            if args.expect_greeting:
                greetings["file"] = greeting_count(" ".join(s.text for s in segments))
                assert greetings["file"], "Expected file greeting was not recognized"
            print(
                json.dumps(
                    {"operation": "segments", "segments": [vars(s) for s in segments]}, default=vars
                )
            )
            if args.word_times:
                words = [vars(word) for segment in segments for word in segment.words]
                check_word_times(words, len(audio), args.expect_greeting)
                word_counts["file"] = len(words)
            operations.append("segments")
        finally:
            worker.close()
    if args.report:
        versions = {}
        for name in ("torch", "transformers", "nemo_toolkit"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        report = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "result": "passed",
            "validation": (
                "Short English greeting regression; not a general accuracy benchmark"
                if args.expect_greeting
                else "functional only; NOT accuracy"
            ),
            "engine": args.engine,
            "artifact_name": args.artifact.name,
            "revision": args.revision,
            "device": config.device,
            "compute_type": config.compute_type,
            "native_timeout_seconds": config.native_timeout_seconds,
            "operations": operations,
            "audio_samples": len(audio),
            "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            "versions": versions,
            "network": "AF_INET/AF_INET6 denied by seccomp",
            "word_times": args.word_times,
            "stream_repeat": args.stream_repeat,
            "expected_greeting_counts": greetings,
            "word_counts": word_counts,
            **timings,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("FUNCTIONAL_SMOKE_PASSED; fixture regression is not a general accuracy benchmark")


if __name__ == "__main__":
    main()
