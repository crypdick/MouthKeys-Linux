import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from .smoke_native import build_tiny, tiny_fixture


def test_nemo_builder_adds_private_tokenizer_path_without_changing_fixture(monkeypatch, tmp_path):
    config_snapshot = tiny_fixture("nemo")
    tokenizer_dirs = []

    def train(*, model_prefix, **options):
        corpus = Path(options["input"])
        assert corpus.read_text().startswith("hello world mouth keys")
        tokenizer_dirs.append(corpus.parent)
        Path(f"{model_prefix}.model").touch()

    def tokenizer(*, model_file):
        assert Path(model_file).exists()
        return SimpleNamespace(get_piece_size=lambda: 2, id_to_piece=("hello", "world").__getitem__)

    class Model:
        def __init__(self, config):
            directory = tokenizer_dirs[0]
            assert config == config_snapshot | {"tokenizer": {"dir": str(directory), "type": "bpe"}}
            assert (directory / "vocab.txt").read_text() == "hello\nworld\n"

        def save_to(self, path):
            Path(path).write_bytes(b"synthetic model")

    def fake_module(name, **members):
        module = ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)

    seeds = []
    fake_module("torch", manual_seed=seeds.append)
    fake_module(
        "sentencepiece",
        SentencePieceTrainer=SimpleNamespace(train=train),
        SentencePieceProcessor=tokenizer,
    )
    fake_module("nemo.collections.asr.models", EncDecRNNTBPEModel=Model)
    fake_module("omegaconf", OmegaConf=SimpleNamespace(create=lambda value: value))
    destination = tmp_path / "nemo"
    build_tiny("nemo", destination)
    assert seeds == [1]
    assert (destination / "tiny.nemo").read_bytes() == b"synthetic model"
    marker = destination / "mouthkeys-smoke-artifact.json"
    assert json.loads(marker.read_bytes()) == {"random_tiny_model": "nemo"}
    assert not tokenizer_dirs[0].exists()
    assert tiny_fixture("nemo") == config_snapshot
