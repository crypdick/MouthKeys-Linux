"""Locate app-owned model artifacts and remove only the selected cached revision."""

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import huggingface_hub
from huggingface_hub import CachedRepoInfo, CachedRevisionInfo, DeleteCacheStrategy, HFCacheInfo

from .config import cache_dir, xdg

READY = ".mouthkeys-ready"


def managed_path(model: str, engine: str = "onnx") -> Path:
    """Never interpret a model ID as a filesystem path for a managed artifact."""
    digest = hashlib.sha256(model.encode()).hexdigest()[:24]
    return cache_dir() / "models" / engine / digest


def onnx_path(model: str, revision: str = "main") -> Path:
    return managed_path(model if revision == "main" else f"{model}@{revision}")


def whisper_repositories() -> dict[str, str]:
    # Whisper's alias-to-repository table is needed only for explicit model operations.
    from faster_whisper.utils import _MODELS  # noqa: PLC0415

    return dict(_MODELS)


def whisper_cache_index() -> dict[str, CachedRepoInfo]:
    root = cache_dir() / "models"
    if not root.is_dir():
        return {}
    return {
        repo.repo_id: repo
        for repo in huggingface_hub.scan_cache_dir(root).repos
        if repo.repo_type == "model"
    }


def whisper_revision(repo: CachedRepoInfo | None, revision: str) -> CachedRevisionInfo | None:
    if repo is None:
        return None
    return next(
        (item for item in repo.revisions if revision == item.commit_hash or revision in item.refs),
        None,
    )


def usable_whisper_snapshot(path: str | Path) -> bool:
    """Check local completeness, including the tokenizer's offline dependency."""
    path = Path(path)
    required = [path / name for name in ("model.bin", "config.json", "tokenizer.json")]
    vocabulary = [
        item for name in ("vocabulary.json", "vocabulary.txt") if (item := path / name).is_file()
    ]
    return bool(vocabulary) and all(
        item.is_file() and item.stat().st_size > 0 for item in [*required, *vocabulary]
    )


def _require_owned(path: Path, root: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError("Refusing model cache removal outside the selected cache")
    depth = len(path.relative_to(root).parts)
    if any(part.is_symlink() for part in (path, *path.parents[:depth])):
        raise ValueError("Refusing linked model cache")


def _revision_plan(
    cache: HFCacheInfo, repo: CachedRepoInfo, selected: CachedRevisionInfo
) -> DeleteCacheStrategy:
    # Scope the plan to this repository before asking the Hub to remove a commit.
    plan = replace(cache, repos=frozenset({repo})).delete_revisions(selected.commit_hash)
    if plan.repos:
        # The Hub normally removes the whole repo for its last snapshot. Keep
        # other files in that repo, including partial downloads.
        plan = replace(
            plan,
            repos=frozenset(),
            snapshots=frozenset({selected.snapshot_path}),
            refs=frozenset(repo.repo_path / "refs" / name for name in selected.refs),
            blobs=frozenset(
                item.blob_path
                for item in selected.files
                if not item.blob_path.is_relative_to(selected.snapshot_path)
            ),
        )
    return plan


def _delete_whisper(model: str, revision: str, root: Path) -> None:
    repo_id = whisper_repositories().get(model, model)
    parts = repo_id.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Expected model alias or Hugging Face owner/model ID")
    path = root / ("models--" + "--".join(parts))
    _require_owned(path, root)
    if not path.is_dir():
        return
    repo_path = path.resolve()
    # Refusing these before scanning also prevents reading refs through linked cache folders.
    for child in ("refs", "snapshots", "blobs"):
        _require_owned(repo_path / child, repo_path)
    cache = huggingface_hub.scan_cache_dir(root)
    cached = next((repo for repo in cache.repos if repo.repo_path == repo_path), None)
    if cached is None:
        return
    selected = whisper_revision(cached, revision)
    if selected is None:
        return
    plan = _revision_plan(cache, cached, selected)
    for directory in plan.repos | plan.snapshots:
        _require_owned(directory, repo_path)
    for file in plan.refs | plan.blobs:
        # Blob entries may be symlinks into the Hub's shared store. Only their
        # repo-local directory must be owned; the Hub unlinks the entries themselves.
        _require_owned(file.parent, repo_path)
    plan.execute()


def delete_model(model: str, engine: str = "whisper", *, revision: str = "main") -> None:
    """Delete an app-owned revision, never an entered local model path."""
    if Path(model).expanduser().exists():
        raise ValueError("Local model paths must be managed outside MouthKeys")
    app_cache = xdg("CACHE", ".cache")
    _require_owned(app_cache, app_cache)
    root = app_cache / "models"
    if engine == "whisper":
        _delete_whisper(model, revision, root)
        return
    if engine in ("nemo", "transformers"):
        from .native_asr import artifact_path  # noqa: PLC0415 - avoid model registry import cycle

        path = artifact_path(model, engine, revision)
    elif engine == "onnx":
        path = onnx_path(model, revision)
    else:
        raise ValueError("Unknown engine")
    _require_owned(path, root)
    if path.is_dir():
        shutil.rmtree(path)
