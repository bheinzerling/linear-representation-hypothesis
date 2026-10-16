from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.utils import ensure_dir


CACHE_VERSION = 1
TASK_NAME = "prompt_sequence_activation_cache"
DEFAULT_CACHE_ROOT_NAME = "source_value_probe_canonical_prompt_sequences"


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".safetensors", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        save_file(tensors, str(tmp_path))
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _sanitize_attn_implementation(attn_implementation: str | None) -> str:
    return str(attn_implementation) if attn_implementation is not None else "none"


def build_prompt_sequence_cache_key(
    *,
    prompt: str,
    input_ids: list[int],
    model_id: str,
    model_revision: str,
    layer_idx: int,
    attn_implementation: str | None,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "prompt": str(prompt),
        "input_ids": [int(value) for value in input_ids],
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "layer_idx": int(layer_idx),
        "attn_implementation": _sanitize_attn_implementation(attn_implementation),
    }
    return _stable_hash(payload), payload


@dataclass(frozen=True)
class PromptSequenceShardSpec:
    model_alias: str
    model_id: str
    model_revision: str
    layer_idx: int
    attn_implementation: str | None
    family: str
    cache_root: Path

    @property
    def hash_payload(self) -> dict[str, Any]:
        return {
            "task_name": TASK_NAME,
            "cache_version": CACHE_VERSION,
            "model_alias": str(self.model_alias),
            "model_id": str(self.model_id),
            "model_revision": str(self.model_revision),
            "layer_idx": int(self.layer_idx),
            "attn_implementation": _sanitize_attn_implementation(self.attn_implementation),
            "family": str(self.family),
        }

    @property
    def shard_hash(self) -> str:
        return _stable_hash(self.hash_payload)[:12]

    @property
    def shard_id(self) -> str:
        return f"{TASK_NAME}__{self.shard_hash}__r0"

    @property
    def shard_dir(self) -> Path:
        return Path(self.cache_root) / self.shard_id


@contextmanager
def _exclusive_lock(path: Path):
    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PromptSequenceActivationCache:
    def __init__(self, spec: PromptSequenceShardSpec):
        self.spec = spec
        self.shard_dir = spec.shard_dir
        self.chunks_dir = self.shard_dir / "chunks"
        self.manifest_path = self.shard_dir / "manifest.json"
        self.lock_path = self.shard_dir / ".lock"
        self._manifest_cache: dict[str, Any] | None = None
        self._chunk_cache: dict[str, dict[str, np.ndarray]] = {}

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "task_name": TASK_NAME,
            "cache_version": CACHE_VERSION,
            "shard_id": self.spec.shard_id,
            "hash_payload": self.spec.hash_payload,
            "entries": {},
            "chunk_files": [],
            "next_chunk_index": 0,
        }

    def _load_manifest(self, *, force_reload: bool = False) -> dict[str, Any]:
        if self._manifest_cache is not None and not force_reload:
            return self._manifest_cache
        if not self.manifest_path.exists():
            manifest = self._empty_manifest()
        else:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._manifest_cache = manifest
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        _atomic_write_text(self.manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self._manifest_cache = manifest

    def prepare_entries(self, *, prompts: list[str], tokenizer, layer_idx: int) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for prompt in prompts:
            input_ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
            cache_key, key_payload = build_prompt_sequence_cache_key(
                prompt=str(prompt),
                input_ids=[int(value) for value in input_ids],
                model_id=self.spec.model_id,
                model_revision=self.spec.model_revision,
                layer_idx=layer_idx,
                attn_implementation=self.spec.attn_implementation,
            )
            entries.append(
                {
                    "cache_key": cache_key,
                    "prompt": str(prompt),
                    "input_ids": [int(value) for value in input_ids],
                    "key_payload": key_payload,
                }
            )
        return entries

    def get_missing_entries(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        manifest = self._load_manifest()
        known_entries = manifest["entries"]
        return [entry for entry in entries if entry["cache_key"] not in known_entries]

    def store_missing_sequences(
        self,
        *,
        entries: list[dict[str, Any]],
        sequences: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        if not entries:
            return []
        if len(entries) != len(sequences):
            raise ValueError(f"entries/sequences length mismatch: {len(entries)} != {len(sequences)}")

        ensure_dir(self.chunks_dir)
        stored_entries: list[dict[str, Any]] = []
        with _exclusive_lock(self.lock_path):
            manifest = self._load_manifest(force_reload=True)
            missing_entries = [entry for entry in entries if entry["cache_key"] not in manifest["entries"]]
            if not missing_entries:
                return []

            tensor_map: dict[str, torch.Tensor] = {}
            entry_lookup = {str(entry["cache_key"]): entry for entry in entries}
            sequence_lookup = {
                str(entry["cache_key"]): np.asarray(sequence, dtype=np.float32)
                for entry, sequence in zip(entries, sequences, strict=False)
            }
            chunk_name = f"chunk_{int(manifest['next_chunk_index']):06d}.safetensors"
            chunk_path = self.chunks_dir / chunk_name
            for entry in missing_entries:
                cache_key = str(entry["cache_key"])
                sequence = sequence_lookup[cache_key]
                if sequence.ndim != 2:
                    raise ValueError(f"Expected cached sequence tensor to be 2D, got shape {sequence.shape!r}")
                tensor_map[cache_key] = torch.from_numpy(np.ascontiguousarray(sequence))

            _atomic_save_safetensors(chunk_path, tensor_map)
            manifest["chunk_files"].append(chunk_name)
            manifest["next_chunk_index"] = int(manifest["next_chunk_index"]) + 1
            for cache_key, tensor in tensor_map.items():
                entry = entry_lookup[cache_key]
                manifest["entries"][cache_key] = {
                    "chunk_file": chunk_name,
                    "tensor_name": cache_key,
                    "prompt": str(entry["prompt"]),
                    "input_ids": [int(value) for value in entry["input_ids"]],
                    "key_payload": dict(entry["key_payload"]),
                    "prompt_hash": _stable_hash({"prompt": str(entry["prompt"])}),
                    "seq_len": int(tensor.shape[0]),
                    "model_dim": int(tensor.shape[1]),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                }
                stored_entries.append(manifest["entries"][cache_key])
            self._write_manifest(manifest)
        return stored_entries

    def _load_chunk(self, chunk_name: str) -> dict[str, np.ndarray]:
        if chunk_name in self._chunk_cache:
            return self._chunk_cache[chunk_name]
        chunk_path = self.chunks_dir / chunk_name
        tensors: dict[str, np.ndarray] = {}
        with safe_open(str(chunk_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key).detach().cpu().numpy().astype(np.float32, copy=False)
        self._chunk_cache[chunk_name] = tensors
        return tensors

    def load_sequence(self, cache_key: str) -> np.ndarray:
        manifest = self._load_manifest()
        entry = manifest["entries"].get(cache_key)
        if entry is None:
            raise KeyError(f"Unknown cache key {cache_key}")
        chunk_tensors = self._load_chunk(str(entry["chunk_file"]))
        return np.asarray(chunk_tensors[str(entry["tensor_name"])], dtype=np.float32)


def default_prompt_sequence_cache_root() -> Path:
    return Path("out") / "activation_caches" / DEFAULT_CACHE_ROOT_NAME


def build_prompt_sequence_cache_spec(
    *,
    cache_root: str | Path,
    model_alias: str,
    model_id: str,
    model_revision: str,
    layer_idx: int,
    attn_implementation: str | None,
    family: str,
) -> PromptSequenceShardSpec:
    return PromptSequenceShardSpec(
        model_alias=str(model_alias),
        model_id=str(model_id),
        model_revision=str(model_revision),
        layer_idx=int(layer_idx),
        attn_implementation=attn_implementation,
        family=str(family),
        cache_root=Path(cache_root),
    )
