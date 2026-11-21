"""Chunked model/LoRA uploads from the user's computer: in-order appends, verified on the last chunk.

Civitai blocks datacenter IPs, so every Civitai-sourced file travels from the user's computer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .fetcher import resolve_dest


class UploadError(Exception):
    """The chunk cannot be accepted; the client should restart or give up."""


UPLOAD_FOLDERS = (
    "loras",
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "vae",
    "upscale_models",
    "ultralytics",
)


def _target(models_dir: Path, dest: str) -> Path:
    folder = Path(dest).parts[:1]
    if not folder or folder[0] not in UPLOAD_FOLDERS:
        raise ValueError(f"uploads are limited to {', '.join(UPLOAD_FOLDERS)}: {dest!r}")
    target = resolve_dest(models_dir, dest)
    if not target.is_relative_to((models_dir / folder[0]).resolve()):
        raise ValueError(f"upload escapes {folder[0]}/: {dest!r}")
    return target


def _marker(target: Path) -> Path:
    return target.with_name(target.name + ".sha256")


def is_verified(models_dir: Path, dest: str, sha256: str) -> bool:
    target = _target(models_dir, dest)
    marker = _marker(target)
    return target.is_file() and marker.is_file() and marker.read_text().strip() == sha256


def append_chunk(
    models_dir: Path, dest: str, sha256: str, offset: int, total: int, data: bytes
) -> str:
    """Append one chunk; return "partial", or "done" once the whole file verified."""
    target = _target(models_dir, dest)
    partial = target.with_name(target.name + ".part")
    if offset < 0 or offset + len(data) > total:
        raise UploadError(f"chunk exceeds total size {total}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if offset == 0:
        partial.unlink(missing_ok=True)
        _marker(target).unlink(missing_ok=True)
    current = partial.stat().st_size if partial.exists() else 0
    if offset != current:
        raise UploadError(f"expected offset {current}, got {offset}")
    with partial.open("ab") as stream:
        stream.write(data)
    if offset + len(data) < total:
        return "partial"
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise UploadError(f"sha256 mismatch for {dest}")
    partial.replace(target)
    _marker(target).write_text(sha256)
    return "done"
