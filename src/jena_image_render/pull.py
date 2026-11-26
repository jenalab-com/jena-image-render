"""Download URL files for a local ComfyUI: into its own folders when it runs on
this computer and reports them, else into the models folder.

Each file goes to its models-folder path (catalog LoRAs under loras/<engine>/),
through a .part file that a later run resumes with an HTTP Range request. The
SHA-256 is checked when it is known; a file already there with the right hash
is skipped. HF_TOKEN is sent to huggingface.co only.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from jena_image_render.assets import Hashes
from jena_image_render.folder import ComfyFolders, ModelsFolder
from jena_image_render.models import FileSpec

LOGGER = logging.getLogger(__name__)
CHUNK = 8 << 20


class PullError(RuntimeError):
    """A download failed or does not match its SHA-256."""


def headers(url: str, hf_token: str) -> dict[str, str]:
    host = urlparse(url).hostname or ""
    if hf_token and (host == "huggingface.co" or host.endswith(".huggingface.co")):
        return {"Authorization": f"Bearer {hf_token}"}
    return {}


def _hash_file(path: Path) -> Any:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest


def _present(target: Path, spec: FileSpec, hashes: Hashes) -> bool:
    if not target.is_file():
        return False
    if spec.sha256:
        return hashes(target) == spec.sha256
    return not spec.size or target.stat().st_size == spec.size


def pull_one(
    spec: FileSpec,
    target: Path,
    hashes: Hashes,
    hf_token: str = "",
    opener: Callable[..., Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> bool:
    """Download one file; False when it was already there. `progress` hears
    (bytes so far, total or 0) after each chunk."""
    if _present(target, spec, hashes):
        LOGGER.info("Already there: %s", target)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = headers(spec.url, hf_token)
    if offset:
        request["Range"] = f"bytes={offset}-"
    LOGGER.info("Downloading %s → %s", spec.url, target)
    opener = opener or requests.get
    with opener(spec.url, headers=request, stream=True, timeout=(30, 300)) as response:
        if offset and response.status_code == 416:  # the .part is already complete
            digest = _hash_file(partial)
        else:
            response.raise_for_status()
            # 206: the server continues the .part; 200: it sends the whole file.
            resumed = bool(offset) and response.status_code == 206
            digest = _hash_file(partial) if resumed else hashlib.sha256()
            done = offset if resumed else 0
            with partial.open("ab" if resumed else "wb") as stream:
                for chunk in response.iter_content(CHUNK):
                    stream.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, spec.size)
    sha = digest.hexdigest()
    if spec.sha256 and sha != spec.sha256:
        partial.unlink(missing_ok=True)
        raise PullError(f"{target.name}: sha256 mismatch, download removed — try again")
    partial.replace(target)
    hashes.remember(target, sha)
    return True


def pull(
    files: list[FileSpec],
    folder: ModelsFolder | ComfyFolders,
    hashes: Hashes,
    hf_token: str = "",
    opener: Callable[..., Any] | None = None,
) -> int:
    """Download every URL file; returns how many were downloaded."""
    count = 0
    for spec in files:
        if not spec.url:
            LOGGER.info("Local file, nothing to download: %s", spec.local)
            continue
        count += pull_one(spec, folder.path(spec.dest), hashes, hf_token, opener)
    return count
