"""Which files each image needs, and how they reach the ComfyUI server.

RunPod: URL files are fetched by the pod (a URL wins over a local copy). Files
that are only on this computer go to your storage once and are fetched from
there, or, without storage, are uploaded to the pod (SHA-256 verified, skipped
when the pod already holds them).
Local ComfyUI: files are found by name in ComfyUI's own models folders. When
that ComfyUI runs on this computer, missing files are first downloaded or
linked into its folders (see `local.py`); a ComfyUI elsewhere must already have
them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path, PurePosixPath
from typing import Callable

from jena_image_render import env, models
from jena_image_render.comfy import ComfyClient, ComfyError
from jena_image_render.config import Render
from jena_image_render.graph import Names
from jena_image_render.models import FileSpec, Registry

LOGGER = logging.getLogger(__name__)
FETCH_TIMEOUT = 60 * 60
POLL_SECONDS = 10.0


class AssetError(ValueError):
    """A needed file is missing or does not match."""


def needed(renders: list[Render], registry: Registry) -> list[FileSpec]:
    """Every file the renders load, once each."""
    files: dict[str, FileSpec] = {}
    for render in renders:
        model = registry.model(render.model)
        specs = [*model.files.values(), *(registry.lora(n).file for n, _ in render.loras)]
        for spec in specs:
            files.setdefault(spec.dest, spec)
    return list(files.values())


Plan = list[tuple[str, FileSpec]]


def plan(files: list[FileSpec], target: str, storage: bool = False) -> Plan:
    """How each file reaches the server: fetch, storage, upload, local or missing.

    With no target the RunPod plan is shown: it is the one that needs files on
    this computer.
    """
    steps = []
    for spec in files:
        if target == "local":
            how = "local"  # ComfyUI must list it by name
        elif spec.url:
            how = "fetch"
        elif not Path(spec.local).is_file():
            how = "missing"
        else:
            how = "storage" if storage else "upload"
        steps.append((how, spec))
    return steps


def _names(render: Render, registry: Registry, name_of: Callable[[FileSpec], str]) -> Names:
    files = registry.model(render.model).files
    base = files.get("checkpoint") or files["diffusion"]
    return Names(
        base=name_of(base),
        clip=name_of(files["text_encoder"]) if "text_encoder" in files else "",
        vae=name_of(files["vae"]) if "vae" in files else "",
        loras=tuple(name_of(registry.lora(n).file) for n, _ in render.loras),
    )


# ── RunPod ───────────────────────────────────────────────────────────
class Hashes:
    """SHA-256 per local file, cached by (size, mtime): big files are read once."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or env.STATE_DIR / "sha256.json"
        self.data: dict[str, list[object]] = (
            json.loads(self.path.read_text("utf-8")) if self.path.is_file() else {}
        )

    def remember(self, file: Path, sha256: str) -> None:
        stat = file.stat()
        self.data[str(file.resolve())] = [stat.st_size, stat.st_mtime, sha256]
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), "utf-8")

    def __call__(self, file: Path) -> str:
        stat = file.stat()
        key = str(file.resolve())
        cached = self.data.get(key)
        if cached and cached[:2] == [stat.st_size, stat.st_mtime]:
            return str(cached[2])
        LOGGER.info("Hashing %s (%.1f GB, once per file)", file.name, stat.st_size / 1e9)
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        self.data[key] = [stat.st_size, stat.st_mtime, digest.hexdigest()]
        self._save()
        return digest.hexdigest()


def prepare_uploads(files: list[FileSpec], hashes: Hashes) -> list[FileSpec]:
    """Local-only files with their real size and SHA-256; URL files unchanged."""
    missing = models.missing_local(files)
    if missing:
        raise AssetError("local files not found:\n  " + "\n  ".join(missing))
    ready = []
    for spec in files:
        if spec.url:
            ready.append(spec)
            continue
        path = Path(spec.local)
        sha = hashes(path)
        if spec.sha256 and spec.sha256 != sha:
            raise AssetError(f"{path}: sha256 differs from the expected one")
        ready.append(FileSpec(spec.dest, local=spec.local, sha256=sha, size=path.stat().st_size))
    return ready


def deliver(
    client: ComfyClient,
    files: list[FileSpec],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Start pod downloads, upload local files meanwhile, then wait for downloads."""
    downloads = [f for f in files if f.url]
    uploads = [f for f in files if f.local and not f.url]
    if downloads:
        client.fetch(downloads)
    for item in sorted(uploads, key=lambda f: f.size):
        LOGGER.info("Uploading %s (%.2f GB)", item.dest, item.size / 1e9)
        client.upload(item)
    if downloads:
        wait_fetch(client, {f.dest for f in downloads}, sleep)


def wait_fetch(
    client: ComfyClient, dests: set[str], sleep: Callable[[float], None] = time.sleep
) -> None:
    deadline = time.monotonic() + FETCH_TIMEOUT
    while time.monotonic() < deadline:
        state = {k: v for k, v in client.fetch_status().items() if k in dests}
        failed = {k: v.get("error") for k, v in state.items() if v["status"] == "error"}
        if failed:
            raise ComfyError(f"download failed: {failed}")
        done = sum(v["status"] == "done" for v in state.values())
        if done == len(dests):
            return
        got = sum(int(v.get("bytes", 0)) for v in state.values())
        total = sum(int(v.get("total", 0)) for v in state.values())
        percent = f"{100 * got / total:.0f}%" if total else f"{got / 1e9:.1f} GB"
        LOGGER.info("Downloading models on the pod: %s/%s files, %s", done, len(dests), percent)
        sleep(POLL_SECONDS)
    raise ComfyError("model download timed out")


def pod_names(render: Render, registry: Registry) -> Names:
    """On the pod every file sits exactly at its dest name."""
    return _names(render, registry, lambda spec: spec.name)


# ── local ComfyUI ────────────────────────────────────────────────────
class LocalNames:
    """Match files to what the ComfyUI lists, by file name.

    `filled` says the files were already downloaded or linked into a ComfyUI on
    this computer, which changes the hint when one is still missing.
    """

    def __init__(self, client: ComfyClient, filled: bool = False) -> None:
        self.client = client
        self.filled = filled
        self._options: dict[str, list[str]] = {}

    def options(self, folder: str) -> list[str]:
        if folder not in self._options:
            self._options[folder] = self.client.options(folder)
        return self._options[folder]

    def forget(self) -> None:
        """Ask ComfyUI again next time (after files were added)."""
        self._options.clear()

    def lookup(self, spec: FileSpec) -> str:
        """The name ComfyUI lists the file by, or "" when it does not."""
        folder = spec.folder
        options = self.options(folder)
        if spec.name in options:
            return spec.name
        leaf = PurePosixPath(spec.name).name
        hits = [o for o in options if PurePosixPath(o).name == leaf]
        if len(hits) > 1:
            raise AssetError(f"{leaf} is ambiguous in ComfyUI models/{folder}: {hits}")
        return hits[0] if hits else ""

    def find(self, spec: FileSpec) -> str:
        if name := self.lookup(spec):
            return name
        folder = spec.folder
        leaf = PurePosixPath(spec.name).name
        if spec.url and self.filled:
            hint = "--no-download is on; run without it, or `jenair pull JOB.json`"
        elif spec.url:
            hint = "`jenair pull JOB.json` downloads it"
        else:
            hint = "put the file (or a link to it) there"
        message = f"{leaf} is not in your ComfyUI models/{folder}/ — {hint}"
        if self.filled:
            raise AssetError(f"{message}, then refresh ComfyUI's model list")
        raise AssetError(
            f"{message}. A ComfyUI on another computer cannot be filled from here: "
            "put the file in that computer's ComfyUI models/ folder. For a ComfyUI "
            "on this computer that does not report its folders, point "
            "JENA_MODELS_DIR at ComfyUI's models/ folder, or add your models folder "
            "to ComfyUI's extra_model_paths.yaml, then restart ComfyUI or refresh "
            "its model list"
        )

    def names(self, render: Render, registry: Registry) -> Names:
        return _names(render, registry, self.find)
