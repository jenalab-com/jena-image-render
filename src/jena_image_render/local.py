"""A ComfyUI on this computer: find its model folders and fill them before a run.

ComfyUI reports its folders at GET /internal/folder_paths. When its address is
a loopback one (127.0.0.1, localhost, ::1) and at least one reported folder
exists here, it is the same computer, and:

- model and LoRA names are also looked up in those folders (after ./models),
- before rendering, a file ComfyUI does not list is downloaded into its folder
  (URL files) or linked there (files only in ./models): symbolic link, else
  hard link, else a copy.

A ComfyUI on another computer, or one without that API, gets nothing written:
its files are only looked up by name.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from jena_image_render.assets import Hashes, LocalNames
from jena_image_render.comfy import ComfyClient
from jena_image_render.folder import ComfyFolders
from jena_image_render.models import FileSpec
from jena_image_render.pull import pull_one

LOGGER = logging.getLogger(__name__)
QUICK_TIMEOUT = 3.0  # --dry-run and pull: a ComfyUI that is off answers fast

# (how, file): have, download, link or missing
Steps = list[tuple[str, FileSpec]]


def is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def discover(client: ComfyClient, url: str, timeout: float = 15) -> ComfyFolders | None:
    """The folders of a ComfyUI on this computer, or None (then nothing is filled)."""
    if not is_loopback(url):
        LOGGER.info(
            "ComfyUI at %s is on another computer: its files are looked up by name, "
            "nothing is downloaded or linked into it",
            url,
        )
        return None
    paths = client.folder_paths(timeout)
    if paths is None:
        LOGGER.info(
            "This ComfyUI does not report its model folders (/internal/folder_paths): "
            "its files are looked up by name only"
        )
        return None
    folders = ComfyFolders(paths)
    if not folders.exists():
        LOGGER.info(
            "ComfyUI's model folders are not on this computer: its files are looked "
            "up by name only"
        )
        return None
    try:
        where = folders.target("checkpoints").parent
    except ValueError:
        where = "its folders"
    LOGGER.info("ComfyUI on this computer: using its model folders (%s)", where)
    return folders


def plan(
    files: list[FileSpec],
    names: LocalNames,
    folders: ComfyFolders | None,
    download: bool = True,
) -> tuple[Steps, dict[str, Path]]:
    """What each file needs, and where downloads and links go."""
    steps: Steps = []
    places: dict[str, Path] = {}
    for spec in files:
        if names.lookup(spec):
            how = "have"
        elif folders is None:
            how = "missing"
        elif spec.local and Path(spec.local).is_file():
            how = "link"  # a copy here beats downloading it again
        elif spec.url and download:
            how = "download"
        else:
            how = "missing"
        if how in ("link", "download"):
            assert folders is not None
            places[spec.dest] = folders.path(spec.dest)
        steps.append((how, spec))
    return steps, places


def link(
    source: Path,
    target: Path,
    symlink: Callable[[Path, Path], None] = os.symlink,
    hardlink: Callable[[Path, Path], None] = os.link,
    copy: Callable[[Path, Path], Any] = shutil.copy2,
) -> str:
    """Make `source` appear at `target`: a symbolic link, else a hard link (e.g.
    Windows without the link right), else a copy. Returns what was done."""
    if target.exists() or target.is_symlink():
        LOGGER.info("Already there: %s", target)
        return "there"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    try:
        symlink(source, target)
        how = "linked"
    except OSError:
        try:
            hardlink(source, target)
            how = "hard-linked"
        except OSError:
            partial = target.with_name(target.name + ".part")
            copy(source, partial)
            partial.replace(target)
            how = "copied"
    LOGGER.info("%s %s → %s", how.capitalize(), source, target)
    return how


def _summary(downloads: list[FileSpec], places: dict[str, Path]) -> str:
    known = [spec.size for spec in downloads if spec.size]
    size = f"{sum(known) / 1e9:.1f} GB"
    if len(known) < len(downloads):
        size += f" (size known for {len(known)} of {len(downloads)})"
    # The role folder: the place minus the path inside it.
    folders = sorted(
        {str(places[s.dest].parents[s.name.count("/")]) for s in downloads}
    )
    return (
        f"Downloading {len(downloads)} file(s), {size}, into ComfyUI's folders: "
        + ", ".join(folders)
    )


def fill(
    steps: Steps,
    places: dict[str, Path],
    names: LocalNames,
    hashes: Hashes,
    hf_token: str = "",
    opener: Callable[..., Any] | None = None,
) -> None:
    """Link and download what the plan says, then check ComfyUI lists it all."""
    for how, spec in steps:
        if how == "link":
            link(Path(spec.local), places[spec.dest])
    downloads = [spec for how, spec in steps if how == "download"]
    if downloads:
        LOGGER.info("%s", _summary(downloads, places))
    for spec in downloads:
        pull_one(spec, places[spec.dest], hashes, hf_token, opener)
    names.forget()
    for _, spec in steps:
        names.find(spec)
