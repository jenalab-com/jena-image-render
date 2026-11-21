"""The models folder: weight files laid out like ComfyUI's models/.

    models/checkpoints/       my-anime.safetensors
    models/diffusion_models/  <engine>/my-finetune.safetensors
    models/text_encoders/     ...
    models/vae/               ...
    models/loras/             my-style.safetensors   (or loras/<engine>/...)

Files are found by name or stem, in any sub-folder. Paths are returned relative
to the models folder (posix), which is also where the pod stores them.

A ComfyUI on this computer reports its own folders (`ComfyFolders`); names are
looked up there too, after this folder.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

DEFAULT_DIR = "models"
WEIGHTS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
# The role folders this tool loads files from.
ROLES = ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras")


class AmbiguousFile(ValueError):
    """More than one file answers to the same name."""


class ModelsFolder:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(os.environ.get("JENA_MODELS_DIR") or DEFAULT_DIR)
        self._index: dict[str, list[str]] = {}

    def files(self, folder: str) -> list[str]:
        """Weight files under one role folder, as paths relative to the root."""
        if folder not in self._index:
            base = self.root / folder
            found = []
            if base.is_dir():
                for path in sorted(base.rglob("*")):
                    if weight(path.name) and path.is_file():
                        found.append(path.relative_to(self.root).as_posix())
            self._index[folder] = found
        return self._index[folder]

    def find(self, folder: str, query: str) -> str:
        """The one file `query` names (path, name or stem), or "" if none."""
        return match(self.files(folder), query, str(self.root / folder))

    def path(self, rel: str) -> Path:
        return self.root / rel

    def located(self, rel: str) -> Path:
        return self.root / rel


def match(files: list[str], query: str, where: str) -> str:
    """The one of `files` (role-folder/... paths) that `query` names, or "".

    A path inside the role folder wins over a file name or stem; two hits of the
    same kind are an error.
    """
    query = query.replace("\\", "/").strip("/")

    def inner(rel: str) -> PurePosixPath:
        return PurePosixPath(rel.split("/", 1)[1])

    tests = (
        lambda p: str(p) == query or str(p.with_suffix("")) == query,
        lambda p: p.name == query or p.stem == query,
    )
    for test in tests:
        hits = [rel for rel in files if "/" in rel and test(inner(rel))]
        if len(hits) > 1:
            raise AmbiguousFile(
                f"{query!r} matches several files in {where}: "
                + ", ".join(hits)
                + " (use the sub-folder path or rename one)"
            )
        if hits:
            return hits[0]
    return ""


def weight(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in WEIGHTS


class ComfyFolders:
    """The model folders a ComfyUI on this computer reports, per role.

    ComfyUI lists several paths per role: its own folder, legacy names (`unet`,
    `clip`), extra_model_paths.yaml folders and output folders. Files are looked
    up in every listed path that exists (the first path wins for one relative
    path, as in ComfyUI); new files go to `place()`.
    """

    def __init__(self, paths: dict[str, list[str]]) -> None:
        self.paths = {role: [Path(p) for p in paths.get(role, [])] for role in ROLES}
        self._index: dict[str, list[str]] = {}
        self._where: dict[str, Path] = {}

    def exists(self) -> bool:
        """True when at least one reported path is on this computer."""
        return any(p.is_dir() for paths in self.paths.values() for p in paths)

    def files(self, folder: str) -> list[str]:
        """Weight files under one role, as <role>/<path inside its root>."""
        if folder not in self._index:
            found: list[str] = []
            for base in self.paths.get(folder, []):
                if not base.is_dir():
                    continue
                for top, dirs, names in os.walk(base, followlinks=True):
                    dirs.sort()
                    for name in sorted(names):
                        path = Path(top) / name
                        if not weight(name) or not path.is_file():
                            continue
                        rel = f"{folder}/{path.relative_to(base).as_posix()}"
                        if rel not in self._where:
                            self._where[rel] = path
                            found.append(rel)
            self._index[folder] = found
        return self._index[folder]

    def find(self, folder: str, query: str) -> str:
        return match(self.files(folder), query, f"ComfyUI's {folder} folders")

    def located(self, rel: str) -> Path:
        """Where a file `find()` returned really is."""
        self.files(rel.split("/", 1)[0])
        return self._where[rel]

    def target(self, folder: str) -> Path:
        """Where new files of a role go: the path named like the role (not a
        legacy alias such as `unet`), else the first that exists, else the
        first."""
        paths = self.paths.get(folder) or []
        if not paths:
            raise ValueError(f"this ComfyUI reports no folder for {folder}")
        for path in paths:
            if path.name == folder:
                return path
        return next((p for p in paths if p.is_dir()), paths[0])

    def path(self, rel: str) -> Path:
        """Where a file with this models-folder path goes in ComfyUI."""
        folder, _, inner = rel.partition("/")
        return self.target(folder) / inner

    def forget(self) -> None:
        self._index.clear()
        self._where.clear()
