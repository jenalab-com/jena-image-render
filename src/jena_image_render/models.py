"""Model and LoRA names → the weight files they load.

A `model` name in a job is looked up in this order:

1. the built-in catalog (`catalog.json`): official releases and example LoRAs,
2. the models folder (`JENA_MODELS_DIR`, default ./models): a file in
   `checkpoints/` (an SDXL-type checkpoint) or `diffusion_models/<engine>/`
   (a fine-tune of that engine; its other files come from the engine's
   built-in model), by name or stem,
3. the model folders of a ComfyUI on this computer (local target), with the
   same folder rules; the models folder wins when both have the name,
4. your storage (`JENA_STORAGE`), listed with the same folder rules.

A LoRA name goes the same way through the catalog, `loras/` (or
`loras/<engine>/`) and the storage. A job may also write a file's address
(hf://owner/repo[@revision]/path or https://...) instead of a name, or a model
as an object: {"engine": ..., "diffusion": ..., "name": ...}.

The tool ships no weights. A file is a local path (the pod receives it by upload
or from your storage), or a URL (the pod fetches it). Local ComfyUI finds files
by name in its own models folders.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import quote, urlparse

from jena_image_render.engines import ENGINES, FOLDERS, engine
from jena_image_render.folder import AmbiguousFile, ComfyFolders, ModelsFolder, match

# The built-in model whose other files a fine-tune of each engine borrows.
BASE_MODELS = {
    "zimage": "zimage",
    "krea2": "krea2",
    "qwen": "qwen",
    "klein": "klein-4b",
}
URL_DIR = "_url"  # <role folder>/_url/<sha1 of the address>/<file name>
EXAMPLE_URL = "hf://owner/repo/file.safetensors"

# Storage lookup: a role folder → the weight files stored under it.
Remote = Callable[[str], list["FileSpec"]]


class ModelsError(ValueError):
    """A name or address cannot be resolved to weight files."""


@dataclass(frozen=True)
class FileSpec:
    """One weight file and the ComfyUI name it is loaded by."""

    dest: str  # relative to ComfyUI models/, e.g. "vae/ae.safetensors"
    url: str = ""  # fetched by the pod
    local: str = ""  # a copy on this computer (uploaded when there is no url)
    sha256: str = ""  # optional for URLs; computed for local files
    size: int = 0

    @property
    def folder(self) -> str:
        return self.dest.split("/", 1)[0]

    @property
    def name(self) -> str:
        """Name inside the folder, as ComfyUI loaders list it."""
        return self.dest.split("/", 1)[1]

    @property
    def source(self) -> str:
        return self.url or self.local


@dataclass(frozen=True)
class Model:
    alias: str
    engine: str
    files: dict[str, FileSpec]  # role → file


@dataclass(frozen=True)
class Lora:
    alias: str
    engine: str  # "" = no engine known (folder or address): any engine
    file: FileSpec
    trigger: str = ""


def is_address(value: str) -> bool:
    return value.startswith(("hf://", "http://", "https://"))


@dataclass(frozen=True)
class Registry:
    models: dict[str, Model]
    loras: dict[str, Lora]
    folder: ModelsFolder | None = field(default=None, compare=False)
    remote: Remote | None = field(default=None, compare=False)
    # Models a job wrote as an address or object, by the name renders use.
    jobs: dict[str, Model] = field(default_factory=dict, compare=False)
    # The folders of a ComfyUI on this computer (local target only).
    comfy: ComfyFolders | None = field(default=None, compare=False)

    def _places(self) -> list[ModelsFolder | ComfyFolders]:
        return [f for f in (self.folder, self.comfy) if f is not None]

    def _in(self, place: ModelsFolder | ComfyFolders, top: str, name: str) -> str:
        try:
            return place.find(top, name)
        except AmbiguousFile as exc:
            raise ModelsError(str(exc)) from exc

    def _hits(
        self, tops: tuple[str, ...], name: str, stored: bool = True
    ) -> dict[str, FileSpec]:
        """Files `name` matches per role folder: the models folder, else the local
        ComfyUI's folders, else storage."""
        hits = {}
        for place in self._places():
            for top in tops:
                if rel := self._in(place, top, name):
                    hits[top] = FileSpec(rel, local=str(place.located(rel)))
            if hits:
                return hits
        if self.remote is None or not stored:
            return hits
        for top in tops:
            stored = {spec.dest: spec for spec in self.remote(top)}
            try:
                rel = match(list(stored), name, f"your storage's {top}/")
            except AmbiguousFile as exc:
                raise ModelsError(str(exc)) from exc
            if rel:
                hits[top] = stored[rel]
        return hits

    def _unknown(self, kind: str, name: str, known: dict[str, Any], *where: str) -> str:
        names = ", ".join(sorted(known)) or "none"
        message = f"{kind} {name!r} is not a known alias (known: {names})"
        if self.folder is None:
            return message
        root = self.folder.root.as_posix()
        places = " or ".join(f"{root}/{w}/" for w in where)
        stored = " or in your storage" if self.remote is not None else ""
        if self.comfy is not None:
            stored = " or in your ComfyUI's model folders" + stored
        if kind == "model":
            example = f'"model": "{EXAMPLE_URL}"'
        else:
            example = f'"loras": ["{EXAMPLE_URL}"]'
        return (
            f"{message}, a file in {places}{stored} — put the file there, or write "
            f"its address in the job: {example}"
        )

    def model(self, value: Any) -> Model:
        """A model by alias or file name, by address, or as an object."""
        if isinstance(value, dict):
            return self._object_model(value)
        if not isinstance(value, str) or not value:
            raise ModelsError(
                "model: use a name, an address (hf://... or https://...) or an "
                'object with "engine"'
            )
        if value in self.models:
            return self.models[value]
        if value in self.jobs:
            return self.jobs[value]
        if is_address(value):
            spec = _with_local(url_file(value, "checkpoints"), self.folder)
            alias = PurePosixPath(spec.name).stem
            model = Model(alias, "illustrious", {"checkpoint": spec})
            return self._remember(model, 'write it as an object with a "name"')
        hits = self._hits(("checkpoints", "diffusion_models"), value)
        if len(hits) > 1:
            raise ModelsError(
                f"model {value!r} matches {hits['checkpoints'].dest} and "
                f"{hits['diffusion_models'].dest} (use the sub-folder path or rename "
                "one)"
            )
        if spec := hits.get("checkpoints"):
            return Model(value, "illustrious", {"checkpoint": spec})
        if spec := hits.get("diffusion_models"):
            return self._fine_tune(value, spec)
        raise ModelsError(
            self._unknown(
                "model", value, self.models, "checkpoints", "diffusion_models/<engine>"
            )
        )

    def _base(self, name: str, label: str) -> Model:
        if BASE_MODELS[name] not in self.models:
            raise ModelsError(f"{label}: the built-in {BASE_MODELS[name]} is missing")
        return self.models[BASE_MODELS[name]]

    def _fine_tune(self, alias: str, spec: FileSpec) -> Model:
        """diffusion_models/<engine>/file: the rest from the engine's built-in model."""
        parts = spec.dest.split("/")
        eng = parts[1] if len(parts) > 2 else ""
        if eng not in BASE_MODELS:
            raise ModelsError(
                f"model {alias!r}: the engine of {spec.dest} is unknown — move it to "
                f"diffusion_models/<engine>/ ({', '.join(BASE_MODELS)}), or write the "
                'model as an object with an "engine" (an SDXL-type checkpoint goes '
                "in checkpoints/)"
            )
        base = self._base(eng, f"model {alias!r}")
        return Model(alias, eng, {**base.files, "diffusion": spec})

    def _object_model(self, raw: dict[str, Any]) -> Model:
        label = "model object"
        keys = {k for k in raw if not k.startswith("_")}
        unknown = keys - {"engine", "name", *FOLDERS}
        if unknown:
            raise ModelsError(
                f"{label}: unknown keys {sorted(unknown)} (use engine, name and the "
                f"file roles {', '.join(FOLDERS)})"
            )
        if not raw.get("engine"):
            raise ModelsError(f'{label}: needs an "engine" ({", ".join(ENGINES)})')
        eng = engine(str(raw["engine"]))
        roles = {role: raw[role] for role in FOLDERS if role in raw}
        wrong = set(roles) - set(eng.files)
        if wrong:
            raise ModelsError(
                f"{label}: {eng.name} does not use {sorted(wrong)} "
                f"(it loads {', '.join(eng.files)})"
            )
        files = {}
        if eng.name in BASE_MODELS:
            files = dict(self._base(eng.name, label).files)
        missing = [r for r in eng.files if r not in roles and r not in files]
        if missing:
            raise ModelsError(f"{label}: {eng.name} needs {', '.join(missing)}")
        for role, value in roles.items():
            files[role] = self._role_file(value, role, f"{label}.{role}")
        main = files.get("checkpoint") or files["diffusion"]
        alias = str(raw.get("name") or PurePosixPath(main.name).stem)
        if not alias or any(c in alias for c in "/\\") or alias.startswith("."):
            raise ModelsError(f"{label}: invalid name {alias!r} (a plain folder name)")
        return self._remember(Model(alias, eng.name, files), 'give it another "name"')

    def _role_file(self, value: Any, role: str, label: str) -> FileSpec:
        folder = FOLDERS[role]
        if not isinstance(value, str) or not value:
            raise ModelsError(f"{label}: needs an address or a file name")
        if is_address(value):
            return _with_local(url_file(value, folder), self.folder)
        if spec := self._hits((folder,), value, stored=False).get(folder):
            return spec
        root = f"{self.folder.root.as_posix()}/" if self.folder else ""
        where = f"{root}{folder}/"
        raise ModelsError(
            f"{label}: {value!r} is not an address (hf://... or https://...) or a "
            f"file in {where}"
        )

    def _remember(self, model: Model, hint: str) -> Model:
        """A job's own model, kept under the name its renders and outputs use."""
        if model.alias in self.models:
            raise ModelsError(
                f"model name {model.alias!r} is also a built-in alias; {hint}"
            )
        known = self.jobs.setdefault(model.alias, model)
        if known != model:
            raise ModelsError(
                f"two different models are named {model.alias!r}; {hint}"
            )
        return model

    def lora(self, name: str) -> Lora:
        if name in self.loras:
            return self.loras[name]
        if is_address(name):
            return Lora(name, "", _with_local(url_file(name, "loras"), self.folder))
        if spec := self._hits(("loras",), name).get("loras"):
            parts = spec.dest.split("/")
            eng = parts[1] if len(parts) > 2 and parts[1] in ENGINES else ""
            return Lora(name, eng, spec)
        raise ModelsError(self._unknown("LoRA", name, self.loras, "loras"))


def hf_url(path: str) -> str:
    """hf://owner/repo[@revision]/path/in/repo → resolve URL (revision: main)."""
    parts = path.split("/")
    if len(parts) < 3 or not all(parts):
        raise ModelsError(f"hf:// path needs owner/repo/file: hf://{path}")
    owner, repo, rest = parts[0], parts[1], "/".join(parts[2:])
    repo, _, revision = repo.partition("@")
    return (
        f"https://huggingface.co/{owner}/{repo}/resolve/"
        f"{quote(revision or 'main', safe='')}/{quote(rest)}"
    )


def _leaf(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return PurePosixPath(urlparse(source).path).name
    return PurePosixPath(source.replace("\\", "/")).name


def url_file(address: str, folder: str) -> FileSpec:
    """An address written in a job → <folder>/_url/<sha1 8>/<file name>.

    The hash keeps two addresses with the same file name apart. No SHA-256 or
    size is known: the pod accepts the download as it comes.
    """
    url = hf_url(address[5:]) if address.startswith("hf://") else address
    leaf = _leaf(address)
    if not leaf or leaf.startswith("."):
        raise ModelsError(f"{address}: the address must end in a file name")
    digest = hashlib.sha1(address.encode("utf-8")).hexdigest()[:8]
    return FileSpec(f"{folder}/{URL_DIR}/{digest}/{leaf}", url=url)


def _with_local(spec: FileSpec, root: ModelsFolder | None) -> FileSpec:
    """Note a copy of a URL file that already sits in the models folder."""
    if root is None:
        return spec
    path = root.path(spec.dest)
    if not path.is_file():
        try:
            rel = root.find(spec.folder, PurePosixPath(spec.name).name)
        except AmbiguousFile:
            rel = ""
        if not rel:
            return spec
        path = root.path(rel)
    if spec.size and path.stat().st_size != spec.size:
        return spec
    return replace(spec, local=str(path))


def _file(
    raw: Any, folder: str, base: Path, label: str, root: ModelsFolder | None = None
) -> FileSpec:
    if isinstance(raw, str):
        raw = {"source": raw}
    if not isinstance(raw, dict):
        raise ModelsError(f"{label}: expected a path/URL string or an object")
    unknown = set(raw) - {"source", "url", "path", "name", "sha256", "size"}
    if unknown:
        raise ModelsError(f"{label}: unknown keys {sorted(unknown)}")
    source = str(raw.get("source") or raw.get("url") or raw.get("path") or "")
    if not source:
        raise ModelsError(f"{label}: needs a path or URL")
    name = str(raw.get("name") or _leaf(source)).strip("/")
    parts = name.split("/")
    if not name or "" in parts or ".." in parts or name.startswith("."):
        raise ModelsError(f"{label}: invalid name {name!r}")
    sha256 = str(raw.get("sha256", "")).lower()
    if sha256 and (len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)):
        raise ModelsError(f"{label}: sha256 must be 64 hex characters")
    size = int(raw.get("size", 0))
    dest = f"{folder}/{name}"
    if source.startswith("hf://"):
        url = hf_url(source[5:])
        return _with_local(FileSpec(dest, url=url, sha256=sha256, size=size), root)
    if source.startswith(("http://", "https://")):
        return _with_local(FileSpec(dest, url=source, sha256=sha256, size=size), root)
    path = Path(os.path.expanduser(source))
    if not path.is_absolute():
        path = base / path
    return FileSpec(dest, local=str(path), sha256=sha256, size=size)


def _models(
    raw: dict[str, Any], base: Path, root: ModelsFolder | None
) -> dict[str, Model]:
    done: dict[str, Model] = {}
    for alias, entry in raw.items():
        label = f"models.{alias}"
        if not isinstance(entry, dict):
            raise ModelsError(f"{label}: must be an object")
        files = entry.get("files") or {}
        if not isinstance(files, dict):
            raise ModelsError(f"{label}.files: must be an object")
        eng = engine(str(entry.get("engine") or ""))
        unknown = set(files) - set(eng.files)
        missing = [r for r in eng.files if r not in files]
        if unknown:
            raise ModelsError(
                f"{label}: {eng.name} does not use {sorted(unknown)} "
                f"(it needs {', '.join(eng.files)})"
            )
        if missing:
            raise ModelsError(f"{label}: {eng.name} also needs {', '.join(missing)}")
        specs = {
            role: _file(files[role], FOLDERS[role], base, f"{label}.files.{role}", root)
            for role in eng.files
        }
        done[alias] = Model(alias, eng.name, specs)
    return done


def parse(
    data: Any, base: Path = Path("."), folder: ModelsFolder | None = None
) -> Registry:
    """The catalog format (internal: catalog.json and test fixtures)."""
    if not isinstance(data, dict):
        raise ModelsError("the catalog must be a JSON object")
    if data.get("version", 1) != 1:
        raise ModelsError(f"catalog version {data.get('version')} is not supported")
    raw_models = data.get("models") or {}
    if not isinstance(raw_models, dict):
        raise ModelsError("models: must be an object")
    models = _models(raw_models, base, folder)
    loras: dict[str, Lora] = {}
    for alias, entry in (data.get("loras") or {}).items():
        label = f"loras.{alias}"
        if not isinstance(entry, dict):
            raise ModelsError(f'{label}: needs an object with "engine" and "file"')
        eng = engine(str(entry.get("engine", "")))
        if "file" not in entry:
            raise ModelsError(f"{label}: needs a file")
        # loras/<engine>/ keeps same-named files of different engines apart.
        spec = _file(entry["file"], f"loras/{eng.name}", base, f"{label}.file", folder)
        loras[alias] = Lora(alias, eng.name, spec, str(entry.get("trigger", "")))
    _check_conflicts(models, loras)
    return Registry(models, loras, folder)


def _check_conflicts(models: dict[str, Model], loras: dict[str, Lora]) -> None:
    """Two entries may share a pod file name only if they are the same file."""
    seen: dict[str, tuple[str, FileSpec]] = {}
    specs = [(m.alias, f) for m in models.values() for f in m.files.values()]
    specs += [(lo.alias, lo.file) for lo in loras.values()]
    for owner, spec in specs:
        if spec.dest not in seen:
            seen[spec.dest] = (owner, spec)
            continue
        other_owner, other = seen[spec.dest]
        same = (other.sha256 and other.sha256 == spec.sha256) or (
            other.source == spec.source
        )
        if not same:
            raise ModelsError(
                f"{owner} and {other_owner} both name {spec.dest} but point at "
                'different files; give one of them a "name"'
            )


def catalog(folder: ModelsFolder | None = None) -> Registry:
    """The built-in aliases: official releases and example LoRAs."""
    package = resources.files("jena_image_render")
    text = package.joinpath("catalog.json").read_text("utf-8")
    return parse(json.loads(text), folder=folder)


def load(
    folder: ModelsFolder | None = None,
    remote: Remote | None = None,
    comfy: ComfyFolders | None = None,
) -> Registry:
    """The catalog, the models folder, a local ComfyUI's folders (`comfy`) and
    (with `remote`) your storage."""
    registry = catalog(folder or ModelsFolder())
    return replace(registry, remote=remote, comfy=comfy)


def missing_local(files: list[FileSpec]) -> list[str]:
    """Files with no URL whose local copy is not there."""
    return [f.local for f in files if not f.url and not Path(f.local).is_file()]


__all__ = ["ENGINES", "FileSpec", "Lora", "Model", "ModelsError", "Registry", "load"]
