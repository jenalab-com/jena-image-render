"""Your own storage for files that exist only on this computer.

With `JENA_STORAGE=hf://owner/repo` (and a write `HF_TOKEN`), a local file is
uploaded once to that private Hugging Face repo at its models-folder path, and
the pod downloads it from there instead of waiting for an upload from home.
What was uploaded is remembered in .jena/storage.json by SHA-256, so a file is
sent again only when it changes. That record is per computer, so the repo itself
is asked too (its file list, once per run, only when needed): a file already
there with the same SHA-256 is not uploaded again, and a name that is neither
built in nor in the models folder is looked up in it by the same folder rules.

A backend is anything with `sync(spec) -> spec`: S3-compatible storage can be
added next to HFStorage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from jena_image_render import env
from jena_image_render.folder import weight
from jena_image_render.models import FileSpec

LOGGER = logging.getLogger(__name__)
INSTALL = "pip install huggingface_hub"


class StorageError(RuntimeError):
    """Storage is not set up, or an upload failed."""


def configured() -> str:
    """The JENA_STORAGE value, or "" when there is no storage."""
    return env.get("JENA_STORAGE")


def _hf_api(token: str) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise StorageError(
            f"JENA_STORAGE=hf://... needs huggingface_hub: {INSTALL}"
        ) from exc
    return HfApi(token=token)


def _not_found(exc: Exception) -> bool:
    """A repo that does not exist (or that this token cannot see) answers 404."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


class HFStorage:
    def __init__(
        self,
        repo: str,
        token: str,
        api: Callable[[str], Any] = _hf_api,
        state: Path | None = None,
    ) -> None:
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise StorageError(f"JENA_STORAGE must be hf://owner/repo, not hf://{repo}")
        if not token:
            raise StorageError("JENA_STORAGE needs HF_TOKEN with write access in .env")
        self.repo = repo
        self.token = token
        self._make_api = api
        self._client: Any = None
        self._created = False
        self._listing: dict[str, FileSpec] | None = None
        self.head = ""  # the commit the listing was taken at
        self.path = state or env.STATE_DIR / "storage.json"
        self.state: dict[str, dict[str, Any]] = (
            json.loads(self.path.read_text("utf-8")) if self.path.is_file() else {}
        )

    def url(self, dest: str, revision: str) -> str:
        return (
            f"https://huggingface.co/{self.repo}/resolve/"
            f"{quote(revision, safe='')}/{quote(dest)}"
        )

    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_api(self.token)
        return self._client

    def api(self) -> Any:
        """The client, with the repo created (private) if it is not there yet."""
        if not self._created:
            self.client().create_repo(self.repo, private=True, exist_ok=True)
            self._created = True
        return self._client

    def listing(self) -> dict[str, FileSpec]:
        """Every weight file in the repo, pinned to its current commit.

        Asked once per run. A repo that does not exist yet is simply empty; any
        other failure (network, token, no huggingface_hub) is one warning, and
        the repo counts as empty.
        """
        if self._listing is None:
            self._listing = {}
            try:
                info = self.client().model_info(self.repo, files_metadata=True)
                self.head = str(info.sha)
                for item in info.siblings or []:
                    path = str(item.rfilename)
                    if weight(path):
                        self._listing[path] = FileSpec(
                            path,
                            url=self.url(path, self.head),
                            sha256=_lfs(item.lfs, "sha256") or "",
                            size=int(item.size or _lfs(item.lfs, "size") or 0),
                        )
            except Exception as exc:  # the hub raises many kinds of errors
                self._listing = {}
                if _not_found(exc):  # first use: the repo is created on upload
                    LOGGER.info("hf://%s does not exist yet", self.repo)
                else:
                    reason = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                    LOGGER.warning(
                        "Could not list hf://%s: %s", self.repo, reason or type(exc).__name__
                    )
        return self._listing

    def files(self, folder: str) -> list[FileSpec]:
        """The repo's weight files under one role folder."""
        return [f for p, f in self.listing().items() if p.startswith(folder + "/")]

    def sync(self, spec: FileSpec) -> FileSpec:
        """A hashed local file → the same file with its storage URL."""
        if spec.url or not spec.sha256:
            raise StorageError(f"{spec.dest}: only hashed local files go to storage")
        entry = self.state.get(spec.dest) or {}
        if entry.get("repo") == self.repo and entry.get("sha256") == spec.sha256:
            revision = str(entry["revision"])
        else:
            revision = self._upload(spec)
            self.state[spec.dest] = {
                "repo": self.repo,
                "sha256": spec.sha256,
                "size": spec.size,
                "revision": revision,
            }
            self._save()
        return FileSpec(
            spec.dest,
            url=self.url(spec.dest, revision),
            local=spec.local,
            sha256=spec.sha256,
            size=spec.size,
        )

    def _upload(self, spec: FileSpec) -> str:
        """The revision holding `spec`: uploaded now unless the repo has it."""
        stored = self.listing().get(spec.dest)
        if stored is not None and stored.sha256 == spec.sha256:
            LOGGER.info("Already in hf://%s: %s (not uploaded)", self.repo, spec.dest)
            return self.head
        LOGGER.info(
            "Uploading %s to hf://%s (%.2f GB, once)",
            spec.dest,
            self.repo,
            spec.size / 1e9,
        )
        try:
            info = self.api().upload_file(
                path_or_fileobj=spec.local,
                path_in_repo=spec.dest,
                repo_id=self.repo,
                commit_message=f"Add {spec.dest}",
            )
        except StorageError:
            raise
        except Exception as exc:  # the hub raises many kinds of errors
            raise StorageError(f"upload of {spec.dest} failed: {exc}") from exc
        return str(info.oid)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1) + "\n", "utf-8")


def _lfs(info: Any, key: str) -> Any:
    """A field of the hub's LFS info (an object or a dict, by version)."""
    if info is None:
        return None
    if isinstance(info, dict):
        return info.get(key)
    return getattr(info, key, None)


def open_storage(value: str | None = None, **kwargs: Any) -> HFStorage:
    """The storage named by JENA_STORAGE (or `value`)."""
    value = configured() if value is None else value
    if not value:
        raise StorageError("no storage: set JENA_STORAGE=hf://owner/repo in .env")
    if not value.startswith("hf://"):
        raise StorageError(
            f"JENA_STORAGE={value}: only hf://owner/repo is supported for now"
        )
    return HFStorage(value[5:].strip("/"), env.get("HF_TOKEN"), **kwargs)


def sync_all(store: HFStorage, files: list[FileSpec]) -> list[FileSpec]:
    """Local-only files go through storage; URL files are left as they are."""
    return [f if f.url else store.sync(f) for f in files]


class Lookup:
    """Your storage for one run: opened on first use, then shared.

    Called with a role folder, it returns the stored weight files under it, for
    names that are neither built in nor in the models folder. If the storage
    cannot be opened, that is one warning and nothing is found.
    """

    def __init__(self, opener: Callable[[], HFStorage] = open_storage) -> None:
        self._opener = opener
        self._store: HFStorage | None = None
        self._failed = False

    def store(self) -> HFStorage:
        if self._store is None:
            self._store = self._opener()
        return self._store

    def __call__(self, folder: str) -> list[FileSpec]:
        if self._failed:
            return []
        try:
            return self.store().files(folder)
        except StorageError as exc:
            LOGGER.warning("Storage not searched: %s", exc)
            self._failed = True
            return []
