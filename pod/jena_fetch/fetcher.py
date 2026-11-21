"""Background model downloads with hash verification for the ComfyUI pod."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class FetchItem:
    url: str
    dest: str  # relative to the ComfyUI models folder, e.g. "vae/ae.safetensors"
    sha256: str = ""  # empty: not verified (the user's own storage, no hash known)
    size: int = 0  # 0: taken from Content-Length


def resolve_dest(models_dir: Path, dest: str) -> Path:
    """Refuse empty, absolute and escaping paths."""
    if not dest or Path(dest).is_absolute() or ".." in Path(dest).parts:
        raise ValueError(f"invalid dest: {dest!r}")
    root = models_dir.resolve()
    target = (root / dest).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"dest escapes models folder: {dest!r}")
    return target


class Fetcher:
    def __init__(
        self,
        models_dir: Path,
        civitai_token: str = "",
        opener: Callable[..., Any] = requests.get,
        hf_token: str = "",
    ) -> None:
        self.models_dir = models_dir
        self.civitai_token = civitai_token
        self.hf_token = hf_token  # gated Hugging Face repos (e.g. FLUX.2 klein 9B)
        self.opener = opener
        self._state: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._state.items()}

    def busy(self) -> bool:
        with self._lock:
            return any(v["status"] in ("queued", "downloading") for v in self._state.values())

    def submit(self, items: list[FetchItem]) -> None:
        targets = [(item, resolve_dest(self.models_dir, item.dest)) for item in items]
        for item, target in targets:
            with self._lock:
                current = self._state.get(item.dest, {}).get("status")
                if current in ("queued", "downloading"):
                    continue
                self._state[item.dest] = {"status": "queued", "bytes": 0, "total": item.size}
            threading.Thread(target=self._run, args=(item, target), daemon=True).start()

    def wait(self, timeout: float) -> dict[str, dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.busy():
            time.sleep(0.01)
        return self.status()

    def _set(self, dest: str, **values: Any) -> None:
        with self._lock:
            self._state[dest].update(values)

    @staticmethod
    def _marker(target: Path) -> Path:
        return target.with_name(target.name + ".sha256")

    @staticmethod
    def _source(target: Path) -> Path:
        return target.with_name(target.name + ".url")

    def _verified(self, target: Path, item: FetchItem) -> bool:
        """Already here: same hash, or (no hash given) fetched from the same URL."""
        marker = self._marker(target)
        if not (target.is_file() and marker.is_file()):
            return False
        if item.sha256:
            return marker.read_text().strip() == item.sha256
        source = self._source(target)
        return source.is_file() and source.read_text().strip() == item.url

    def _headers(self, url: str) -> dict[str, str]:
        host = urlparse(url).hostname or ""
        if self.civitai_token and (host == "civitai.com" or host.endswith(".civitai.com")):
            return {"Authorization": f"Bearer {self.civitai_token}"}
        if self.hf_token and (host == "huggingface.co" or host.endswith(".huggingface.co")):
            # requests drops this header when HF redirects to its CDN host.
            return {"Authorization": f"Bearer {self.hf_token}"}
        return {}

    def _run(self, item: FetchItem, target: Path) -> None:
        if self._verified(target, item):
            size = target.stat().st_size
            self._set(item.dest, status="done", bytes=size, total=size)
            return
        partial = target.with_name(target.name + ".part")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._set(item.dest, status="downloading")
            digest, done = hashlib.sha256(), 0
            with self.opener(
                item.url, headers=self._headers(item.url), stream=True, timeout=(30, 300)
            ) as response:
                response.raise_for_status()
                if not item.size:
                    length = int(response.headers.get("content-length") or 0)
                    self._set(item.dest, total=length)
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(CHUNK):
                        stream.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        self._set(item.dest, bytes=done)
            if item.sha256 and digest.hexdigest() != item.sha256:
                raise ValueError(f"sha256 mismatch for {item.dest}")
            partial.replace(target)
            self._marker(target).write_text(digest.hexdigest())
            self._source(target).write_text(item.url)
            self._set(item.dest, status="done", total=done)
        except Exception as exc:  # every failure is reported to the client
            partial.unlink(missing_ok=True)
            self._set(item.dest, status="error", error=str(exc))
