"""ComfyUI HTTP + websocket API: a local server, or the pod behind basic auth."""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, Callable, Iterable, Iterator

import certifi
import requests
import websocket

from jena_image_render.models import FileSpec

StepCallback = Callable[[int, int], None]
UPLOAD_CHUNK = 32 << 20  # RunPod's proxy rejects request bodies around 100MB
POLL_SECONDS = 2.0
JOB_TIMEOUT = 45 * 60

# Loader node → input that lists the files of one models/ folder.
LOADER_INPUTS = {
    "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
    "diffusion_models": ("UNETLoader", "unet_name"),
    "text_encoders": ("CLIPLoader", "clip_name"),
    "vae": ("VAELoader", "vae_name"),
    "loras": ("LoraLoaderModelOnly", "lora_name"),
}


GET_ATTEMPTS = 6  # read-only calls ride out a few seconds of proxy trouble
GET_RETRY_SECONDS = 2.0
GATEWAY_ERRORS = {502, 503, 504, 520, 521, 522, 523, 524}


def _transient(response: Any) -> bool:
    """A proxy hiccup, not ComfyUI's answer: a gateway error or an empty 404."""
    status = response.status_code
    if status in GATEWAY_ERRORS:
        return True
    return status == 404 and not (getattr(response, "text", "") or "").strip()


class ComfyError(Exception):
    """ComfyUI rejected or failed a request."""


def watch_progress(
    frames: Iterable[str | bytes], prompt_id: str, on_step: StepCallback
) -> str:
    """Report sampler steps for one prompt; "done" when it finishes, "" if unknown."""
    for frame in frames:
        if isinstance(frame, bytes):  # preview images
            continue
        message = json.loads(frame)
        data = message.get("data") or {}
        if data.get("prompt_id") != prompt_id:
            continue
        kind = message.get("type")
        if kind == "progress":
            on_step(int(data["value"]), int(data["max"]))
        elif kind == "execution_error":
            raise ComfyError(f"execution error: {data.get('exception_message')}")
        elif kind == "executing" and data.get("node") is None:
            return "done"
        elif kind == "execution_success":
            return "done"
    return ""


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        user: str = "",
        password: str = "",
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base = base_url.rstrip("/")
        self.session = session or requests.Session()
        self._sleep = sleep
        self._headers: list[str] = []
        if user or password:
            self.session.auth = (user, password)
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            self._headers = [f"Authorization: Basic {token}"]
        self.client_id = str(uuid.uuid4())

    def _get(self, path: str, **kwargs: Any) -> Any:
        """GET that rides out a proxy hiccup.

        Right after a pod starts (and now and then later) RunPod's proxy answers
        with an empty 404 or a 502/503/504 for a second or two. These calls only
        read, so asking again is safe; a real answer is returned at once.
        """
        response: Any = None
        for attempt in range(GET_ATTEMPTS):
            if attempt:
                self._sleep(GET_RETRY_SECONDS)
            try:
                response = self.session.get(f"{self.base}{path}", **kwargs)
            except requests.RequestException:
                if attempt == GET_ATTEMPTS - 1:
                    raise
                continue
            if not _transient(response):
                break
        return response

    def _json(self, response: Any, action: str) -> Any:
        if response.status_code >= 400:
            raise ComfyError(
                f"{action} failed ({response.status_code}): {response.text[:2000]}"
            )
        return response.json()

    # ── health ───────────────────────────────────────────────────────
    def probe(self, timeout: float = 15) -> str:
        """HTTP status of /system_stats, for start-up diagnostics."""
        try:
            response = self.session.get(f"{self.base}/system_stats", timeout=timeout)
        except requests.RequestException as exc:
            return type(exc).__name__
        return str(response.status_code)

    def ready(self, timeout: float = 15) -> bool:
        return self.probe(timeout) == "200"

    def folder_paths(self, timeout: float = 15) -> dict[str, list[str]] | None:
        """ComfyUI's model folders per role (GET /internal/folder_paths).

        It is an internal API that a version may lack or change: None when it
        does not answer with {role: [path, ...]}.
        """
        try:
            response = self.session.get(
                f"{self.base}/internal/folder_paths", timeout=timeout
            )
            data = response.json() if response.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        paths = {
            str(role): [str(p) for p in value]
            for role, value in data.items()
            if isinstance(value, list) and all(isinstance(p, str) for p in value)
        }
        return paths or None

    # ── files on the server ──────────────────────────────────────────
    def options(self, folder: str) -> list[str]:
        """Files ComfyUI lists for one models/ folder (local mode lookups)."""
        node, field = LOADER_INPUTS[folder]
        response = self._get(f"/object_info/{node}", timeout=60)
        info = self._json(response, f"object_info {node}")
        spec = info.get(node, {}).get("input", {}).get("required", {}).get(field)
        if not spec:
            raise ComfyError(f"{node} is missing on this ComfyUI server")
        choices = spec[0]
        if isinstance(choices, str):  # newer ComfyUI: ["COMBO", {"options": [...]}]
            choices = (spec[1] or {}).get("options", [])
        return [str(c).replace("\\", "/") for c in choices]

    def fetch(self, files: list[FileSpec]) -> None:
        """Ask the pod to download URL files into models/ (jena_fetch)."""
        body = {
            "files": [
                {"url": f.url, "dest": f.dest, "sha256": f.sha256, "size": f.size}
                for f in files
            ]
        }
        # Asking twice is harmless: the pod skips files it is already fetching.
        response: Any = None
        for attempt in range(GET_ATTEMPTS):
            if attempt:
                self._sleep(GET_RETRY_SECONDS)
            response = self.session.post(f"{self.base}/jena/fetch", json=body, timeout=60)
            if not _transient(response):
                break
        self._json(response, "fetch")

    def fetch_status(self) -> dict[str, Any]:
        response = self._get("/jena/fetch", timeout=60)
        status: dict[str, Any] = self._json(response, "fetch status")
        return status

    def upload(
        self,
        item: FileSpec,
        chunk: int = UPLOAD_CHUNK,
        progress: StepCallback | None = None,
    ) -> None:
        """Send a local file in ordered chunks; the pod verifies the SHA-256."""
        ident = {"dest": item.dest, "sha256": item.sha256}
        response = self._get("/jena/upload", params=ident, timeout=60)
        if self._json(response, "upload status").get("verified"):
            return
        total, offset, status = item.size, 0, ""
        with open(item.local, "rb") as stream:
            while status != "done":
                data = stream.read(chunk)
                params: dict[str, str | int] = {**ident, "offset": offset, "total": total}
                response = self.session.post(
                    f"{self.base}/jena/upload", params=params, data=data, timeout=300
                )
                status = str(self._json(response, f"upload {item.dest}").get("status"))
                offset += len(data)
                if progress is not None:
                    progress(offset, total)
                if not data and status != "done":
                    raise ComfyError(f"upload of {item.dest} ended before verification")

    # ── rendering ────────────────────────────────────────────────────
    def connect(self) -> Any:
        """Open the progress websocket before queueing so no event is missed."""
        url = self.base.replace("https://", "wss://").replace("http://", "ws://")
        return websocket.create_connection(
            f"{url}/ws?clientId={self.client_id}",
            header=self._headers,
            sslopt={"ca_certs": certifi.where()},  # python.org builds lack system CAs
            timeout=60,
        )

    @staticmethod
    def frames(connection: Any) -> Iterator[str | bytes]:
        while True:
            try:
                yield connection.recv()
            except (websocket.WebSocketException, OSError):
                return

    def queue(self, graph: dict[str, Any]) -> str:
        body: dict[str, Any] = {"prompt": graph, "client_id": self.client_id}
        response = self.session.post(f"{self.base}/prompt", json=body, timeout=60)
        if response.status_code >= 400:
            raise ComfyError(f"prompt rejected: {response.text[:4000]}")
        return str(response.json()["prompt_id"])

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        response = self._get(f"/history/{prompt_id}", timeout=60)
        entry: dict[str, Any] | None = self._json(response, "history").get(prompt_id)
        return entry

    def image(self, filename: str, subfolder: str, kind: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": kind}
        response = self._get("/view", params=params, timeout=300)
        if response.status_code >= 400:
            raise ComfyError(f"view failed ({response.status_code}) for {filename}")
        content: bytes = response.content
        return content

    def run(
        self,
        graph: dict[str, Any],
        output_node: str,
        on_step: StepCallback | None = None,
        timeout: float = JOB_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> bytes:
        """Queue one graph, stream its steps, and return the first output image."""
        connection = None
        if on_step is not None:
            try:
                connection = self.connect()
            except Exception:  # progress is optional; history polling still works
                connection = None
        prompt_id = self.queue(graph)
        if connection is not None and on_step is not None:
            try:
                watch_progress(self.frames(connection), prompt_id, on_step)
            finally:
                connection.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = self.history(prompt_id)
            status = (entry or {}).get("status", {})
            if status.get("status_str") == "error":
                raise ComfyError(
                    f"execution error: {json.dumps(status.get('messages'))[:2000]}"
                )
            if entry and status.get("completed"):
                images = entry["outputs"][output_node]["images"]
                first = images[0]
                return self.image(
                    first["filename"], first.get("subfolder", ""), first.get("type", "temp")
                )
            sleep(POLL_SECONDS)
        raise ComfyError("generation timed out")

    def interrupt(self) -> None:
        self.session.post(f"{self.base}/interrupt", timeout=30)
