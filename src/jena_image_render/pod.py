"""RunPod pod lifecycle: start (Community once, then Secure), reuse, stop.

Cost guards: a pod that fails to start is always terminated; the pod image
terminates itself after IDLE_SECONDS without activity and after MAX_SECONDS in
total; Ctrl+C during a run terminates it right away.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from jena_image_render import env
from jena_image_render.comfy import ComfyClient
from jena_image_render.runpod_api import NoCapacity, RunPodAPI, proxy_url

LOGGER = logging.getLogger(__name__)
USER = "render"
POD_PREFIX = "jena-render-"
LIVE = {"RUNNING", "CREATED", "STARTING", "RESTARTING"}
POLL_SECONDS = 10.0
DIAG_SECONDS = 30.0
READY_TIMEOUT = 20 * 60
START_ATTEMPTS = 3
CRASH_RESTARTS = 2
TYPICAL_START_SECONDS = 5 * 60  # a healthy host: container in 3–4.5 min, then ComfyUI

# Start progress for a UI: {"phase": ..., "cloud": ..., "attempt": ..., ...}.
Progress = Callable[[dict[str, Any]], None]


@dataclass
class Settings:
    """runpod.json (optional) — which GPUs to rent and how long to wait."""

    community: list[str] = field(
        default_factory=lambda: [
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX A6000",
            "NVIDIA A40",
        ]
    )
    secure: list[str] = field(
        default_factory=lambda: [
            "NVIDIA RTX A5000",
            "NVIDIA A40",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX A6000",
            "NVIDIA L4",
        ]
    )
    # Host RAM tiers in GB, roomiest first. Qwen-Image offloads ~32GB to RAM.
    host_memory_gb: list[int] = field(default_factory=lambda: [62, 48, 32])
    # The image is built on CUDA 12.6; newer host drivers are fine.
    cuda_versions: list[str] = field(
        default_factory=lambda: ["12.6", "12.7", "12.8", "12.9", "13.0", "13.1", "13.2"]
    )
    idle_minutes: int = 20
    disk_gb: int = 100
    # A healthy host starts the container in 3–4.5 min; a host still pulling
    # after this is replaced (Community is tried once, then Secure).
    community_pull_minutes: float = 5
    secure_pull_minutes: float = 8

    @classmethod
    def load(cls, path: Path = Path("runpod.json")) -> "Settings":
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text("utf-8"))
        known = set(cls.__dataclass_fields__)
        unknown = [k for k in data if k not in known and not k.startswith("_")]
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown}")
        settings = cls(**{k: v for k, v in data.items() if k in known})
        for name in ("community", "secure", "host_memory_gb", "cuda_versions"):
            value = getattr(settings, name)
            if not isinstance(value, list) or not value:
                raise ValueError(f"{path}: {name} must be a non-empty list")
        return settings


class StartFailed(RuntimeError):
    """The pod could not be created or did not become ready."""


class StuckPull(Exception):
    """The host never started the container; another host may do better."""


class Cancelled(Exception):
    """The caller asked to stop while a pod was starting (the pod is terminated)."""


class Sessions:
    """The pod this tool started, remembered in .jena/session.json."""

    def __init__(self, folder: Path = env.STATE_DIR) -> None:
        self.path = folder / "session.json"

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        data: dict[str, Any] = json.loads(self.path.read_text("utf-8"))
        return data

    def save(self, session: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(session, indent=2) + "\n", "utf-8")
        os.chmod(self.path, 0o600)  # holds the pod login

    def forget(self) -> None:
        self.path.unlink(missing_ok=True)


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class Pods:
    def __init__(
        self,
        api: RunPodAPI,
        settings: Settings | None = None,
        sessions: Sessions | None = None,
        image: str = "",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api = api
        self.settings = settings or Settings()
        self.sessions = sessions or Sessions()
        self.image = image or env.get("JENA_POD_IMAGE", env.DEFAULT_POD_IMAGE)
        self.sleep = sleep
        self.clock = clock

    def client(self, session: dict[str, Any]) -> ComfyClient:
        return ComfyClient(session["url"], session["user"], session["password"])

    # ── reuse ────────────────────────────────────────────────────────
    def live(self) -> dict[str, Any] | None:
        """The recorded pod, or None (and forget it) once it has shut down."""
        session = self.sessions.load()
        if session is None:
            return None
        pod = self.api.get_pod(session["pod_id"])
        if pod and pod.get("desiredStatus") in LIVE:
            return session
        LOGGER.info("Pod %s is gone (idle shutdown?); forgetting it", session["pod_id"])
        self.sessions.forget()
        return None

    # ── start ────────────────────────────────────────────────────────
    def start(
        self,
        max_hours: float,
        secure: bool = False,
        progress: Progress | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Rent and wait for a pod. `progress` hears each phase; `should_stop`
        returning True terminates the pod being started (Cancelled)."""
        report = progress or (lambda event: None)
        stop = should_stop or (lambda: False)
        password = secrets.token_urlsafe(24)
        pod_env = {
            "COMFY_USER": USER,
            "COMFY_PASSWORD": password,
            "IDLE_SECONDS": str(self.settings.idle_minutes * 60),
            "MAX_SECONDS": str(int(max_hours * 3600)),
        }
        if token := env.get("HF_TOKEN"):
            pod_env["HF_TOKEN"] = token
        if token := env.get("CIVITAI_TOKEN"):
            pod_env["CIVITAI_API_TOKEN"] = token
        if stop():
            raise Cancelled("pod start cancelled")
        if not secure:
            try:
                report({"phase": "renting", "cloud": "community", "attempt": 1})
                return self._attempt(
                    pod_env, password, max_hours, False,
                    self.settings.community_pull_minutes * 60, report, stop,
                )
            except (StuckPull, StartFailed) as exc:
                LOGGER.warning("Community start failed (%s); trying Secure Cloud", exc)
        for attempt in range(1, START_ATTEMPTS + 1):
            if stop():
                raise Cancelled("pod start cancelled")
            try:
                report({"phase": "renting", "cloud": "secure", "attempt": attempt})
                return self._attempt(
                    pod_env, password, max_hours, True,
                    self.settings.secure_pull_minutes * 60, report, stop,
                )
            except StuckPull as exc:
                LOGGER.warning(
                    "Secure attempt %s/%s: %s; trying another host",
                    attempt, START_ATTEMPTS, exc,
                )
        raise StartFailed(f"{START_ATTEMPTS} hosts in a row never started the container")

    def _attempt(
        self,
        pod_env: dict[str, str],
        password: str,
        max_hours: float,
        secure: bool,
        pull_timeout: float,
        report: Progress | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Create one pod and wait for it; any failure terminates that pod."""
        pod = self._create(pod_env, secure)
        try:
            return self._await_ready(
                pod, password, max_hours, pull_timeout, report, stop
            )
        except BaseException:  # timeout, Ctrl+C or any error: never leave it billing
            LOGGER.warning("Start did not finish; terminating pod %s", pod["id"])
            self.api.terminate(pod["id"])
            self.sessions.forget()
            raise

    def _create(self, pod_env: dict[str, str], secure: bool) -> dict[str, str]:
        name = f"{POD_PREFIX}{secrets.token_hex(3)}"
        s = self.settings
        try:
            return self.api.create_pod(
                name,
                self.image,
                pod_env,
                s.secure if secure else s.community,
                secure=secure,
                host_memory_gb=s.host_memory_gb,
                cuda_versions=s.cuda_versions,
                disk_gb=s.disk_gb,
            )
        except NoCapacity as exc:
            raise StartFailed(f"No GPU available, nothing was created: {exc}") from exc
        except requests.RequestException as exc:
            # The request may have created the pod before the connection dropped.
            for orphan in self.api.account()["pods"]:
                if orphan.get("name") == name:
                    self.api.terminate(orphan["id"])
                    LOGGER.warning("Terminated pod %s left by the failed request", orphan["id"])
            raise StartFailed(f"Pod creation request failed: {exc}") from exc

    def _await_ready(
        self,
        pod: dict[str, str],
        password: str,
        max_hours: float,
        pull_timeout: float,
        report: Progress | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        session = {
            "pod_id": pod["id"],
            "gpu": pod["gpu"],
            "url": proxy_url(pod["id"]),
            "user": USER,
            "password": password,
            "created": datetime.now(timezone.utc).isoformat(),
            "max_hours": max_hours,
        }
        self.sessions.save(session)
        LOGGER.info(
            "Pod %s on %s — idle shutdown %smin, hard limit %.1fh. Waiting for ComfyUI…",
            pod["id"], pod["gpu"], self.settings.idle_minutes, max_hours,
        )
        client = self.client(session)
        started = self.clock()
        next_diag = started
        container_started = False
        last_uptime: int | None = None
        restarts = 0
        info = {"pod_id": pod["id"], "gpu": pod["gpu"], "created": session["created"],
                "typical_seconds": TYPICAL_START_SECONDS}
        while True:
            if stop is not None and stop():
                raise Cancelled("pod start cancelled")
            if client.ready():
                LOGGER.info("Ready after %s: %s", _clock(self.clock() - started), session["url"])
                if report is not None:
                    report({**info, "phase": "ready", "elapsed": self.clock() - started})
                return session
            elapsed = self.clock() - started
            if report is not None:
                phase = "starting ComfyUI" if container_started else "pulling image"
                report({**info, "phase": phase, "elapsed": elapsed})
            if self.clock() >= next_diag:
                uptime = self._log_startup(client, pod["id"], elapsed)
                if uptime is not None:
                    container_started = True
                    if last_uptime is not None and uptime < last_uptime:
                        restarts += 1  # uptime went back: the container restarted
                    last_uptime = uptime
                if restarts >= CRASH_RESTARTS:
                    raise StuckPull(f"container restarted {restarts} times (crash loop)")
                next_diag = self.clock() + DIAG_SECONDS
            if not container_started and elapsed >= pull_timeout:
                raise StuckPull(f"container not started after {_clock(elapsed)}")
            if elapsed >= READY_TIMEOUT:
                raise StartFailed("ComfyUI did not answer in time; the pod was terminated")
            self.sleep(POLL_SECONDS)

    def _log_startup(self, client: ComfyClient, pod_id: str, elapsed: float) -> int | None:
        try:
            pod = self.api.get_pod(pod_id) or {}
        except (RuntimeError, OSError) as exc:
            pod = {"desiredStatus": f"api error {exc}"}
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds")
        LOGGER.info(
            "[start %s] status %s · container %s · proxy %s",
            _clock(elapsed),
            pod.get("desiredStatus"),
            f"up {uptime}s" if uptime is not None else "pulling image",
            client.probe(),
        )
        return int(uptime) if uptime is not None else None

    # ── stop / status ────────────────────────────────────────────────
    def _ours(self, account: dict[str, Any]) -> list[dict[str, Any]]:
        return [p for p in account["pods"] if str(p.get("name", "")).startswith(POD_PREFIX)]

    def stop(self) -> bool:
        """Terminate the session pod and any other pod this tool created; confirm."""
        session = self.sessions.load()
        targets = {session["pod_id"]} if session else set()
        targets |= {p["id"] for p in self._ours(self.api.account())}
        for target in sorted(targets):
            self.api.terminate(target)
            LOGGER.info("Terminate requested for %s", target)
        account: dict[str, Any] = {}
        for _ in range(30):
            account = self.api.account()
            if not self._ours(account):
                self.sessions.forget()
                LOGGER.info(
                    "No pods left. Spend now $%s/h, balance $%s",
                    account.get("currentSpendPerHr"), account.get("clientBalance"),
                )
                return True
            self.sleep(POLL_SECONDS)
        LOGGER.error("Pods still listed: %s — check the RunPod console", self._ours(account))
        return False

    def status(self) -> dict[str, Any]:
        account = self.api.account()
        session = self.sessions.load()
        if session and session["pod_id"] not in {p["id"] for p in account["pods"]}:
            self.sessions.forget()
            session = None
        return {
            "balance": account.get("clientBalance"),
            "spend_per_hour": account.get("currentSpendPerHr"),
            "pods": self._ours(account),
            "session": {k: v for k, v in (session or {}).items() if k != "password"},
        }
