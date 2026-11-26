"""jenair ui: a web page for job files, served to this computer only.

    jenair ui [--port 8765] [--no-browser] [--outputs DIR]

The page (ui/index.html) cannot run commands, so it asks this server, which
calls the same functions as the command line: config.expand, models.load,
assets, local.fill, pod.Pods, runner.render_all and storage.

This server holds your API keys and can rent a GPU, so:

1. it listens on 127.0.0.1 only,
2. a request whose Host header is not 127.0.0.1:<port> or localhost:<port> is
   refused (another site cannot reach it through DNS rebinding),
3. every /api/ request must carry the token made for this launch, which only
   the page served here knows (another site cannot send requests in its name),
4. files are read and written only inside the outputs, jobs/ and configs/
   folders (no absolute paths, no ..),
5. no answer contains a key: settings say only whether a key is set.

State lives in one App object. One long task runs at a time: a connection or
a render. GET /api/status never sends a request to the pod (it would keep the
pod awake): it reads this server's state and, at most every 10 seconds, the
RunPod API.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import secrets
import string
import sys
import threading
import time
import webbrowser
from collections import Counter, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from jena_image_render import (
    __version__,
    assets,
    config,
    env,
    local,
    models,
    runner,
    storage,
)
from jena_image_render.comfy import ComfyClient, ComfyError
from jena_image_render.engines import (
    ASPECTS,
    ENGINES,
    FIXED_CFG,
    SAMPLER_ALIASES,
    cfg_of,
)
from jena_image_render.folder import ComfyFolders, ModelsFolder
from jena_image_render.models import BASE_MODELS

LOGGER = logging.getLogger(__name__)
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TOKEN_HEADER = "X-Jenair-Token"
TOKEN_SLOT = "__JENAIR_TOKEN__"
MAX_BODY = 8 << 20
LOG_LINES = 200
ACCOUNT_SECONDS = 10.0  # RunPod API, never the pod
DEFAULT_MAX_HOURS = 3.0
JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,79}")
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp"}
# Suggestions for the page; any name ComfyUI knows is accepted.
COMFY_SAMPLERS = (
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms",
    "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
    "ddpm", "lcm", "ipndm", "deis", "res_multistep", "gradient_estimation",
    "er_sde", "uni_pc", "uni_pc_bh2", "ddim",
)
COMFY_SCHEDULERS = (
    "simple", "normal", "karras", "exponential", "sgm_uniform", "beta",
    "ddim_uniform", "linear_quadratic", "kl_optimal",
)
PHASES = ("renting", "pulling image", "starting ComfyUI", "ready")
# The keys the Settings screen may write to .env: secrets are never answered.
SECRET_KEYS = ("RUNPOD_API_KEY", "HF_TOKEN", "CIVITAI_TOKEN")
PLAIN_KEYS = ("JENA_STORAGE", "JENA_TARGET", "JENA_COMFY_URL", "JENA_POD_IMAGE",
              "JENA_MODELS_DIR")
# Files the file browser shows: weights a job can load.
BROWSE_SUFFIXES = {".safetensors", ".sft", ".ckpt", ".gguf"}
BROWSE_KINDS = ("loras", "checkpoints", "diffusion_models")


# Folders people keep next to their checkpoints that hold other kinds of models.
HELPER_FOLDERS = {
    "controlnet", "ipadapter", "ip-adapter", "image_encoder", "clip_vision",
    "upscale_models", "embeddings", "vae", "annotators", "insightface",
}


def _helper_file(rel: str) -> bool:
    """Not something to offer as a model or LoRA: a helper model or an old pickle."""
    parts = [part.lower() for part in PurePosixPath(rel).parts]
    if PurePosixPath(rel).suffix.lower() in {".bin", ".pt", ".pth"}:
        return True
    return any(part in HELPER_FOLDERS for part in parts[1:-1])


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def inside(base: Path, rel: str) -> Path:
    """`rel` inside `base`, or ApiError: no absolute path, no .., no way out."""
    if not isinstance(rel, str) or not rel or "\\" in rel or "\0" in rel:
        raise ApiError(400, "give a relative path with / separators")
    part = PurePosixPath(rel)
    if part.is_absolute() or ".." in part.parts or rel.startswith("~"):
        raise ApiError(403, f"{rel!r} is outside the allowed folders")
    root = base.resolve()
    path = (root / part).resolve()
    if path != root and root not in path.parents:
        raise ApiError(403, f"{rel!r} is outside the allowed folders")
    return path


class _Lines(logging.Handler):
    """The package's log lines, for the page's log area."""

    def __init__(self, sink: Callable[[str], None]) -> None:
        super().__init__(logging.INFO)
        self.sink = sink
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink(self.format(record))
        except Exception:  # a log line must never break a request
            self.handleError(record)


class _Warnings(logging.Handler):
    """Warnings logged by one thread (a check), not by a render running beside it."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.thread = threading.get_ident()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread:
            self.lines.append(record.getMessage())


def _default_pods() -> Any:
    from jena_image_render.pod import Pods, Settings
    from jena_image_render.runpod_api import RunPodAPI

    return Pods(RunPodAPI(env.get("RUNPOD_API_KEY")), Settings.load())


def _runpod_api(key: str) -> Any:
    from jena_image_render.runpod_api import RunPodAPI

    return RunPodAPI(key)


def _hf_api(token: str) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _hidden(text: str, *secrets_: str) -> str:
    """An error message with any secret blanked out, on one short line."""
    for secret in secrets_:
        if secret:
            text = text.replace(secret, "***")
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:300]


# macOS mounts its own service volumes under /Volumes; nobody keeps models there.
SYSTEM_VOLUMES = {"Recovery", "Preboot", "VM", "Update", "xarts", "iSCPreboot", "Hardware"}


def _places() -> list[Path]:
    """Mounted volumes and drives."""
    if sys.platform == "win32":
        return [Path(f"{letter}:\\") for letter in string.ascii_uppercase
                if Path(f"{letter}:\\").exists()]
    roots = [Path("/Volumes")] if sys.platform == "darwin" else [Path("/mnt"),
                                                                    Path("/media")]
    found = []
    for root in roots:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if entry.name in SYSTEM_VOLUMES:
                continue
            if root.name == "media" and not os.path.ismount(entry):
                try:  # /media/<user>/<volume>
                    found += [e for e in sorted(entry.iterdir())
                              if e.is_dir() and not e.name.startswith(".")]
                except OSError:
                    pass
                continue
            found.append(entry)
    return found


def _idle_minutes() -> int:
    from jena_image_render.pod import Settings

    try:
        return int(Settings.load().idle_minutes)
    except (OSError, ValueError):
        return int(Settings().idle_minutes)


def _epoch(iso: str) -> float | None:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


def _idle() -> dict[str, Any]:
    return {
        "connection": {"state": "idle", "message": "", "detail": {}},
        "pod": {"id": "", "gpu": "", "phase": "", "elapsed": 0, "typical_seconds": 0,
                "attempt": 0, "cloud": "", "idle_deadline": None,
                "hard_deadline": None, "balance": None, "spend_per_hour": None},
    }


def _files_idle() -> dict[str, Any]:
    return {"state": "idle", "items": [], "bytes": 0, "total": 0}


def _render_idle() -> dict[str, Any]:
    return {"state": "idle", "total": 0, "done": 0, "failed": 0, "skipped": 0,
            "current": None, "started": None, "eta_seconds": None, "results": [],
            "message": ""}


class App:
    """Everything the server knows, behind one lock."""

    def __init__(
        self,
        root: Path = Path("."),
        outputs: Path = Path("outputs"),
        comfy: Callable[[str], Any] = ComfyClient,
        pods: Callable[[], Any] = _default_pods,
        clock: Callable[[], float] = time.time,
        env_file: Path | None = None,
        runpod_api: Callable[[str], Any] | None = None,
        hf_api: Callable[[str], Any] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.env_file = env_file if env_file is not None else Path.cwd() / ".env"
        self.make_runpod_api = runpod_api or _runpod_api
        self.make_hf_api = hf_api or _hf_api
        self.link = local.link  # symbolic link, else hard link, else a copy
        # Keys a real environment variable sets (it wins over .env).
        written = env.read_dotenv(self.env_file)
        self.from_environment = {
            key for key in (*SECRET_KEYS, *PLAIN_KEYS)
            if os.environ.get(key) is not None and os.environ[key] != written.get(key)
        }
        self.outputs = outputs if outputs.is_absolute() else self.root / outputs
        self.jobs_dir = self.root / "jobs"
        self.configs_dir = self.root / "configs"
        self.token = secrets.token_urlsafe(32)
        self.make_comfy = comfy
        self.make_pods = pods
        self.clock = clock
        self.comfy_url = env.get("JENA_COMFY_URL", env.DEFAULT_COMFY_URL)
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.task: threading.Thread | None = None
        self.stopper: threading.Thread | None = None
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self.target = ""
        self.state = _idle()
        self.files = _files_idle()
        self.render = _render_idle()
        self.client: Any = None  # the connected ComfyUI (local or the pod's)
        self.folders: ComfyFolders | None = None
        self.pods: Any = None
        self.session: dict[str, Any] | None = None
        self.max_hours = DEFAULT_MAX_HOURS
        self.idle_minutes = _idle_minutes()
        self.last_request: float | None = None  # our last request to the pod
        self.account_at = 0.0
        self.refreshing = False
        self.lookup: storage.Lookup | None = None
        self.handler = _Lines(self._log)
        logging.getLogger("jena_image_render").addHandler(self.handler)

    def close(self) -> None:
        logging.getLogger("jena_image_render").removeHandler(self.handler)

    def _log(self, line: str) -> None:
        with self.lock:
            self.log.append(line)

    # ── static information ───────────────────────────────────────────
    def meta(self) -> dict[str, Any]:
        engines = [
            {
                "name": e.name,
                "steps": e.steps,
                "cfg": cfg_of(e, e.guidance),
                "cfg_fixed": e.name in FIXED_CFG,
                "negative_needs_cfg": e.name == "krea2",
                "sampler": e.sampler,
                "scheduler": e.scheduler,
                "width": e.width,
                "height": e.height,
                "multiple": e.multiple,
                "uses_negative": e.uses_negative,
                "default_negative": e.default_negative,
                "files": list(e.files),
                "notes": e.notes,
            }
            for e in ENGINES.values()
        ]
        target = env.get("JENA_TARGET")
        return {
            "version": __version__,
            "engines": engines,
            "aspects": {k: list(v) for k, v in ASPECTS.items()},
            "samplers": [*COMFY_SAMPLERS, *SAMPLER_ALIASES],
            "sampler_aliases": {k: list(v) for k, v in SAMPLER_ALIASES.items()},
            "schedulers": list(COMFY_SCHEDULERS),
            "settings": {
                "runpod_key": bool(env.get("RUNPOD_API_KEY")),
                "hf_token": bool(env.get("HF_TOKEN")),
                "storage": storage.configured(),
                "default_target": target if target in config.TARGETS else "local",
                "comfy_url": self.comfy_url,
                "pod_image": env.get("JENA_POD_IMAGE", env.DEFAULT_POD_IMAGE),
                "idle_minutes": self.idle_minutes,
            },
        }

    # ── names to choose from ─────────────────────────────────────────
    def _storage(self) -> storage.Lookup | None:
        if not storage.configured():
            return None
        if self.lookup is None:
            self.lookup = storage.Lookup()
        return self.lookup

    def _local_folders(self) -> ComfyFolders | None:
        """The local ComfyUI's folders: from the connection, else a quick look."""
        with self.lock:
            if self.target == "local" and self.state["connection"]["state"] == "ready":
                return self.folders
            url = self.comfy_url
        client = self.make_comfy(url)
        if not client.ready(local.QUICK_TIMEOUT):
            return None
        return local.discover(client, url, local.QUICK_TIMEOUT)

    def choices(self, target: str, refresh: bool = False) -> dict[str, Any]:
        if target not in config.TARGETS:
            raise ApiError(400, 'target must be "local" or "runpod"')
        if refresh:
            self.lookup = None
        folder = ModelsFolder()
        comfy = self._local_folders() if target == "local" else None
        lookup = self._storage()
        registry = models.load(folder, lookup, comfy)
        found: dict[str, list[dict[str, Any]]] = {"models": [], "loras": []}
        names: dict[str, set[str]] = {"models": set(), "loras": set()}

        def add(kind: str, item: dict[str, Any]) -> None:
            if item["name"] not in names[kind]:
                names[kind].add(item["name"])
                found[kind].append(item)

        for alias, model in registry.models.items():
            note = ENGINES[model.engine].notes.split(". ")[0]
            add("models", {"name": alias, "engine": model.engine, "source": "built-in",
                           "note": note})
        for alias, lora in registry.loras.items():
            add("loras", {"name": alias, "engine": lora.engine,
                          "trigger": lora.trigger, "source": "built-in"})
        places: list[tuple[str, Any]] = [("models folder", folder)]
        if comfy is not None:
            places.append(("ComfyUI", comfy))
        listed = [(source, top, place.files(top))
                  for source, place in places
                  for top in ("checkpoints", "diffusion_models", "loras")]
        if lookup is not None:
            for top in ("checkpoints", "diffusion_models", "loras"):
                listed.append(("storage", top, [s.dest for s in lookup(top)]))
        for source, top, rels in listed:
            for rel in rels:
                if _helper_file(rel):
                    continue
                if top == "loras":
                    lora = self._lora_name(registry, rel)
                    if lora is not None:
                        add("loras", {"name": lora.alias, "engine": lora.engine,
                                      "trigger": lora.trigger, "source": source})
                    continue
                model = self._model_name(registry, rel)
                if model is not None:
                    add("models", {"name": model.alias, "engine": model.engine,
                                   "source": source, "note": rel})
        return found

    @staticmethod
    def _candidates(rel: str) -> list[str]:
        inner = PurePosixPath(rel.split("/", 1)[1])
        options = [inner.stem, inner.name, str(inner.with_suffix("")), str(inner)]
        return list(dict.fromkeys(options))

    def _model_name(self, registry: models.Registry, rel: str) -> models.Model | None:
        """The name a job can write for this file (resolved back to it), or None."""
        role = "checkpoint" if rel.startswith("checkpoints/") else "diffusion"
        for name in self._candidates(rel):
            if name in registry.models:
                continue
            try:
                model = registry.model(name)
            except ValueError:
                continue
            if model.files.get(role) and model.files[role].dest == rel:
                return model
        return None

    def _lora_name(self, registry: models.Registry, rel: str) -> models.Lora | None:
        for name in self._candidates(rel):
            if name in registry.loras:
                continue
            try:
                lora = registry.lora(name)
            except ValueError:
                continue
            if lora.file.dest == rel:
                return lora
        return None

    # ── check ────────────────────────────────────────────────────────
    def _target(self, body: dict[str, Any], project: Any, required: bool = False) -> str:
        value = body.get("target")
        if not value and required:
            raise ApiError(400, 'say where to render: "target": "local" or "runpod"')
        if not value:
            try:
                value = config.target(project, "project")
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc
            value = value or self.meta()["settings"]["default_target"]
        if value not in config.TARGETS:
            raise ApiError(400, 'target must be "local" or "runpod"')
        return str(value)

    def _plan(
        self, project: Any, target: str, overwrite: bool, output: str
    ) -> dict[str, Any]:
        client, folders, answering = None, None, False
        if target == "local":
            with self.lock:
                if self.target == "local" and self.state["connection"]["state"] == "ready":
                    client, folders, answering = self.client, self.folders, True
                url = self.comfy_url
            if client is None:
                client = self.make_comfy(url)
                answering = client.ready(local.QUICK_TIMEOUT)
                if answering:
                    folders = local.discover(client, url, local.QUICK_TIMEOUT)
        lookup = self._storage()
        registry = models.load(remote=lookup, comfy=folders)
        renders = config.expand(project, registry, "project", default_output=output)
        todo, skipped = runner.select(renders, self.outputs, overwrite=overwrite)
        files = assets.needed(todo, registry)
        if answering:
            names = assets.LocalNames(client)
            steps, _ = local.plan(files, names, folders, True)
        else:
            steps = assets.plan(files, target, lookup is not None)
        return {"renders": renders, "todo": todo, "skipped": skipped, "files": files,
                "steps": steps, "fillable": folders is not None, "answering": answering}

    def check(self, body: dict[str, Any]) -> dict[str, Any]:
        project = body.get("project")
        target = self._target(body, project)
        output = str(body.get("name") or "ui")
        warnings = _Warnings()
        logger = logging.getLogger("jena_image_render")
        logger.addHandler(warnings)
        result: dict[str, Any] = {"ok": False, "target": target, "errors": [],
                                  "warnings": warnings.lines, "images": 0,
                                  "skipped": 0, "per_batch": [], "files": []}
        try:
            plan = self._plan(project, target, bool(body.get("overwrite")), output)
        except (ValueError, OSError, RuntimeError, ComfyError) as exc:
            result["errors"].append(str(exc))
            return result
        finally:
            logger.removeHandler(warnings)
        todo, renders = plan["todo"], plan["renders"]
        saved = Counter(r.batch for r in renders) - Counter(r.batch for r in todo)
        images = Counter(r.batch for r in todo)
        order = list(dict.fromkeys(r.batch for r in renders))
        result["per_batch"] = [{"name": b, "images": images[b], "saved": saved[b]}
                               for b in order]
        result["files"] = [{"how": how, "source": spec.source, "dest": spec.dest,
                            "size": spec.size} for how, spec in plan["steps"]]
        for how, spec in plan["steps"]:
            if how != "missing":
                continue
            if target == "runpod":
                why = f"not found on this computer: {spec.local}"
            elif not plan["fillable"]:
                why = "ComfyUI does not list it, and it cannot be filled from here"
            else:
                why = "ComfyUI does not list it, and it is not on this computer"
            result["errors"].append(f"missing models/{spec.dest}: {why}")
        result.update(ok=not result["errors"], images=len(todo),
                      skipped=plan["skipped"])
        return result

    # ── long tasks ───────────────────────────────────────────────────
    def busy(self) -> bool:
        with self.lock:
            return any(t is not None and t.is_alive() for t in (self.task, self.stopper))

    def _start(self, work: Callable[..., None], *args: Any) -> None:
        with self.lock:
            if self.busy():
                raise ApiError(409, "busy: wait for the current connection or render")
            self.cancel.clear()
            self.task = threading.Thread(target=self._guard, args=(work, *args),
                                         name="jenair-task", daemon=True)
            self.task.start()

    def _guard(self, work: Callable[..., None], *args: Any) -> None:
        try:
            work(*args)
        except Exception as exc:  # a thread has no caller: show it on the page
            LOGGER.error("%s", exc)
            with self.lock:
                if self.state["connection"]["state"] == "connecting":
                    self.state["connection"].update(state="error", message=str(exc))
                    self.state["pod"]["phase"] = ""
                if self.render["state"] == "running":
                    self.render.update(state="error", message=str(exc), current=None)
                if self.files["state"] == "working":
                    self.files["state"] = "error"

    # ── connect ──────────────────────────────────────────────────────
    def connect(self, body: dict[str, Any]) -> dict[str, Any]:
        target = body.get("target")
        if target not in config.TARGETS:
            raise ApiError(400, 'target must be "local" or "runpod"')
        if target == "runpod" and not env.get("RUNPOD_API_KEY"):
            raise ApiError(400, "RUNPOD_API_KEY is not set: put it in .env and "
                                "start jenair ui again")
        url = str(body.get("comfy_url") or "").strip()
        if url and not url.startswith(("http://", "https://")):
            raise ApiError(400, "the ComfyUI address starts with http:// or https://")
        hours = self._hours(body)
        secure = bool(body.get("secure"))
        if target == "local" and url:
            with self.lock:
                self.comfy_url = url.rstrip("/")
        self._start(self._connect, target, hours, secure)
        return {"started": True}

    @staticmethod
    def _hours(body: dict[str, Any]) -> float:
        try:
            hours = float(body.get("max_hours") or DEFAULT_MAX_HOURS)
        except (TypeError, ValueError) as exc:
            raise ApiError(400, "max_hours must be a number") from exc
        if not 0 < hours <= 24:
            raise ApiError(400, "max_hours must be more than 0 and at most 24")
        return hours

    def _connect(self, target: str, hours: float, secure: bool) -> None:
        if target == "local":
            self._connect_local()
        else:
            self._connect_runpod(hours, secure)

    def _connecting(self, target: str, message: str) -> None:
        with self.lock:
            self.target = target
            self.state = _idle()
            self.state["connection"].update(state="connecting", message=message)
            self.client, self.folders, self.session = None, None, None

    def _connect_local(self) -> None:
        url = self.comfy_url
        self._connecting("local", f"Asking ComfyUI at {url}")
        client = self.make_comfy(url)
        if not client.ready(local.QUICK_TIMEOUT):
            raise RuntimeError(
                f"ComfyUI is not answering on {url} (status "
                f"{client.probe(local.QUICK_TIMEOUT)}). Start it (docs/local.md) or "
                "check the address."
            )
        stats = client.system_stats(local.QUICK_TIMEOUT) or {}
        folders = local.discover(client, url, 15)
        system = stats.get("system") or {}
        devices = stats.get("devices") or [{}]
        detail = {
            "url": url,
            "version": str(system.get("comfyui_version") or ""),
            "device": str((devices[0] or {}).get("name") or ""),
            "same_computer": folders is not None,
        }
        message = (
            "ready: missing files are downloaded or linked into it"
            if folders is not None
            else "ready: its files are used by name only"
        )
        with self.lock:
            self.client, self.folders = client, folders
            self.state["connection"].update(state="ready", message=message,
                                            detail=detail)

    def _touch(self) -> None:
        with self.lock:
            self.last_request = self.clock()

    def _connect_runpod(self, hours: float, secure: bool) -> None:
        from jena_image_render.pod import Cancelled

        self._connecting("runpod", "Looking for a running pod")
        pods = self.make_pods()
        with self.lock:
            self.pods, self.max_hours = pods, hours
            self.idle_minutes = int(pods.settings.idle_minutes)
        session = pods.live()
        if session is not None:
            LOGGER.info("Reusing pod %s", session["pod_id"])
            client = pods.client(session)
            if client.ready(local.QUICK_TIMEOUT):
                self._touch()
        else:
            try:
                session = pods.start(hours, secure, progress=self._on_pod,
                                     should_stop=self.cancel.is_set)
            except Cancelled:
                with self.lock:
                    self.state = _idle()
                    self.state["connection"]["message"] = "cancelled; the pod was terminated"
                return
            self._touch()
            client = pods.client(session)
        created = _epoch(str(session.get("created", "")))
        limit = float(session.get("max_hours") or hours)
        with self.lock:
            self.session, self.client = session, client
            pod = self.state["pod"]
            pod.update(id=session["pod_id"], gpu=session.get("gpu", ""), phase="ready",
                       hard_deadline=created + limit * 3600 if created else None)
            self.state["connection"].update(state="ready", message="ready",
                                            detail={"url": session.get("url", "")})
        self._refresh_account()

    def _on_pod(self, event: dict[str, Any]) -> None:
        with self.lock:
            pod = self.state["pod"]
            if event.get("cloud") == "secure" and pod["cloud"] == "community":
                self.state["connection"]["message"] = (
                    "Community did not start; trying Secure Cloud")
            pod.update(phase=event["phase"],
                       cloud=event.get("cloud", pod["cloud"]),
                       attempt=event.get("attempt", pod["attempt"]))
            if event["phase"] == "renting":
                pod.update(id="", gpu="", elapsed=0, hard_deadline=None)
                return
            created = _epoch(str(event.get("created", "")))
            pod.update(id=event.get("pod_id", ""), gpu=event.get("gpu", ""),
                       elapsed=round(float(event.get("elapsed", 0))),
                       typical_seconds=event.get("typical_seconds", 0),
                       hard_deadline=created + self.max_hours * 3600 if created else None)
            if event["phase"] != "ready":
                self.state["connection"]["message"] = (
                    f"{event['phase']} on {pod['gpu'] or 'a GPU'}")

    def _refresh_account(self) -> None:
        """Balance and spend, and whether our pod still exists (RunPod API)."""
        with self.lock:
            if self.refreshing or self.pods is None:
                return
            self.refreshing = True
            pods = self.pods
        try:
            account = pods.api.account()
        except Exception as exc:  # the RunPod API can fail in many ways
            LOGGER.warning("RunPod account not read: %s", exc)
            account = None
        with self.lock:
            self.refreshing = False
            self.account_at = self.clock()
            if account is None:
                return
            self.state["pod"].update(balance=account.get("clientBalance"),
                                     spend_per_hour=account.get("currentSpendPerHr"))
            ids = {p.get("id") for p in account.get("pods") or []}
            if (self.target == "runpod" and self.session is not None
                    and self.session["pod_id"] not in ids
                    and self.state["connection"]["state"] == "ready"):
                LOGGER.info("Pod %s is gone (idle shutdown, hard limit or stopped "
                            "elsewhere)", self.session["pod_id"])
                pods.sessions.forget()
                balance = self.state["pod"]["balance"], self.state["pod"]["spend_per_hour"]
                self.state = _idle()
                self.state["pod"].update(balance=balance[0], spend_per_hour=balance[1])
                self.state["connection"]["message"] = "the pod is gone (it stopped itself)"
                self.session, self.client = None, None

    def disconnect(self) -> dict[str, Any]:
        with self.lock:
            target = self.target
            stopping = self.stopper is not None and self.stopper.is_alive()
        self.cancel.set()
        if target != "runpod":
            with self.lock:
                if not (self.task is not None and self.task.is_alive()):
                    self.state, self.client, self.folders = _idle(), None, None
            return {"stopping": False}
        if not stopping:
            with self.lock:
                self.state["pod"]["phase"] = "stopping"
                self.stopper = threading.Thread(target=self._stop_pod,
                                                name="jenair-stop", daemon=True)
                self.stopper.start()
        return {"stopping": True}

    def _stop_pod(self) -> None:
        try:
            pods = self.pods or self.make_pods()
            stopped = pods.stop()
        except Exception as exc:  # shown on the page; the console is the fallback
            LOGGER.error("Stopping the pod failed: %s — check the RunPod console", exc)
            stopped = False
        with self.lock:
            balance = self.state["pod"]["balance"]
            self.state = _idle()
            self.state["pod"]["balance"] = balance
            self.state["connection"]["message"] = (
                "pod stopped" if stopped else "stopping failed: check the RunPod console")
            if not stopped:
                self.state["connection"]["state"] = "error"
            self.session, self.client = None, None

    # ── settings (.env) ──────────────────────────────────────────────
    def settings(self) -> dict[str, Any]:
        return {
            "file": self.env_file.name,
            "secrets": {key: {"set": bool(env.get(key)),
                              "from_environment": key in self.from_environment}
                        for key in SECRET_KEYS},
            "values": {key: {"value": env.get(key),
                             "from_environment": key in self.from_environment}
                       for key in PLAIN_KEYS},
            "defaults": {"JENA_COMFY_URL": env.DEFAULT_COMFY_URL,
                         "JENA_POD_IMAGE": env.DEFAULT_POD_IMAGE,
                         "JENA_MODELS_DIR": "models", "JENA_TARGET": "local"},
        }

    def save_settings(self, body: dict[str, Any]) -> dict[str, Any]:
        values = body.get("values")
        if not isinstance(values, dict):
            raise ApiError(400, '"values" must be an object of settings')
        known = (*SECRET_KEYS, *PLAIN_KEYS)
        clean: dict[str, str] = {}
        for key, value in values.items():
            if key not in known:
                raise ApiError(400, f"{key!r} is not a setting this page can change "
                                    f"(known: {', '.join(known)})")
            if value is not None and not isinstance(value, str):
                raise ApiError(400, f"{key}: give text, or null to clear it")
            text = (value or "").strip()
            if "\n" in text or "\r" in text or "\0" in text:
                raise ApiError(400, f"{key}: the value may not contain a line break")
            if key == "JENA_TARGET" and text not in ("", *config.TARGETS):
                raise ApiError(400, 'JENA_TARGET must be "local" or "runpod"')
            if key == "JENA_STORAGE" and text and not text.startswith("hf://"):
                raise ApiError(400, "JENA_STORAGE is hf://owner/repo")
            if key == "JENA_COMFY_URL" and text and not text.startswith(
                    ("http://", "https://")):
                raise ApiError(400, "the ComfyUI address starts with http:// or https://")
            clean[key] = text
        if not clean:
            return {**self.settings(), "saved": []}
        try:
            env.update_dotenv(self.env_file, clean)
        except OSError as exc:
            raise ApiError(500, f"could not write {self.env_file.name}: "
                                f"{exc.strerror or type(exc).__name__}") from exc
        for key, value in clean.items():
            if key in self.from_environment:
                continue  # a real environment variable wins, as at start-up
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        self._settings_changed(set(clean))
        LOGGER.info("Settings saved to %s: %s", self.env_file.name,
                    ", ".join(sorted(clean)))
        return {**self.settings(), "saved": sorted(clean)}

    def _settings_changed(self, keys: set[str]) -> None:
        """Drop what was built from the old values, so no restart is needed."""
        with self.lock:
            self.lookup = None  # storage and its HF token
            if keys & {"RUNPOD_API_KEY"}:
                self.pods = None
                if self.session is not None and env.get("RUNPOD_API_KEY"):
                    try:
                        self.pods = self.make_pods()
                    except (ValueError, OSError):
                        self.pods = None
                self.account_at = 0.0
            if "JENA_COMFY_URL" in keys and self.target != "local":
                self.comfy_url = env.get("JENA_COMFY_URL", env.DEFAULT_COMFY_URL)

    def test_settings(self, body: dict[str, Any]) -> dict[str, Any]:
        which = body.get("which")
        typed = body.get("value")
        if typed is not None and (not isinstance(typed, str) or "\n" in typed):
            raise ApiError(400, "value must be one line of text")
        if which == "runpod":
            key = (typed or "").strip() or env.get("RUNPOD_API_KEY")
            if not key:
                return {"ok": False, "message": "no RunPod API key is set"}
            try:
                account = self.make_runpod_api(key).account()
            except Exception as exc:  # whatever went wrong, without the key
                return {"ok": False, "message": _hidden(str(exc), key)
                        or type(exc).__name__}
            balance = account.get("clientBalance")
            text = f"${float(balance):.2f}" if balance is not None else "unknown"
            return {"ok": True, "message": f"RunPod account balance: {text}"}
        if which == "hf":
            token = (typed or "").strip() or env.get("HF_TOKEN")
            if not token:
                return {"ok": False, "message": "no Hugging Face token is set"}
            try:
                info = self.make_hf_api(token).whoami()
            except Exception as exc:  # whatever went wrong, without the token
                return {"ok": False, "message": _hidden(str(exc), token)
                        or type(exc).__name__}
            name = str((info or {}).get("name") or "?")
            role = ((info or {}).get("auth") or {}).get("accessToken", {}).get("role")
            return {"ok": True, "message": f"Hugging Face account: {name}"
                                           + (f" ({role} token)" if role else "")}
        raise ApiError(400, 'which is "runpod" or "hf"')

    # ── file browser and import ──────────────────────────────────────
    def _kind(self, kind: str) -> str:
        if kind not in BROWSE_KINDS:
            raise ApiError(400, f"kind is one of {', '.join(BROWSE_KINDS)}")
        return kind

    def _browse_places(self, kind: str) -> list[dict[str, str]]:
        places = [("Home", Path.home())]
        folder = ModelsFolder()
        own = (folder.root / kind).absolute()
        if own.is_dir():
            places.append((f"models/{kind}", own))
        else:  # the sub-folder is made on the first import
            places.append((folder.root.as_posix(), folder.root.absolute()))
        with self.lock:
            folders = self.folders if self.target == "local" else None
        if folders is not None:
            for path in folders.paths.get(kind, []):
                places.append((f"ComfyUI {path.name}", path))
        places += [(path.name or str(path), path) for path in _places()]
        seen, out = set(), []
        for name, path in places:
            if str(path) not in seen and path.is_dir():
                seen.add(str(path))
                out.append({"name": name, "path": str(path)})
        return out

    def browse(self, raw: str, kind: str) -> dict[str, Any]:
        places = self._browse_places(self._kind(kind))
        if not raw:
            if not places:
                raise ApiError(404, "no folder to start from")
            raw = places[0]["path"]
        path = Path(raw).expanduser()
        if not path.is_absolute() or "\0" in raw:
            raise ApiError(400, "give an absolute folder path")
        path = Path(os.path.normpath(path))
        parent = str(path.parent) if path.parent != path else ""
        result: dict[str, Any] = {"ok": True, "message": "", "path": str(path),
                                  "parent": parent, "places": places,
                                  "dirs": [], "files": []}
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name.lower())
        except FileNotFoundError:
            return {**result, "ok": False, "message": f"{path} does not exist"}
        except NotADirectoryError:
            return {**result, "ok": False, "message": f"{path} is not a folder"}
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            return {**result, "ok": False, "message": f"{path} cannot be read: {reason}"}
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    result["dirs"].append(entry.name)
                elif (entry.is_file()
                      and PurePosixPath(entry.name).suffix.lower() in BROWSE_SUFFIXES):
                    result["files"].append({"name": entry.name,
                                            "size": entry.stat().st_size})
            except OSError:
                continue
        return result

    def _registry_now(self) -> models.Registry:
        with self.lock:
            folders = self.folders if self.target == "local" else None
        return models.load(ModelsFolder(), None, folders)

    def _name_for(self, rel: str, kind: str) -> tuple[str, str]:
        """The name a job writes for models/<rel>, checked through the registry."""
        registry = self._registry_now()
        if kind == "loras":
            lora = self._lora_name(registry, rel)
            if lora is not None:
                return lora.alias, lora.engine
        else:
            model = self._model_name(registry, rel)
            if model is not None:
                return model.alias, model.engine
        raise ApiError(409, f"{rel} is in place, but its name also matches another "
                            "file: rename one of them")

    def import_file(self, body: dict[str, Any]) -> dict[str, Any]:
        kind = self._kind(str(body.get("kind") or ""))
        raw = body.get("path")
        if not isinstance(raw, str) or not raw or "\0" in raw:
            raise ApiError(400, "give the file's absolute path")
        source = Path(raw).expanduser()
        if not source.is_absolute():
            raise ApiError(400, "give the file's absolute path")
        if PurePosixPath(source.name).suffix.lower() not in BROWSE_SUFFIXES:
            raise ApiError(400, "pick a weight file (.safetensors, .sft, .ckpt, .gguf)")
        if not source.is_file():
            raise ApiError(404, f"{source} is not a file")
        engine = str(body.get("engine") or "")
        if kind == "diffusion_models" and engine not in BASE_MODELS:
            raise ApiError(400, "a diffusion model needs its engine: "
                                + ", ".join(BASE_MODELS))
        if kind == "loras" and engine and engine not in ENGINES:
            raise ApiError(400, f"unknown engine {engine!r}")
        folder = ModelsFolder()
        root = folder.root.absolute()
        # Already where the name rules look: use it as it is.
        bases = [(root / kind, kind)]
        with self.lock:
            folders = self.folders if self.target == "local" else None
        if folders is not None:
            bases += [(base, kind) for base in folders.paths.get(kind, [])]
        for base, top in bases:
            for candidate in (source.absolute(), source.resolve()):
                for where in (base.absolute(), base.resolve()):
                    if where in candidate.parents:
                        rel = f"{top}/{candidate.relative_to(where).as_posix()}"
                        name, found = self._name_for(rel, kind)
                        return {"name": name, "engine": found, "how": "already there",
                                "dest": f"{folder.root.as_posix()}/{rel}"}
        if kind == "loras":
            rel = f"loras/{engine}/{source.name}" if engine else f"loras/{source.name}"
        elif kind == "checkpoints":
            rel = f"checkpoints/{source.name}"
        else:
            rel = f"diffusion_models/{engine}/{source.name}"
        dest = root / rel
        if dest.exists() or dest.is_symlink():
            same = dest.exists() and (
                dest.resolve() == source.resolve()
                or dest.stat().st_size == source.stat().st_size)
            if same:
                how = "already there"
            elif not body.get("replace"):
                raise ApiError(409, f"{folder.root.as_posix()}/{rel} already exists and "
                                    "is a different file: replace it, or rename yours")
            else:
                dest.unlink()
                how = self.link(source, dest)
        else:
            try:
                how = self.link(source, dest)
            except OSError as exc:
                raise ApiError(500, f"could not put the file in {folder.root.as_posix()}"
                                    f"/{rel}: {exc.strerror or exc}") from exc
        if how == "there":
            how = "already there"
        with self.lock:
            self.lookup = None
        name, found = self._name_for(rel, kind)
        LOGGER.info("%s %s → %s/%s (job name %r)", how, source, folder.root.as_posix(),
                    rel, name)
        return {"name": name, "engine": found, "how": how,
                "dest": f"{folder.root.as_posix()}/{rel}"}

    # ── render ───────────────────────────────────────────────────────
    def start_render(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.busy():
            raise ApiError(409, "busy: wait for the current connection or render")
        project = body.get("project")
        target = self._target(body, project, required=True)
        if target == "runpod" and not env.get("RUNPOD_API_KEY"):
            raise ApiError(400, "RUNPOD_API_KEY is not set: put it in .env and "
                                "start jenair ui again")
        result = self.check({**body, "target": target})
        if not result["ok"]:
            return {**result, "started": False}
        if not result["images"]:
            return {**result, "started": False,
                    "message": "nothing to render: every image is already saved"}
        options = {"overwrite": bool(body.get("overwrite")),
                   "keep": bool(body.get("continue_on_error")),
                   "output": str(body.get("name") or "ui"),
                   "hours": self._hours(body), "secure": bool(body.get("secure"))}
        with self.lock:
            self.files = _files_idle()
            self.render = _render_idle()
            self.render.update(state="running", started=self.clock(),
                               total=result["images"], skipped=result["skipped"])
        try:
            self._start(self._render, project, target, options)
        except ApiError:
            with self.lock:
                self.render = _render_idle()
            raise
        return {**result, "started": True}

    def _connected(self, target: str) -> bool:
        with self.lock:
            return self.target == target and self.state["connection"]["state"] == "ready"

    def _render(self, project: Any, target: str, options: dict[str, Any]) -> None:
        if not self._connected(target):
            self._connect(target, options["hours"], options["secure"])
            if self.cancel.is_set():
                with self.lock:
                    self.render.update(state="cancelled", current=None)
                return
            if not self._connected(target):
                raise RuntimeError("not connected")
        with self.lock:
            client, folders = self.client, self.folders
        lookup = self._storage()
        registry = models.load(remote=lookup, comfy=folders)
        renders = config.expand(project, registry, "project",
                                default_output=options["output"])
        todo, skipped = runner.select(renders, self.outputs,
                                      overwrite=options["overwrite"])
        with self.lock:
            self.render.update(total=len(todo), skipped=skipped)
        files = assets.needed(todo, registry)
        if target == "local":
            names = assets.LocalNames(client, filled=folders is not None)
            steps, places = local.plan(files, names, folders, True)
            self._files_begin(steps)
            if any(how in ("link", "download") for how, _ in steps):
                local.fill(steps, places, names, assets.Hashes(), env.get("HF_TOKEN"),
                           progress=self._on_file)
            for spec in files:
                names.find(spec)

            def name_of(r: config.Render) -> Any:
                return names.names(r, registry)

            where = self.comfy_url
        else:
            prepared = assets.prepare_uploads(files, assets.Hashes())
            if lookup is not None:
                prepared = storage.sync_all(lookup.store(), prepared)
            self._files_begin([("fetch" if f.url else "upload", f) for f in prepared])
            if not self.cancel.is_set():
                self._touch()
                assets.deliver(client, prepared, progress=self._on_file)

            def name_of(r: config.Render) -> Any:
                return assets.pod_names(r, registry)

            where = (self.session or {}).get("gpu", "runpod")
        with self.lock:
            if self.files["state"] == "working":
                self.files["state"] = "done"
        summary = runner.render_all(
            client, todo, name_of, self.outputs, options["keep"], where,
            progress=self._on_render, should_stop=self.cancel.is_set,
        )
        LOGGER.info("%s saved, %s failed, %s already saved before",
                    summary.done, summary.failed, skipped)
        with self.lock:
            cancelled = summary.cancelled or self.cancel.is_set()
            self.render.update(state="cancelled" if cancelled else "done",
                               current=None, eta_seconds=None)

    def _files_begin(self, steps: list[tuple[str, models.FileSpec]]) -> None:
        items = [{"dest": spec.dest, "how": how, "bytes": 0, "total": spec.size,
                  "status": "done" if how == "have" else "waiting"}
                 for how, spec in steps]
        with self.lock:
            working = any(i["status"] != "done" for i in items)
            self.files = {"state": "working" if working else "done", "items": items,
                          "bytes": 0, "total": sum(i["total"] for i in items)}

    def _on_file(self, event: dict[str, Any]) -> None:
        if self.target == "runpod":
            self._touch()
        with self.lock:
            items = self.files["items"]
            item = next((i for i in items if i["dest"] == event["dest"]), None)
            if item is None:
                item = {"dest": event["dest"]}
                items.append(item)
            item.update(how=event["how"], bytes=event["bytes"],
                        total=event["total"] or item.get("total", 0),
                        status=event["status"])
            self.files["bytes"] = sum(int(i.get("bytes") or 0) for i in items)
            self.files["total"] = sum(int(i.get("total") or 0) for i in items)

    def _on_render(self, event: dict[str, Any]) -> None:
        if self.target == "runpod":
            self._touch()
        r: config.Render = event["render"]
        with self.lock:
            render = self.render
            kind = event["event"]
            if kind in ("start", "step"):
                render["current"] = {
                    "label": f"{r.batch + '/' if r.batch else ''}{r.model}/{r.name}"
                             f" · seed {r.seed}",
                    "batch": r.batch, "model": r.model, "name": r.name, "seed": r.seed,
                    "step": event.get("step", 0), "steps": event.get("steps", r.steps),
                    "index": event["index"],
                }
                return
            result = {"batch": r.batch, "model": r.model, "name": r.name,
                      "seed": r.seed, "path": r.output, "seconds": 0.0,
                      "ok": kind == "done", "error": event.get("error", "")}
            if kind == "done":
                render["done"] += 1
                result["seconds"] = event["seconds"]
            else:
                render["failed"] += 1
            render["results"].append(result)
            times = [x["seconds"] for x in render["results"] if x["ok"]]
            left = render["total"] - render["done"] - render["failed"]
            render["eta_seconds"] = (round(sum(times) / len(times) * left)
                                     if times else None)

    def cancel_task(self) -> dict[str, Any]:
        self.cancel.set()
        with self.lock:
            client = self.client if self.render["state"] == "running" else None
        if client is not None:
            threading.Thread(target=self._interrupt, args=(client,),
                             name="jenair-interrupt", daemon=True).start()
        return {"cancelling": self.busy()}

    @staticmethod
    def _interrupt(client: Any) -> None:
        try:
            client.interrupt()
        except Exception as exc:  # the image still stops at the next check
            LOGGER.warning("ComfyUI interrupt failed: %s", exc)

    # ── status ───────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self.lock:
            stale = self.clock() - self.account_at >= ACCOUNT_SECONDS
            if (self.target == "runpod" and self.pods is not None and stale
                    and not self.refreshing
                    and self.state["connection"]["state"] == "ready"
                    and not (self.stopper is not None and self.stopper.is_alive())):
                threading.Thread(target=self._refresh_account, name="jenair-account",
                                 daemon=True).start()
            pod = dict(self.state["pod"])
            if self.target == "runpod" and self.last_request is not None and pod["id"]:
                pod["idle_deadline"] = self.last_request + self.idle_minutes * 60
            return json.loads(json.dumps({
                "target": self.target,
                "connection": self.state["connection"],
                "pod": pod,
                "files": self.files,
                "render": self.render,
                "log": list(self.log),
                "busy": self.busy(),
                "now": self.clock(),
            }, default=str))

    # ── files ────────────────────────────────────────────────────────
    def image(self, rel: str) -> tuple[bytes, str]:
        path = inside(self.outputs, rel)
        kind = IMAGE_TYPES.get(path.suffix.lower())
        if kind is None or not path.is_file():
            raise ApiError(404, "no such image")
        return path.read_bytes(), kind

    def list_files(self) -> dict[str, Any]:
        def under(folder: Path, pattern: str) -> list[str]:
            if not folder.is_dir():
                return []
            return sorted(p.relative_to(self.root).as_posix()
                          for p in folder.glob(pattern) if p.is_file())

        return {"jobs": under(self.jobs_dir, "*.json"),
                "examples": under(self.configs_dir, "**/*.json")}

    def read_file(self, rel: str) -> dict[str, Any]:
        path = inside(self.root, rel)
        allowed = [self.jobs_dir.resolve(), self.configs_dir.resolve()]
        if not any(folder in path.parents for folder in allowed):
            raise ApiError(403, "only files in jobs/ and configs/ can be opened")
        if path.suffix != ".json" or not path.is_file():
            raise ApiError(404, f"no such job file: {rel}")
        try:
            data = json.loads(path.read_text("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"{rel} is not valid JSON: {exc}") from exc
        return {"path": rel, "name": path.stem, "project": data}

    def save_file(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name") or "").strip().removesuffix(".json")
        if not JOB_NAME.fullmatch(name):
            raise ApiError(400, "a job name is letters, digits, spaces, . _ and -, "
                                "starting with a letter or digit")
        project = body.get("project")
        if not isinstance(project, dict):
            raise ApiError(400, "project must be a JSON object")
        path = inside(self.jobs_dir, f"{name}.json")
        if path.exists() and not body.get("overwrite"):
            raise ApiError(409, f"jobs/{name}.json already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(project, indent=2, ensure_ascii=False) + "\n",
                           "utf-8")
        partial.replace(path)
        LOGGER.info("Saved jobs/%s.json", name)
        return {"path": f"jobs/{name}.json", "name": name}

    def page(self) -> bytes:
        text = resources.files("jena_image_render").joinpath("ui").joinpath("index.html")
        return text.read_text("utf-8").replace(TOKEN_SLOT, self.token).encode("utf-8")


def _query(query: dict[str, list[str]], key: str) -> str:
    return (query.get(key) or [""])[0]


ROUTES: dict[tuple[str, str], Callable[[App, Any], Any]] = {
    ("GET", "/api/meta"): lambda app, q: app.meta(),
    ("GET", "/api/choices"): lambda app, q: app.choices(
        _query(q, "target") or "local", _query(q, "refresh") == "1"),
    ("GET", "/api/status"): lambda app, q: app.status(),
    ("GET", "/api/image"): lambda app, q: app.image(_query(q, "path")),
    ("GET", "/api/files"): lambda app, q: app.list_files(),
    ("GET", "/api/file"): lambda app, q: app.read_file(_query(q, "path")),
    ("POST", "/api/file"): lambda app, b: app.save_file(b),
    ("POST", "/api/check"): lambda app, b: app.check(b),
    ("POST", "/api/render"): lambda app, b: app.start_render(b),
    ("POST", "/api/connect"): lambda app, b: app.connect(b),
    ("POST", "/api/disconnect"): lambda app, b: app.disconnect(),
    ("POST", "/api/cancel"): lambda app, b: app.cancel_task(),
    ("GET", "/api/settings"): lambda app, q: app.settings(),
    ("POST", "/api/settings"): lambda app, b: app.save_settings(b),
    ("POST", "/api/settings/test"): lambda app, b: app.test_settings(b),
    ("GET", "/api/browse"): lambda app, q: app.browse(
        _query(q, "path"), _query(q, "kind") or "loras"),
    ("POST", "/api/import"): lambda app, b: app.import_file(b),
}
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "jenair"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.debug("%s " + format, self.address_string(), *args)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def _send(self, status: int, data: bytes, kind: str, extra: dict[str, str]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, data, "application/json; charset=utf-8", {})

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        host = (self.headers.get("Host") or "").strip().lower()
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ApiError(400, "bad Content-Length") from exc
        if length > MAX_BODY:
            raise ApiError(413, "request too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"the request is not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ApiError(400, "the request must be a JSON object")
        return body

    def _handle(self, method: str) -> None:
        app: App = self.server.app  # type: ignore[attr-defined]
        try:
            if not self._host_ok():
                raise ApiError(403, "this server answers only to 127.0.0.1/localhost")
            url = urlsplit(self.path)
            if method == "GET" and url.path in ("/", "/index.html"):
                self._send(200, app.page(), "text/html; charset=utf-8", PAGE_HEADERS)
                return
            if not url.path.startswith("/api/"):
                raise ApiError(404, "not found")
            given = (self.headers.get(TOKEN_HEADER) or "").encode("utf-8")
            if not hmac.compare_digest(given, app.token.encode("utf-8")):
                raise ApiError(403, "missing or wrong token: reload the page")
            route = ROUTES.get((method, url.path))
            if route is None:
                raise ApiError(404, f"no {method} {url.path}")
            arg = self._body() if method == "POST" else parse_qs(url.query)
            result = route(app, arg)
            if isinstance(result, tuple):
                self._send(200, result[0], result[1], {})
            else:
                self._json(200, result)
        except ApiError as exc:
            self._json(exc.status, {"error": str(exc)})
        except Exception as exc:  # any failure is an answer, not a dropped connection
            LOGGER.error("%s %s: %s", method, self.path.split("?")[0], exc)
            self._json(500, {"error": str(exc)})


def make_server(app: App, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bound to 127.0.0.1 only; port 0 takes a free one."""
    httpd = ThreadingHTTPServer((HOST, port), Handler)
    httpd.daemon_threads = True
    httpd.app = app  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="jenair ui",
        description="A web page for job files, on this computer only "
                    "(http://127.0.0.1:PORT).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open the page in a browser")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    app = App(Path.cwd(), args.outputs)
    httpd = make_server(app, args.port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    LOGGER.info("jenair ui: %s (this computer only; Ctrl+C stops it)", url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping jenair ui")
    finally:
        httpd.server_close()
        app.close()
        if app.target == "runpod" and app.session is not None:
            LOGGER.info(
                "Pod %s stays up and stops itself after %s idle minutes "
                "(`jenair runpod down` stops it now)",
                app.session["pod_id"], app.idle_minutes,
            )
    return 0
