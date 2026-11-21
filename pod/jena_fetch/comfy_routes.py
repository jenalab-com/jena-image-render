"""ComfyUI glue: routes, activity tracking and the idle self-termination thread."""

import asyncio
import logging
import os
import threading
from pathlib import Path

import folder_paths
import requests
from aiohttp import web
from server import PromptServer

from .fetcher import Fetcher, FetchItem
from .idle import IdleWatch, run_watch
from .upload import UploadError, append_chunk, is_verified

LOGGER = logging.getLogger("jena_fetch")
FETCHER = Fetcher(
    Path(folder_paths.models_dir),
    os.environ.get("CIVITAI_API_TOKEN", ""),
    hf_token=os.environ.get("HF_TOKEN", ""),
)
WATCH = IdleWatch(
    idle_seconds=float(os.environ.get("IDLE_SECONDS", "1200")),
    max_seconds=float(os.environ.get("MAX_SECONDS", "0")),
)
SERVER = PromptServer.instance
routes = SERVER.routes


@web.middleware
async def track_activity(request: web.Request, handler):
    WATCH.touch()
    return await handler(request)


SERVER.app.middlewares.append(track_activity)


@routes.post("/jena/fetch")
async def submit(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        items = [
            FetchItem(
                str(f["url"]),
                str(f["dest"]),
                str(f.get("sha256") or "").lower(),
                int(f.get("size") or 0),
            )
            for f in body["files"]
        ]
        FETCHER.submit(items)
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"accepted": len(items)})


@routes.get("/jena/fetch")
async def status(request: web.Request) -> web.Response:
    return web.json_response(FETCHER.status())


MODELS_DIR = Path(folder_paths.models_dir)


@routes.post("/jena/upload")
async def upload(request: web.Request) -> web.Response:
    """Append one raw-body chunk: ?dest=loras/..&sha256=..&offset=N&total=N."""
    query = request.query
    try:
        dest, sha256 = query["dest"], query["sha256"].lower()
        offset, total = int(query["offset"]), int(query["total"])
        data = await request.read()
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(
            None, append_chunk, MODELS_DIR, dest, sha256, offset, total, data
        )
    except UploadError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    except (KeyError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"status": status})


@routes.get("/jena/upload")
async def upload_status(request: web.Request) -> web.Response:
    try:
        verified = is_verified(MODELS_DIR, request.query["dest"], request.query["sha256"].lower())
    except (KeyError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"verified": verified})


def _queue_busy() -> bool:
    return bool(SERVER.prompt_queue.get_tasks_remaining() > 0)


def _terminate(reason: str) -> bool:
    """Ask RunPod to remove this pod; True only on a confirmed success."""
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    LOGGER.warning("jena_fetch: terminating pod %s (%s)", pod, reason)
    if not pod or not key:
        LOGGER.error("jena_fetch: RUNPOD_POD_ID or RUNPOD_API_KEY missing; cannot self-terminate")
        return False
    query = 'mutation { podTerminate(input: {podId: "%s"}) }' % pod
    response = requests.post(
        "https://api.runpod.io/graphql",
        json={"query": query},
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
    )
    errors = response.json().get("errors") if response.content else None
    if response.status_code >= 400 or errors:
        LOGGER.error("jena_fetch: terminate failed (%s): %s", response.status_code, errors)
        return False
    return True


threading.Thread(
    target=run_watch,
    args=(WATCH, lambda: _queue_busy() or FETCHER.busy(), _terminate),
    name="jena-idle-watch",
    daemon=True,
).start()
