"""jenair: the jena-image-render command line.

    jenair jobs/cafe.json                your ComfyUI (default :8188); on this
                                         computer it also gets missing files
    jenair jobs/cafe.json --dry-run      check everything, render nothing
    jenair jobs/cafe.json --local URL    a ComfyUI on another machine
    jenair jobs/cafe.json --runpod       start or reuse a RunPod pod
    jenair jobs/cafe.json --runpod --dry-run   show the RunPod plan
    jenair pull zimage | JOB.json        download into ComfyUI or the models folder
    jenair push JOB.json                 upload local files to storage
    jenair runpod status | down          see or stop the pod
    jenair engines                       supported engines and sizes

Jobs render on your own ComfyUI unless --runpod, "target": "runpod" in the job
file or JENA_TARGET=runpod says otherwise. A GPU is never rented by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

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
from jena_image_render import pull as puller
from jena_image_render.comfy import ComfyClient, ComfyError
from jena_image_render.engines import ASPECTS, ENGINES, FIXED_CFG, cfg_of
from jena_image_render.folder import ComfyFolders, ModelsFolder

LOGGER = logging.getLogger("jena_image_render")
DEFAULT_TARGET = "local"  # renting a GPU is always asked for, never assumed


def _render_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jenair",
        description="Render prompt-only JSON batches with ComfyUI, locally or on RunPod.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("jobs", nargs="+", type=Path, help="job JSON files")
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--local",
        nargs="?",
        const="",
        metavar="URL",
        help=f"render on your ComfyUI (default {env.DEFAULT_COMFY_URL})",
    )
    where.add_argument("--runpod", action="store_true", help="render on a RunPod pod")
    parser.add_argument(
        "--dry-run", action="store_true", help="check and show the plan, render nothing"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="local: do not download missing files into ComfyUI (links still made)",
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--only", default="", help="only jobs whose name starts with this")
    parser.add_argument("--limit", type=int, default=0, help="render at most N images")
    parser.add_argument("--overwrite", action="store_true", help="re-render saved images")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--secure", action="store_true", help="RunPod: skip Community, use Secure Cloud")
    parser.add_argument("--max-hours", type=float, default=3.0, help="RunPod: hard pod lifetime")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _pods():
    from jena_image_render.pod import Pods, Settings
    from jena_image_render.runpod_api import RunPodAPI

    return Pods(RunPodAPI(env.get("RUNPOD_API_KEY")), Settings.load())


def cmd_runpod(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="jenair runpod")
    parser.add_argument("action", choices=["status", "down", "up"])
    parser.add_argument("--secure", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.0)
    args = parser.parse_args(argv)
    pods = _pods()
    if args.action == "status":
        print(json.dumps(pods.status(), indent=2))
        return 0
    if args.action == "down":
        return 0 if pods.stop() else 1
    if (session := pods.live()) is not None:
        LOGGER.info("Pod %s is already running: %s", session["pod_id"], session["url"])
        return 0
    pods.start(args.max_hours, args.secure)
    return 0


def cmd_engines() -> int:
    for eng in ENGINES.values():
        print(
            f"{eng.name:12} files: {', '.join(eng.files):32} "
            f"{eng.width}x{eng.height}, {eng.steps} steps, cfg "
            f"{cfg_of(eng, eng.guidance):g}{' (fixed)' if eng.name in FIXED_CFG else ''}, "
            f"{eng.sampler}/{eng.scheduler} — {eng.notes}"
        )
    print("aspects: " + ", ".join(f"{k}={w}x{h}" for k, (w, h) in ASPECTS.items()))
    return 0


def choose_target(args: argparse.Namespace, jobs: list[tuple[Path, Any]]) -> str:
    """Command line > the job files' "target" > JENA_TARGET > local."""
    if args.runpod:
        return "runpod"
    if args.local is not None:
        return "local"
    asked = {config.target(data, str(path)) for path, data in jobs} - {""}
    if len(asked) > 1:
        raise ValueError(
            "the job files ask for different targets (local and runpod); "
            "choose one with --local or --runpod"
        )
    if asked:
        return asked.pop()
    value = env.get("JENA_TARGET")
    if value and value not in config.TARGETS:
        raise ValueError(f'JENA_TARGET must be "local" or "runpod", not {value!r}')
    return value or DEFAULT_TARGET


def _registry(
    folder: ModelsFolder | None = None, comfy: ComfyFolders | None = None
) -> tuple[models.Registry, storage.Lookup | None]:
    """Catalog + models folder (+ a local ComfyUI's folders), and your storage
    when JENA_STORAGE is set."""
    lookup = storage.Lookup() if storage.configured() else None
    return models.load(folder, lookup, comfy), lookup


def _comfy_url(args: argparse.Namespace | None = None) -> str:
    given = args.local if args is not None else None
    return given or env.get("JENA_COMFY_URL", env.DEFAULT_COMFY_URL)


def _expand(
    jobs: list[tuple[Path, Any]], registry: models.Registry
) -> list[config.Render]:
    return [
        render
        for path, data in jobs
        for render in config.expand(data, registry, str(path), default_output=path.stem)
    ]


def cmd_render(args: argparse.Namespace) -> int:
    jobs = [(path, config.read(path)) for path in args.jobs]
    target = choose_target(args, jobs)
    url, client, folders, answering = "", None, None, False
    if target == "local":  # first: names may live in ComfyUI's own folders
        url = _comfy_url(args)
        timeout = local.QUICK_TIMEOUT if args.dry_run else 15
        client = ComfyClient(url)
        answering = client.ready(timeout)
        if answering:
            folders = local.discover(client, url, timeout)
        elif args.dry_run:
            LOGGER.info("ComfyUI is not answering on %s: planning without it", url)
        else:  # before the names: some may only be in ComfyUI's folders
            LOGGER.error(
                "ComfyUI is not answering on %s (status %s). Start it (docs/local.md), "
                "give its address with --local URL, or render on a rented GPU with "
                '--runpod / "target": "runpod" (docs/runpod.md).',
                url, client.probe(local.QUICK_TIMEOUT),
            )
            return 1
    registry, lookup = _registry(comfy=folders)
    renders = _expand(jobs, registry)
    todo, skipped = runner.select(
        renders, args.outputs, args.only, args.limit, args.overwrite
    )
    files = assets.needed(todo, registry)
    if args.dry_run:
        if client is not None and answering:
            names = assets.LocalNames(client)
            steps, places = local.plan(files, names, folders, not args.no_download)
            fillable = folders is not None
            print(runner.describe(todo, skipped, steps, target, places, fillable))
            missing = [f"models/{spec.dest}" for how, spec in steps if how == "missing"]
            title = "missing files (ComfyUI does not list them)"
        else:
            plan = assets.plan(files, target, bool(storage.configured()))
            print(runner.describe(todo, skipped, plan, target))
            missing = [spec.local for how, spec in plan if how == "missing"]
            title = "missing local files"
        if missing:
            print(f"{title}:\n  " + "\n  ".join(missing))
            return 1
        return 0
    if not todo:
        LOGGER.info("Nothing to render (%s already saved; --overwrite to redo)", skipped)
        return 0
    keep = args.continue_on_error

    if target == "local":
        assert client is not None
        names = assets.LocalNames(client, filled=folders is not None)
        steps, places = local.plan(files, names, folders, not args.no_download)
        if any(how in ("link", "download") for how, _ in steps):
            local.fill(steps, places, names, assets.Hashes(), env.get("HF_TOKEN"))
        for spec in files:  # fail before the first image, not halfway through
            names.find(spec)
        summary = runner.render_all(client, todo, lambda r: names.names(r, registry),
                                    args.outputs, keep, url)
    else:
        prepared = assets.prepare_uploads(files, assets.Hashes())
        if lookup is not None:  # before renting: the pod would idle meanwhile
            prepared = storage.sync_all(lookup.store(), prepared)
        pods = _pods()
        session = None
        try:
            session = pods.live() or pods.start(args.max_hours, args.secure)
            client = pods.client(session)
            assets.deliver(client, prepared)
            summary = runner.render_all(
                client, todo, lambda r: assets.pod_names(r, registry), args.outputs,
                keep, session.get("gpu", "runpod"),
            )
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted — terminating the pod (saved images are kept)")
            if session is not None:
                pods.stop()
            return 130
        LOGGER.info(
            "The pod stays up for your next run and stops itself after %s min idle "
            "(`jenair runpod down` stops it now)",
            pods.settings.idle_minutes,
        )
    LOGGER.info(
        "%s saved, %s failed, %s already saved before", summary.done, summary.failed, skipped
    )
    return 1 if summary.failed else 0


def _files_parser(prog: str, what: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"jenair {prog}")
    parser.add_argument("items", nargs="+", metavar=what)
    return parser


def _named(registry: models.Registry, name: str) -> list[models.FileSpec]:
    if name in registry.models:
        return list(registry.models[name].files.values())
    if name in registry.loras:
        return [registry.loras[name].file]
    raise models.ModelsError(
        f"{name!r} is not a model or LoRA alias (models: "
        f"{', '.join(sorted(registry.models))}; LoRAs: "
        f"{', '.join(sorted(registry.loras))})"
    )


def cmd_pull(argv: list[str]) -> int:
    """Download the URL files of aliases or job files into a ComfyUI on this
    computer, or else into the models folder."""
    args = _files_parser("pull", "NAME|JOB.json").parse_args(argv)
    folder = ModelsFolder()
    url = _comfy_url()
    client = ComfyClient(url)
    comfy = None
    if client.ready(local.QUICK_TIMEOUT):
        comfy = local.discover(client, url, local.QUICK_TIMEOUT)
    registry, _ = _registry(folder, comfy)
    files: dict[str, models.FileSpec] = {}
    for item in args.items:
        path = Path(item)
        if path.suffix == ".json" and path.is_file():
            specs = assets.needed(config.load(path, registry), registry)
        else:
            specs = _named(registry, item)
        for spec in specs:
            files.setdefault(spec.dest, spec)
    hf_token = env.get("HF_TOKEN")
    if comfy is None:
        count = puller.pull(list(files.values()), folder, assets.Hashes(), hf_token)
        LOGGER.info("%s file(s) downloaded into %s/", count, folder.root)
        return 0
    names = assets.LocalNames(client)
    todo = []
    for spec in files.values():
        if have := names.lookup(spec):
            LOGGER.info("ComfyUI already has %s/%s", spec.folder, have)
        else:
            todo.append(spec)
    count = puller.pull(todo, comfy, assets.Hashes(), hf_token)
    LOGGER.info("%s file(s) downloaded into ComfyUI's model folders", count)
    return 0


def cmd_push(argv: list[str]) -> int:
    """Upload the jobs' local-only files to JENA_STORAGE, without a pod."""
    args = _files_parser("push", "JOB.json").parse_args(argv)
    registry, lookup = _registry()
    renders = [r for item in args.items for r in config.load(Path(item), registry)]
    files = [f for f in assets.needed(renders, registry) if not f.url]
    if not files:
        LOGGER.info("Nothing to push: every file has a URL")
        return 0
    store = lookup.store() if lookup is not None else storage.open_storage()
    for spec in storage.sync_all(store, assets.prepare_uploads(files, assets.Hashes())):
        print(f"{spec.dest} → {spec.url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "huggingface_hub"):  # one line per request
        logging.getLogger(noisy).setLevel(logging.WARNING)
    env.load_dotenv()
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv[:1] == ["runpod"]:
            return cmd_runpod(argv[1:])
        if argv[:1] == ["engines"]:
            return cmd_engines()
        if argv[:1] == ["pull"]:
            return cmd_pull(argv[1:])
        if argv[:1] == ["push"]:
            return cmd_push(argv[1:])
        return cmd_render(_render_parser().parse_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except (OSError, ValueError, RuntimeError, ComfyError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
