"""Render a plan of images on a ComfyUI server and save them with their settings."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jena_image_render.comfy import ComfyClient, ComfyError
from jena_image_render.config import Render
from jena_image_render.graph import OUTPUT_NODE, Names, build_graph
from jena_image_render.models import FileSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class Summary:
    done: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: bool = False


# Render progress for a UI: {"event": "start" | "step" | "done" | "failed",
# "index", "total", "render", and "step"/"steps", "path"/"seconds" or "error"}.
Progress = Callable[[dict[str, Any]], None]


def select(
    renders: list[Render],
    outputs: Path,
    only: str = "",
    limit: int = 0,
    overwrite: bool = False,
) -> tuple[list[Render], int]:
    """Images still to render, and how many were skipped as already saved."""
    chosen = [r for r in renders if not only or r.name.startswith(only)]
    todo = [r for r in chosen if overwrite or not (outputs / r.output).is_file()]
    skipped = len(chosen) - len(todo)
    if limit:
        todo = todo[:limit]
    return todo, skipped


def describe(
    renders: list[Render],
    skipped: int,
    plan: list[tuple[str, FileSpec]],
    target: str = "",
    places: dict[str, Path] | None = None,
    fillable: bool = False,
) -> str:
    """The dry-run report: images, and how each file reaches the server.

    `places` comes with an answering ComfyUI: where downloads and links go
    (`fillable`: it is on this computer and can get files).
    """
    lines = [f"{len(renders)} image(s) to render" + (f", {skipped} already saved" if skipped else "")]
    for model, count in Counter(r.model for r in renders).items():
        lines.append(f"  {model}: {count}")
    if plan:
        names = {"local": "local ComfyUI", "runpod": "RunPod"}
        where = names.get(target, "RunPod; no target")
        lines.append(f"files ({where}):")
        if target == "local" and places is None:
            lines.append(
                "  (ComfyUI did not answer. A ComfyUI on this computer gets the files "
                "it lacks on the run — downloaded or linked; one elsewhere must "
                "have them)"
            )
        for how, spec in plan:
            if places is not None:
                lines.append(_local_step(how, spec, places, fillable))
                continue
            if how != "local":
                lines.append(f"  {how:7} {spec.source} → models/{spec.dest}")
                continue
            if spec.local and Path(spec.local).is_file():
                note = f"copy at {spec.local}: linked if missing"
            elif spec.url:
                note = "not in your models folder: downloaded if missing"
            else:
                note = "not in your models folder: ComfyUI must list it"
            lines.append(f"  {how:7} models/{spec.dest} ({note})")
    lines.append("outputs:")
    lines.extend(f"  {r.output}" for r in renders[:20])
    if len(renders) > 20:
        lines.append(f"  … and {len(renders) - 20} more")
    return "\n".join(lines)


def _local_step(
    how: str, spec: FileSpec, places: dict[str, Path], fillable: bool
) -> str:
    if how == "download":
        return f"  {how:8} {spec.url} → {places[spec.dest]}"
    if how == "link":
        return f"  {how:8} {spec.local} → {places[spec.dest]}"
    if how == "have":
        return f"  {how:8} models/{spec.dest} (ComfyUI lists it)"
    if not fillable:
        note = "this ComfyUI cannot be filled from here"
    elif spec.url:
        note = "--no-download is on"
    else:
        note = "not on this computer"
    return f"  {how:8} models/{spec.dest} ({note}; ComfyUI does not list it)"


def save(outputs: Path, render: Render, data: bytes, meta: dict[str, object]) -> Path:
    path = outputs / render.output
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(path)
    sidecar = {**render.to_json(), **meta}
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    return path


def render_all(
    client: ComfyClient,
    todo: list[Render],
    names: Callable[[Render], Names],
    outputs: Path,
    keep_going: bool = False,
    where: str = "",
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Summary:
    """Render one by one. `should_stop` is asked before each image (and when one
    fails, e.g. after ComfyUI's interrupt): True ends the run, cancelled."""
    summary = Summary()
    stop = should_stop or (lambda: False)

    def report(event: str, index: int, render: Render, **info: Any) -> None:
        if progress is not None:
            progress({"event": event, "index": index, "total": len(todo),
                      "render": render, **info})

    for index, render in enumerate(todo, 1):
        if stop():
            summary.cancelled = True
            break
        label = f"[{index}/{len(todo)}] {render.model} · {render.name} · seed {render.seed}"
        started = time.monotonic()
        report("start", index, render)

        def on_step(
            value: int, total: int, label: str = label, index: int = index,
            render: Render = render,
        ) -> None:
            if value == total or value % max(1, total // 4) == 0:
                LOGGER.info("%s step %s/%s", label, value, total)
            report("step", index, render, step=value, steps=total)

        try:
            graph = build_graph(render, names(render))
            data = client.run(graph, OUTPUT_NODE, on_step)
        except (ComfyError, LookupError, ValueError) as exc:
            if stop():
                summary.cancelled = True
                break
            summary.failed += 1
            LOGGER.error("%s FAILED: %s", label, exc)
            report("failed", index, render, error=str(exc))
            if not keep_going:
                break
            continue
        seconds = round(time.monotonic() - started, 1)
        path = save(outputs, render, data, {"seconds": seconds, "server": where})
        summary.done += 1
        LOGGER.info("%s done in %.1fs → %s", label, seconds, path)
        report("done", index, render, path=path, seconds=seconds)
    if summary.cancelled:
        LOGGER.warning(
            "Cancelled: %s of %s image(s) not rendered",
            len(todo) - summary.done - summary.failed, len(todo),
        )
    return summary
