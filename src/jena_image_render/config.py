"""Job JSON (schema v1) → the list of images to render.

A job file holds prompts. `model` and `seeds` may be lists: every combination
of prompt × model × seed becomes one image. A model or LoRA is a name, or the
address of its file (hf://... or https://...); a model may also be an object
naming its engine and files. Sampling settings fall back to the
engine defaults, so most jobs need nothing but a model and a prompt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jena_image_render.engines import (
    ASPECTS,
    FIXED_CFG,
    Engine,
    cfg_of,
    engine,
    guidance_of,
    sampler_pair,
)
from jena_image_render.models import Registry

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1

TOP_KEYS = {"version", "target", "output", "defaults", "jobs"}
TARGETS = ("local", "runpod")
JOB_KEYS = {
    "name",
    "model",
    "prompt",
    "negative",
    "seed",
    "seeds",
    "aspect",
    "width",
    "height",
    "loras",
    "steps",
    "cfg",
    "guidance",
    "sampler",
    "scheduler",
    "skip",
}
LATER = {"hires", "upscale", "face_detail", "soft", "source", "frames", "fps"}


class ConfigError(ValueError):
    """A job file cannot be turned into images."""


@dataclass(frozen=True)
class Render:
    """One image: everything the graph needs, before server names are known."""

    model: str  # the model's name: as written in the job, or given to an address
    engine: str
    name: str
    prompt: str
    negative: str
    seed: int
    width: int
    height: int
    steps: int
    guidance: float
    sampler: str
    scheduler: str
    loras: tuple[tuple[str, float], ...]  # (LoRA name or address, scale)
    output: str  # relative to the outputs folder, e.g. "cafe/zimage/cafe_1.png"
    source: str = ""  # job file and job index, for messages

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "engine": self.engine,
            "prompt": self.prompt,
            "negative": self.negative,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg": cfg_of(engine(self.engine), self.guidance),
            "guidance": self.guidance,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "loras": [{"name": n, "scale": s} for n, s in self.loras],
        }


def _comment(key: str) -> bool:
    """Keys starting with "_" (e.g. "_note") are comments: JSON has no other way."""
    return key.startswith("_")


def _as_list(value: Any, label: str) -> list[Any]:
    if isinstance(value, list):
        if not value:
            raise ConfigError(f"{label}: list must not be empty")
        return value
    return [value]


def _seeds(job: dict[str, Any], defaults: dict[str, Any], label: str) -> list[int]:
    """The job's seed(s) replace the defaults' seed(s) as a whole."""
    layer = job if ("seed" in job or "seeds" in job) else defaults
    if "seeds" in layer and "seed" in layer:
        raise ConfigError(f"{label}: use seed or seeds, not both")
    raw = layer.get("seeds", layer.get("seed", 0))
    try:
        return [int(s) for s in _as_list(raw, f"{label}.seeds")]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label}: seeds must be integers") from exc


def _layer_size(layer: dict[str, Any], label: str) -> tuple[int, int] | None:
    """Size given by one layer (job or defaults), or None to fall through."""
    has_wh = "width" in layer or "height" in layer
    if has_wh and "aspect" in layer:
        raise ConfigError(f"{label}: use aspect or width/height, not both")
    if "aspect" in layer:
        aspect = str(layer["aspect"])
        if aspect not in ASPECTS:
            raise ConfigError(
                f"{label}: aspect {aspect!r} is not one of {', '.join(ASPECTS)} "
                "(use width/height for other sizes)"
            )
        return ASPECTS[aspect]
    if has_wh:
        if "width" not in layer or "height" not in layer:
            raise ConfigError(f"{label}: give both width and height")
        return int(layer["width"]), int(layer["height"])
    return None


def _round(value: int, eng: Engine, label: str) -> int:
    rounded = max(eng.multiple, round(value / eng.multiple) * eng.multiple)
    if rounded != value:
        LOGGER.warning(
            "%s: %s → %s (%s needs multiples of %s)",
            label,
            value,
            rounded,
            eng.name,
            eng.multiple,
        )
    return rounded


LORA_KEYS = {"name", "scale", "trigger"}

# (name or address, scale, trigger). A trigger of None means "the LoRA's own".
LoraItem = tuple[str, float, "str | None"]


def _loras(raw: Any, label: str) -> list[LoraItem]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(f"{label}.loras: must be a list")
    result: list[LoraItem] = []
    for item in raw:
        if isinstance(item, str):
            result.append((item, 1.0, None))
        elif isinstance(item, dict) and item.get("name"):
            odd = {k for k in item if not _comment(k)} - LORA_KEYS
            if odd:
                raise ConfigError(
                    f"{label}.loras: unknown keys {sorted(odd)} in "
                    f"{item['name']!r} (use name, scale, trigger)"
                )
            trigger = item.get("trigger")
            result.append(
                (
                    str(item["name"]),
                    float(_set(item, "scale", 1.0)),
                    None if trigger is None else str(trigger).strip(),
                )
            )
        else:
            raise ConfigError(
                f'{label}.loras: use "name" (or an address) or '
                '{"name": ..., "scale": ..., "trigger": ...}'
            )
    return result


def _output_part(value: str, label: str) -> str:
    part = PurePosixPath(value)
    if not value or part.is_absolute() or ".." in part.parts:
        raise ConfigError(f"{label}: {value!r} must be a relative folder without ..")
    return part.as_posix()


def expand(
    config: Any, registry: Registry, source: str = "config", default_output: str = ""
) -> list[Render]:
    if not isinstance(config, dict):
        raise ConfigError(f"{source}: must be a JSON object")
    version = config.get("version", SCHEMA_VERSION)  # a file without it is v1
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f'{source}: "version": {version!r} is not supported '
            f"(this jenair reads version {SCHEMA_VERSION}; the key may be left out)"
        )
    if "jobs" in config:
        unknown = {k for k in config if not _comment(k)} - TOP_KEYS
        defaults = config.get("defaults") or {}
        jobs = config["jobs"]
    else:  # a bare single job
        unknown = set()
        defaults = {}
        jobs = [{k: v for k, v in config.items() if k not in TOP_KEYS}]
    if unknown:
        LOGGER.warning("%s: ignoring unknown top-level keys %s", source, sorted(unknown))
    if not isinstance(defaults, dict) or not isinstance(jobs, list):
        raise ConfigError(f"{source}: defaults must be an object and jobs an array")
    output = _output_part(str(config.get("output") or default_output or "batch"), f"{source}.output")

    renders: list[Render] = []
    for index, job in enumerate(jobs, 1):
        label = f"{source} job {index}"
        if not isinstance(job, dict):
            raise ConfigError(f"{label}: must be an object")
        for keys, where in ((job, "job"), (defaults, "defaults")):
            later = set(keys) & LATER
            if later:
                LOGGER.warning(
                    "%s: %s %s not supported in v1; ignored", label, where, sorted(later)
                )
            odd = {k for k in keys if not _comment(k)} - JOB_KEYS - LATER
            if odd:
                LOGGER.warning("%s: ignoring unknown %s keys %s", label, where, sorted(odd))
        merged = {**defaults, **job}
        if merged.get("skip"):
            continue
        if not str(merged.get("prompt", "")).strip():
            raise ConfigError(f"{label}: prompt is required")
        if "model" not in merged:
            raise ConfigError(f"{label}: model is required")
        size = _layer_size(job, label) or _layer_size(defaults, f"{source} defaults")
        name = _output_part(str(merged.get("name") or f"job{index}"), f"{label}.name")
        loras = _loras(merged.get("loras"), label)
        for value in _as_list(merged["model"], f"{label}.model"):
            model = registry.model(value)
            eng = engine(model.engine)
            for seed in _seeds(job, defaults, label):
                renders.append(
                    _render(merged, model.alias, eng, registry, loras, size, seed,
                            name, output, label)
                )
    seen: dict[str, str] = {}
    for item in renders:
        if item.output in seen:
            raise ConfigError(
                f"{item.source} and {seen[item.output]} both write {item.output}; "
                "give the jobs different names"
            )
        seen[item.output] = item.source
    return renders


def _set(job: dict[str, Any], key: str, default: Any) -> Any:
    """A job value, where a missing key or null means the engine default."""
    value = job.get(key)
    return default if value is None else value


def _guidance(job: dict[str, Any], eng: Engine, label: str) -> float:
    """`cfg` is the CFG scale; `guidance` is the older, engine-internal name."""
    cfg, old = job.get("cfg"), job.get("guidance")
    if cfg is not None and old is not None:
        raise ConfigError(f"{label}: use cfg or guidance, not both")
    if cfg is None:
        return float(eng.guidance if old is None else old)
    if eng.name in FIXED_CFG:
        if float(cfg) != 1.0:
            LOGGER.warning(
                "%s: %s always runs at CFG 1; cfg %s is ignored", label, eng.name, cfg
            )
        return eng.guidance
    return guidance_of(eng, float(cfg))


def _render(
    job: dict[str, Any],
    alias: str,
    eng: Engine,
    registry: Registry,
    loras: list[LoraItem],
    size: tuple[int, int] | None,
    seed: int,
    name: str,
    output: str,
    label: str,
) -> Render:
    prompt = str(job["prompt"]).strip()
    triggers = []
    for lora_alias, _, own in loras:
        lora = registry.lora(lora_alias)
        if lora.engine and lora.engine != eng.name:  # "": any engine
            raise ConfigError(
                f"{label}: LoRA {lora_alias!r} is for {lora.engine}, "
                f"but model {alias!r} is {eng.name}"
            )
        # The job's trigger replaces a built-in one; "" switches it off.
        trigger = lora.trigger if own is None else own
        if trigger and trigger not in prompt and trigger not in triggers:
            triggers.append(trigger)
    if triggers:
        prompt = ", ".join([*triggers, prompt])

    guidance = _guidance(job, eng, label)
    negative = str(job.get("negative") or "")
    # krea2 encodes a negative only above CFG 1 (its guidance is CFG - 1).
    negative_used = eng.uses_negative and (eng.name != "krea2" or guidance > 0)
    if negative and not negative_used:
        LOGGER.warning("%s: %s ignores negative here (no CFG pass)", label, eng.name)
    negative = (negative or eng.default_negative) if negative_used else ""

    width, height = size or (eng.width, eng.height)
    sampler, scheduler = sampler_pair(
        eng, str(_set(job, "sampler", eng.sampler)), job.get("scheduler")
    )
    return Render(
        model=alias,
        engine=eng.name,
        name=name,
        prompt=prompt,
        negative=negative,
        seed=seed,
        width=_round(int(width), eng, f"{label} width"),
        height=_round(int(height), eng, f"{label} height"),
        steps=int(_set(job, "steps", eng.steps)),
        guidance=guidance,
        sampler=sampler,
        scheduler=scheduler,
        loras=tuple((n, s) for n, s, _ in loras),
        output=f"{output}/{alias}/{name}_{seed}.png",
        source=f"{label} ({alias}, seed {seed})",
    )


def target(config: Any, source: str = "config") -> str:
    """Where the job file asks to render: "local", "runpod" or "" (not said)."""
    value = config.get("target", "") if isinstance(config, dict) else ""
    if value and value not in TARGETS:
        raise ConfigError(f'{source}: target must be "local" or "runpod", not {value!r}')
    return str(value or "")


def read(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc


def load(path: Path, registry: Registry) -> list[Render]:
    return expand(read(path), registry, str(path), default_output=path.stem)
