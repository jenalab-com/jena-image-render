"""Supported engines: the files each one loads and its sampling defaults.

Defaults follow the official ComfyUI templates (ComfyUI v0.38.0) the graphs were
verified against. A config only overrides them when it really means to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ComfyUI models/<folder> for each file role a model entry may list.
FOLDERS = {
    "checkpoint": "checkpoints",
    "diffusion": "diffusion_models",
    "text_encoder": "text_encoders",
    "vae": "vae",
}

SDXL_NEGATIVE = (
    "lowres, (worst quality, low quality:1.2), bad anatomy, bad hands, "
    "missing fingers, extra digits, jpeg artifacts, signature, watermark, blurry"
)

# A1111-style sampler names (common in SDXL model cards) → ComfyUI sampler/scheduler.
SAMPLER_ALIASES = {
    "Euler a": ("euler_ancestral", "normal"),
    "Euler": ("euler", "normal"),
    "DPM++ 2M": ("dpmpp_2m", "normal"),
    "DPM++ 2M Karras": ("dpmpp_2m", "karras"),
    "DPM++ SDE Karras": ("dpmpp_sde", "karras"),
    "UniPC": ("uni_pc", "normal"),
}


@dataclass(frozen=True)
class Engine:
    name: str
    files: tuple[str, ...]  # file roles a model of this engine needs
    steps: int
    guidance: float
    width: int
    height: int
    sampler: str = "euler"
    scheduler: str = "simple"
    multiple: int = 16  # width/height must be multiples of this
    uses_negative: bool = False  # False: the graph never encodes a negative prompt
    default_negative: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


UNET_FILES = ("diffusion", "text_encoder", "vae")

# Distilled engines whose graph always samples at CFG 1: a job's cfg is ignored.
FIXED_CFG = frozenset({"zimage", "qwen", "klein"})


def cfg_of(eng: "Engine", guidance: float) -> float:
    """The real CFG scale behind an engine's internal guidance value."""
    if eng.name in FIXED_CFG:
        return 1.0
    return 1.0 + guidance if eng.name == "krea2" else guidance


def guidance_of(eng: "Engine", cfg: float) -> float:
    """A CFG scale as the engine's internal guidance (krea2 stores CFG - 1)."""
    return max(0.0, cfg - 1.0) if eng.name == "krea2" else cfg

ENGINES: dict[str, Engine] = {
    e.name: e
    for e in (
        Engine(
            "zimage",
            UNET_FILES,
            steps=9,
            guidance=0.0,
            width=832,
            height=1248,
            sampler="res_multistep",
            notes="Z-Image Turbo. No negative prompt (CFG 1).",
        ),
        Engine(
            "krea2",
            UNET_FILES,
            steps=8,
            guidance=0.0,
            width=1024,
            height=1024,
            uses_negative=True,
            notes="Krea 2 Turbo. cfg 1 (default) is the turbo mode with no negative pass.",
        ),
        Engine(
            "qwen",
            UNET_FILES,
            steps=25,
            guidance=0.0,
            width=832,
            height=1248,
            notes="Qwen-Image 2.1. No negative prompt (CFG 1).",
        ),
        Engine(
            "klein",
            UNET_FILES,
            steps=4,
            guidance=0.0,
            width=832,
            height=1248,
            notes="FLUX.2 klein 4B / 9B. No negative prompt (CFG 1).",
        ),
        Engine(
            "illustrious",
            ("checkpoint",),
            steps=28,
            guidance=6.0,
            width=896,
            height=1152,
            sampler="Euler a",
            scheduler="normal",
            multiple=8,
            uses_negative=True,
            default_negative=SDXL_NEGATIVE,
            notes="Any SDXL-architecture checkpoint (Illustrious, NoobAI, SDXL).",
        ),
    )
}

# Shorthand sizes. Every value is a multiple of 32, so it satisfies every engine,
# and stays near 1 megapixel (close to SDXL training buckets). Only 2:3 is exact.
ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "2:3": (832, 1248),
    "3:2": (1248, 832),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
}


def engine(name: str) -> Engine:
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; supported: {', '.join(ENGINES)}")
    return ENGINES[name]


def sampler_pair(eng: Engine, sampler: str, scheduler: str | None) -> tuple[str, str]:
    """Resolve an A1111 alias; an explicit scheduler always wins."""
    name, alias_scheduler = SAMPLER_ALIASES.get(sampler, (sampler, None))
    return name, scheduler or alias_scheduler or eng.scheduler
