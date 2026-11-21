"""ComfyUI API graphs, one per engine, following the official v0.38.0 templates.

The user never sees these: a Render plus the server-side file names is enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jena_image_render.config import Render

Graph = dict[str, dict[str, Any]]
Link = list[Any]
OUTPUT_NODE = "output"


@dataclass(frozen=True)
class Names:
    """File names as the ComfyUI loaders list them."""

    base: str  # diffusion model, or the checkpoint for illustrious
    clip: str = ""
    vae: str = ""
    loras: tuple[str, ...] = ()


class _Builder:
    def __init__(self, render: Render, names: Names) -> None:
        self.r = render
        self.names = names
        self.graph: Graph = {}

    def add(self, key: str, class_type: str, **inputs: Any) -> Link:
        self.graph[key] = {"class_type": class_type, "inputs": inputs}
        return [key, 0]

    def encode(self, key: str, clip: Link, text: str) -> Link:
        return self.add(key, "CLIPTextEncode", clip=clip, text=text)

    def zero(self, positive: Link) -> Link:
        return self.add("negative", "ConditioningZeroOut", conditioning=positive)

    def _scales(self) -> list[float]:
        return [scale for _, scale in self.r.loras]

    # ── loaders ──────────────────────────────────────────────────────
    def unet(self, clip_type: str) -> tuple[Link, Link, Link]:
        model = self.add(
            "unet", "UNETLoader", unet_name=self.names.base, weight_dtype="default"
        )
        for index, (lora, scale) in enumerate(zip(self.names.loras, self._scales())):
            model = self.add(
                f"lora{index}",
                "LoraLoaderModelOnly",
                model=model,
                lora_name=lora,
                strength_model=scale,
            )
        clip = self.add("clip", "CLIPLoader", clip_name=self.names.clip, type=clip_type)
        vae = self.add("vae", "VAELoader", vae_name=self.names.vae)
        return model, clip, vae

    # ── engines: (model, vae, positive, negative, cfg) ───────────────
    def zimage(self) -> tuple[Link, Link, Link, Link, float]:
        model, clip, vae = self.unet("lumina2")
        model = self.add("shift", "ModelSamplingAuraFlow", model=model, shift=3.0)
        positive = self.encode("positive", clip, self.r.prompt)
        return model, vae, positive, self.zero(positive), 1.0

    def krea2(self) -> tuple[Link, Link, Link, Link, float]:
        model, clip, vae = self.unet("krea2")
        positive = self.encode("positive", clip, self.r.prompt)
        guidance = self.r.guidance  # Krea's guidance is CFG - 1; 0 = turbo
        if guidance > 0:
            negative = self.encode("negative", clip, self.r.negative)
            return model, vae, positive, negative, 1.0 + guidance
        return model, vae, positive, self.zero(positive), 1.0

    def qwen(self) -> tuple[Link, Link, Link, Link, float]:
        model, clip, vae = self.unet("qwen_image")
        model = self.add(
            "cache", "QwenImage21Cache", model=model, device="auto", dtype="default"
        )
        self.add(
            "encode",
            "TextEncodeQwenImage21",
            clip=clip,
            prompt=self.r.prompt,
            negative_prompt="",
            resolution=1024,
        )
        return model, vae, ["encode", 0], ["encode", 1], 1.0

    def klein(self) -> tuple[Link, Link, Link, Link, float]:
        model, clip, vae = self.unet("flux2")
        positive = self.encode("positive", clip, self.r.prompt)
        return model, vae, positive, self.zero(positive), 1.0

    def illustrious(self) -> tuple[Link, Link, Link, Link, float]:
        model: Link = self.add(
            "checkpoint", "CheckpointLoaderSimple", ckpt_name=self.names.base
        )
        clip: Link = ["checkpoint", 1]
        for index, (lora, scale) in enumerate(zip(self.names.loras, self._scales())):
            key = f"lora{index}"
            self.add(
                key,
                "LoraLoader",
                model=model,
                clip=clip,
                lora_name=lora,
                strength_model=scale,
                strength_clip=scale,
            )
            model, clip = [key, 0], [key, 1]
        positive = self.encode("positive", clip, self.r.prompt)
        negative = self.encode("negative", clip, self.r.negative)
        return model, ["checkpoint", 2], positive, negative, self.r.guidance

    # ── sampling ─────────────────────────────────────────────────────
    def klein_sample(self, model: Link, positive: Link, negative: Link) -> Link:
        r = self.r
        guider = self.add(
            "guider", "CFGGuider", model=model, positive=positive, negative=negative,
            cfg=1.0,
        )
        sampler = self.add("sampler", "KSamplerSelect", sampler_name=r.sampler)
        sigmas = self.add(
            "sigmas", "Flux2Scheduler", steps=r.steps, width=r.width, height=r.height
        )
        noise = self.add("noise", "RandomNoise", noise_seed=r.seed)
        latent = self.add(
            "latent", "EmptyFlux2LatentImage", width=r.width, height=r.height,
            batch_size=1,
        )
        return self.add(
            "sample",
            "SamplerCustomAdvanced",
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )

    def build(self) -> Graph:
        r = self.r
        model, vae, positive, negative, cfg = getattr(self, r.engine)()
        if r.engine == "klein":
            samples = self.klein_sample(model, positive, negative)
        else:
            # As in the verified graphs: SD3 latents for zimage/krea2, plain for
            # qwen (its 2.1 template) and illustrious (SDXL).
            kind = (
                "EmptySD3LatentImage"
                if r.engine in ("zimage", "krea2")
                else "EmptyLatentImage"
            )
            latent = self.add(
                "latent", kind, width=r.width, height=r.height, batch_size=1
            )
            samples = self.add(
                "sample",
                "KSampler",
                model=model,
                seed=r.seed,
                steps=r.steps,
                cfg=cfg,
                sampler_name=r.sampler,
                scheduler=r.scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=1.0,
            )
        image = self.add("decode", "VAEDecode", samples=samples, vae=vae)
        # Temp output: ComfyUI does not keep it; the runner saves its own copy.
        self.add(OUTPUT_NODE, "PreviewImage", images=image)
        return self.graph


def build_graph(render: Render, names: Names) -> Graph:
    if len(names.loras) != len(render.loras):
        raise ValueError("every LoRA must be resolved before building the graph")
    return _Builder(render, names).build()
