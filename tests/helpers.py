"""Shared fixtures: a small registry covering every engine."""

from __future__ import annotations

from pathlib import Path

from jena_image_render import models

SHA_A = "a" * 64

REGISTRY_DATA = {
    "version": 1,
    "models": {
        "zimage": {
            "engine": "zimage",
            "files": {
                "diffusion": "hf://org/zimage@rev1/split/z.safetensors",
                "text_encoder": {"source": "hf://org/zimage/te/qwen_3_4b.safetensors", "sha256": SHA_A},
                "vae": "https://example.com/vae/ae.safetensors",
            },
        },
        "krea2": {
            "engine": "krea2",
            "files": {
                "diffusion": "https://example.com/k.safetensors",
                "text_encoder": "https://example.com/kte.safetensors",
                "vae": "https://example.com/kvae.safetensors",
            },
        },
        "qwen": {
            "engine": "qwen",
            "files": {
                "diffusion": "https://example.com/q.safetensors",
                "text_encoder": "https://example.com/qte.safetensors",
                "vae": "https://example.com/qvae.safetensors",
            },
        },
        "klein": {
            "engine": "klein",
            "files": {
                "diffusion": "https://example.com/f.safetensors",
                "text_encoder": {"source": "https://example.com/other/qwen_3_4b.safetensors", "sha256": SHA_A},
                "vae": "https://example.com/flux2-vae.safetensors",
            },
        },
        "ill": {"engine": "illustrious", "files": {"checkpoint": "ckpt/ill.safetensors"}},
    },
    "loras": {
        "zstyle": {"engine": "zimage", "file": "https://example.com/zs.safetensors", "trigger": "zst"},
        "istyle": {"engine": "illustrious", "file": "loras/is.safetensors"},
    },
}


def registry(base: Path = Path("/base")) -> models.Registry:
    return models.parse(REGISTRY_DATA, base)
