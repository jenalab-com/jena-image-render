"""Settings from the environment and an optional .env file (real env wins)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_POD_IMAGE = "ghcr.io/jenalab-com/jena-image-render-pod:v1"
STATE_DIR = Path(".jena")


def load_dotenv(path: Path = Path(".env")) -> None:
    """KEY=value lines; # comments; optional quotes. Never overrides real env."""
    if not path.is_file():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()
