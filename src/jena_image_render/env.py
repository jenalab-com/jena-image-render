"""Settings from the environment and an optional .env file (real env wins)."""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_POD_IMAGE = "ghcr.io/jenalab-com/jena-image-render-pod:v1"
STATE_DIR = Path(".jena")
_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def read_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """KEY=value lines; # comments; optional quotes and `export`."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip().startswith("#") or (match := _LINE.match(line)) is None:
            continue
        values[match.group(1)] = _value(line[match.end():])
    return values


def load_dotenv(path: Path = Path(".env")) -> None:
    """Put .env values in the environment. Never overrides real env."""
    for key, value in read_dotenv(path).items():
        if key not in os.environ:
            os.environ[key] = value


def _quoted(value: str) -> str:
    """A value as written to .env: quoted only when the reader would strip quotes."""
    if value and value[0] in "\"'" and value[-1] == value[0]:
        other = "'" if value[0] == '"' else '"'
        return f"{other}{value}{other}"
    return value


def update_dotenv(path: Path, values: dict[str, str]) -> None:
    """Set KEY=value in .env: each key's line is changed in place, or appended.

    Comments, other lines and their order are kept. "" writes `KEY=`. The file
    holds keys, so it is readable by its owner only.
    """
    lines = path.read_text("utf-8").splitlines() if path.is_file() else []
    left = dict(values)
    for index, line in enumerate(lines):
        match = None if line.strip().startswith("#") else _LINE.match(line)
        if match is not None and match.group(1) in left:
            key = match.group(1)
            lines[index] = f"{key}={_quoted(left.pop(key))}"
    lines += [f"{key}={_quoted(value)}" for key, value in left.items()]
    partial = path.with_name(path.name + ".part")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    os.chmod(partial, 0o600)
    partial.replace(path)


def get(name: str, default: str = "") -> str:
    """A setting; empty (as `KEY=` in .env) counts as not set."""
    return os.environ.get(name, "").strip() or default
