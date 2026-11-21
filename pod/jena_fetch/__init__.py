"""ComfyUI extension: model downloads on request and pod self-termination when idle."""

from importlib.util import find_spec

NODE_CLASS_MAPPINGS: dict = {}

# The pure modules (fetcher, idle) are importable outside ComfyUI for tests.
if find_spec("server") and find_spec("folder_paths"):
    from . import comfy_routes  # noqa: F401  registers routes on import
