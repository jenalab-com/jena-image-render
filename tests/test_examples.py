"""The shipped examples must run as they are: configs/ against the built-in
catalog and the addresses they write, with an empty models folder."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from jena_image_render import assets, config, models
from jena_image_render.engines import ENGINES
from jena_image_render.folder import ModelsFolder

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted(
    [
        ROOT / "configs/full.json",
        *(ROOT / "configs/basics").glob("*.json"),
        *(ROOT / "configs/engines").glob("*.json"),
    ]
)
YOUR_FILES = sorted((ROOT / "configs/your-files").glob("*.json"))
YOURS = (  # the files the your-files examples tell the reader to add
    "checkpoints/my-checkpoint.safetensors",
    "loras/my-lora.safetensors",
    "diffusion_models/zimage/my-zimage-finetune.safetensors",
)


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = os.getcwd()
        os.chdir(cls.tmp.name)
        cls.registry = models.load(ModelsFolder(Path(cls.tmp.name) / "models"))
        cls.renders = {path.name: config.load(path, cls.registry) for path in CONFIGS}

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def test_every_example_expands(self) -> None:
        self.assertTrue(CONFIGS)
        for name, renders in self.renders.items():
            self.assertTrue(renders, name)

    def test_nothing_needs_a_file_on_this_computer(self) -> None:
        """A fresh clone has no weights: every example file is a URL."""
        everything = [r for renders in self.renders.values() for r in renders]
        files = assets.needed(everything, self.registry)
        catalog = {
            spec.dest
            for model in self.registry.models.values()
            for spec in model.files.values()
        } | {lora.file.dest for lora in self.registry.loras.values()}
        for spec in files:
            self.assertTrue(spec.url, spec.dest)
            self.assertFalse(spec.local, spec.dest)
            if spec.dest not in catalog:  # an address written in a job
                self.assertIn("/_url/", spec.dest)
                continue
            self.assertIn("/resolve/", spec.url, spec.dest)  # built in: pinned
            self.assertNotIn("/resolve/main/", spec.url, spec.dest)
            self.assertEqual(len(spec.sha256), 64, spec.dest)
            self.assertGreater(spec.size, 0, spec.dest)
        self.assertEqual({how for how, _ in assets.plan(files, "runpod")}, {"fetch"})

    def test_addresses_example(self) -> None:
        renders = self.renders["08-from-url.json"]
        self.assertEqual(
            [(r.model, r.engine) for r in renders],
            [
                ("krea2", "krea2"),
                ("Illustrious-XL-v1.0", "illustrious"),
                ("zimage-from-url", "zimage"),
            ],
        )
        files = assets.needed(renders, self.registry)
        own = [f for f in files if "/_url/" in f.dest]
        self.assertEqual(
            sorted(f.dest.split("/")[0] for f in own),
            ["checkpoints", "loras", "vae"],
        )

    def test_every_engine_has_a_lora_example(self) -> None:
        used = {
            r.engine
            for renders in self.renders.values()
            for r in renders
            if r.loras
        }
        self.assertEqual(used, set(ENGINES))

    def test_every_catalog_lora_is_used(self) -> None:
        used = {
            alias
            for renders in self.renders.values()
            for r in renders
            for alias, _ in r.loras
        }
        self.assertEqual(used & set(self.registry.loras), set(self.registry.loras))

    def test_no_models_json_left(self) -> None:
        self.assertFalse((ROOT / "models.example.json").exists())
        texts = [ROOT / "README.md", ROOT / ".env.example", ROOT / ".gitignore"]
        texts += [*(ROOT / "docs").glob("*.md"), *(ROOT / "configs").rglob("*.*")]
        texts += [ROOT / "models/README.md", ROOT / ".github/workflows/ci.yml"]
        for path in texts:
            text = path.read_text("utf-8")
            self.assertIsNone(re.search(r"(?<![\w-])models\.json", text), path.name)
            self.assertFalse("JENA_MODELS=" in text, path.name)

    def test_examples_carry_a_note_and_no_warnings(self) -> None:
        for path in [*CONFIGS, *YOUR_FILES]:
            self.assertIn("_note", json.loads(path.read_text("utf-8")), path.name)
        with self.assertNoLogs("jena_image_render", "WARNING"):
            for path in CONFIGS:
                config.load(path, self.registry)

    def test_your_files_examples_need_only_the_named_files(self) -> None:
        self.assertEqual(len(YOUR_FILES), 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = models.load(ModelsFolder(root / "none"))
            for path in YOUR_FILES:
                with self.assertRaises(ValueError, msg=path.name):
                    config.load(path, empty)
            for rel in YOURS:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"x")
            reg = models.load(ModelsFolder(root))
            for path in YOUR_FILES:
                renders = config.load(path, reg)
                self.assertTrue(renders, path.name)
                plan = assets.plan(assets.needed(renders, reg), "runpod")
                self.assertNotIn("missing", {how for how, _ in plan}, path.name)
                self.assertIn("upload", {how for how, _ in plan}, path.name)

    def test_full_json_is_the_dictionary(self) -> None:
        """configs/full.json shows every key and every form: keep it complete."""
        full = json.loads((ROOT / "configs/full.json").read_text("utf-8"))
        batches = full["batches"]
        layers = [full["defaults"]]
        for batch in batches:
            layers += [batch.get("defaults") or {}, *batch["jobs"]]
        used = {k for layer in layers for k in layer if not k.startswith("_")}
        legacy = {"guidance"}  # still read; `cfg` is the documented key
        self.assertEqual(config.JOB_KEYS - legacy - used, set())
        # A file with batches has no top-level jobs (they may not be used together).
        self.assertEqual(config.TOP_KEYS - {"jobs"} - set(full), set())
        self.assertNotIn("jobs", full)
        in_batches = {k for batch in batches for k in batch if not k.startswith("_")}
        self.assertEqual(config.BATCH_KEYS - in_batches, set())
        self.assertTrue(any(batch.get("skip") is True for batch in batches))
        self.assertIn("basics/02", full["_note"])  # one batch: defaults + jobs
        loras = [x for layer in layers for x in (layer.get("loras") or [])]
        self.assertTrue(any(isinstance(x, str) for x in loras))
        entries = [x for x in loras if isinstance(x, dict)]
        self.assertEqual(config.LORA_KEYS - {k for x in entries for k in x}, set())
        self.assertTrue(any(x.get("trigger") == "" for x in entries))
        self.assertTrue(any("://" in x["name"] for x in entries))  # by address
        self.assertIn([], [layer.get("loras") for layer in layers])  # none
        kinds = [layer["model"] for layer in layers if "model" in layer]
        self.assertTrue(any(isinstance(m, list) for m in kinds))
        self.assertTrue(any(isinstance(m, dict) for m in kinds))
        self.assertTrue(any(isinstance(m, str) and "://" in m for m in kinds))
        self.assertTrue(any(None in layer.values() for layer in layers))  # null


if __name__ == "__main__":
    unittest.main()
