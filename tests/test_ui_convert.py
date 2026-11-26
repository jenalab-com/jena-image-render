"""The page's JSON conversion (index.html, <script id="convert">), run with Node.

A file opened in the page and saved again must render the same images:
config.expand of the original and of the page's export agree on every image,
and the export is always in the `batches` form. Skipped without Node.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from jena_image_render import config, models
from jena_image_render.folder import ModelsFolder

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "src/jena_image_render/ui/index.html"
NODE = shutil.which("node")
DRIVER = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const block = html.match(/<script id="convert">([\s\S]*?)<\/script>/);
if (!block) throw new Error("no convert block");
const context = {};
vm.createContext(context);
vm.runInContext(block[1], context);
const C = context.JenairConvert;
const inputs = JSON.parse(fs.readFileSync(0, "utf8"));
const out = inputs.map(x => {
  const once = C.exportProject(C.importProject(x));
  return { once, twice: C.exportProject(C.importProject(once)) };
});
process.stdout.write(JSON.stringify(out));
"""
YOURS = (
    "checkpoints/my-checkpoint.safetensors",
    "loras/my-lora.safetensors",
    "diffusion_models/zimage/my-zimage-finetune.safetensors",
)
# Shapes the examples do not show: layers, groups, nulls, comments, odd keys.
CASES = [
    {
        "_note": "top", "version": 1, "target": "runpod", "output": "edge",
        "defaults": {"_c": "shared", "seeds": [1, 2], "aspect": "1:1", "negative": "blur",
                     "steps": 20,
                     "loras": ["zimage-pixel-art"]},
        "batches": [
            {"_note": "a batch", "name": "one",
             "defaults": {"model": "zimage", "seed": 5, "steps": None, "prompt": "shared",
                          "negative": "",
                          "loras": [{"name": "zimage-pixel-art", "scale": 0.5,
                                     "trigger": "", "_c": 1}]},
             "jobs": [{"name": "a"}, {"name": "b", "prompt": "own", "seeds": [7, 8],
                                      "width": 640, "height": 512},
                      {"prompt": "unnamed", "loras": []}, {"name": "c", "skip": True}]},
            {"name": "two", "defaults": {"model": ["zimage", "klein-4b"], "width": 768,
                                         "height": 768, "loras": None, "negative": ""},
             "jobs": [{"name": "m", "prompt": "two models", "sampler": "euler"}]},
            {"name": "three", "defaults": {"model": "illustrious", "guidance": 4.0,
                                           "sampler": "DPM++ 2M Karras", "aspect": "3:4",
                                           "loras": ["illustrious-niji"]},
             "jobs": [{"name": "g", "prompt": "legacy", "scheduler": None}]},
            {"name": "off", "skip": True, "jobs": [{"name": "z", "prompt": "never",
                                                    "model": "nope"}]},
        ],
    },
    {"_note": "one job", "target": "local", "model": "illustrious", "prompt": "x",
     "loras": [{"name": "illustrious-niji", "scale": 0.7, "trigger": "own words"}],
     "seed": 3, "width": 1000, "height": 1000, "cfg": 4.5, "negative": "bad",
     "steps": 22, "scheduler": "karras"},
    {"defaults": {"model": "zimage", "skip": True, "name": "n"},
     "jobs": [{"prompt": "kept", "skip": False}, {"prompt": "skipped"},
              {"name": "own", "prompt": "named", "skip": False}]},
    {"output": "o", "defaults": {"model": "krea2", "cfg": 2.5, "negative": "noise"},
     "jobs": [{"name": "k", "prompt": "turbo off",
               "loras": ["hf://krea/Krea-2-LoRA-dotmatrix/dotmatrix.safetensors"]}]},
]


def key(render: config.Render) -> tuple:
    return (render.model, render.engine, render.name, render.prompt, render.negative,
            render.seed, render.width, render.height, render.steps, render.guidance,
            render.sampler, render.scheduler, render.loras)


@unittest.skipIf(NODE is None, "Node is not installed")
class ConvertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        for rel in YOURS:
            (root / "models" / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / "models" / rel).write_bytes(b"x")
        cls.folder = ModelsFolder(root / "models")
        cls.cwd = os.getcwd()
        os.chdir(root)
        files = [ROOT / "configs/full.json", *sorted((ROOT / "configs").glob("*/*.json"))]
        cls.inputs = [(p.relative_to(ROOT).as_posix(), json.loads(p.read_text("utf-8")))
                      for p in files]
        cls.inputs += [(f"case {i}", case) for i, case in enumerate(CASES, 1)]
        driver = root / "driver.js"
        driver.write_text(DRIVER, "utf-8")
        done = subprocess.run(
            [NODE, str(driver), str(PAGE)], input=json.dumps([d for _, d in cls.inputs]),
            capture_output=True, text=True, timeout=60, check=False,
        )
        if done.returncode:
            raise AssertionError(done.stderr)
        cls.outputs = json.loads(done.stdout)

    @classmethod
    def tearDownClass(cls) -> None:
        os.chdir(cls.cwd)
        cls.tmp.cleanup()

    def expand(self, data):
        registry = models.load(self.folder)
        return config.expand(data, registry, "t", default_output="dflt")

    def test_saved_again_renders_the_same_images(self) -> None:
        self.assertGreater(len(self.inputs), 20)
        for (name, original), out in zip(self.inputs, self.outputs):
            with self.subTest(file=name):
                exported = out["once"]
                self.assertIn("batches", exported)
                self.assertNotIn("jobs", exported)
                before, after = self.expand(original), self.expand(exported)
                self.assertTrue(before, name)
                self.assertEqual([key(r) for r in before], [key(r) for r in after])
                if "batches" in original:
                    self.assertEqual([r.output for r in before], [r.output for r in after])
                else:  # one tab: the batch folder is added under the output
                    for b, a in zip(before, after):
                        head, rest = b.output.split("/", 1)
                        self.assertEqual(a.output, f"{head}/main/{rest}")
                self.assertEqual(out["twice"], exported)  # nothing drifts on a re-save

    def test_kept_keys_survive(self) -> None:
        exported = self.outputs[[n for n, _ in self.inputs].index("case 1")]["once"]
        self.assertEqual((exported["_note"], exported["version"], exported["target"]),
                         ("top", 1, "runpod"))
        one, two, three, off = exported["batches"]
        self.assertEqual(one["_note"], "a batch")
        self.assertEqual(exported["defaults"], {"_c": "shared"})  # not copied per tab
        self.assertNotIn("_c", one["defaults"])
        self.assertEqual(one["defaults"]["loras"][0]["_c"], 1)
        self.assertEqual(three["defaults"]["guidance"], 4.0)
        self.assertEqual(off["skip"], True)
        self.assertEqual(one["jobs"][2]["name"], "job3")  # the name expand gives it


if __name__ == "__main__":
    unittest.main()
