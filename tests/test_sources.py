"""Files written directly in a job: addresses and object-form models."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from jena_image_render import assets, models
from jena_image_render.config import ConfigError, expand
from jena_image_render.folder import ModelsFolder

LORA = "hf://someone/their-lora/their-lora.safetensors"
CKPT = "hf://someone/their-ckpt@abc123/sdxl/their-ckpt.safetensors"
DIFF = "https://my-storage.example.com/finetunes/model.safetensors"


def short(address: str) -> str:
    return hashlib.sha1(address.encode("utf-8")).hexdigest()[:8]


class SourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "models"
        self.reg = models.load(ModelsFolder(self.dir))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_job(self, **job):
        return expand({"version": 1, "prompt": "a cafe", **job}, self.reg, "t", "out")

    def test_lora_address_any_engine(self) -> None:
        other = "https://cdn.example.com/x/their-lora.safetensors"
        renders = self.run_job(
            model=["zimage", "illustrious"], loras=[LORA, {"name": other, "scale": 0.8}]
        )
        self.assertEqual([r.prompt for r in renders], ["a cafe", "a cafe"])  # no trigger
        self.assertEqual(renders[0].loras, ((LORA, 1.0), (other, 0.8)))
        first, second = self.reg.lora(LORA), self.reg.lora(other)
        self.assertEqual(first.engine, "")
        self.assertEqual(first.file.dest, f"loras/_url/{short(LORA)}/their-lora.safetensors")
        self.assertEqual(
            first.file.url,
            "https://huggingface.co/someone/their-lora/resolve/main/their-lora.safetensors",
        )
        self.assertEqual((first.file.sha256, first.file.size, first.file.local), ("", 0, ""))
        self.assertEqual(second.file.url, other)
        self.assertNotEqual(first.file.dest, second.file.dest)  # same file name
        self.assertEqual(first.file.name.rsplit("/", 1)[1], second.file.name.rsplit("/", 1)[1])
        names = assets.pod_names(renders[0], self.reg)
        self.assertEqual(names.loras[0], f"_url/{short(LORA)}/their-lora.safetensors")

    def test_checkpoint_address(self) -> None:
        (r,) = self.run_job(model=CKPT)
        self.assertEqual((r.model, r.engine), ("their-ckpt", "illustrious"))
        self.assertEqual(r.output, "out/their-ckpt/job1_0.png")
        spec = self.reg.model("their-ckpt").files["checkpoint"]
        self.assertEqual(spec.dest, f"checkpoints/_url/{short(CKPT)}/their-ckpt.safetensors")
        self.assertIn("/resolve/abc123/sdxl/their-ckpt.safetensors", spec.url)
        (how, _), = assets.plan(assets.needed([r], self.reg), "runpod")
        self.assertEqual(how, "fetch")

    def test_object_model_borrows_missing_roles(self) -> None:
        raw = {"engine": "zimage", "diffusion": DIFF, "name": "my-z"}
        (r,) = self.run_job(model=raw)
        self.assertEqual((r.model, r.engine, r.output), ("my-z", "zimage", "out/my-z/job1_0.png"))
        files = self.reg.model("my-z").files
        official = self.reg.model("zimage").files
        self.assertEqual(files["diffusion"].dest, f"diffusion_models/_url/{short(DIFF)}/model.safetensors")
        self.assertEqual(files["diffusion"].url, DIFF)
        self.assertEqual((files["text_encoder"], files["vae"]), (official["text_encoder"], official["vae"]))
        klein = self.reg.model({"engine": "klein", "vae": "https://x.example/v.safetensors"})
        self.assertEqual(klein.alias, "flux-2-klein-4b")  # stem of the borrowed diffusion
        self.assertEqual(klein.files["diffusion"], self.reg.model("klein-4b").files["diffusion"])
        self.assertEqual(klein.files["vae"].dest.split("/")[:2], ["vae", "_url"])

    def test_object_name_defaults_to_the_stem(self) -> None:
        model = self.reg.model({"engine": "illustrious", "checkpoint": CKPT})
        self.assertEqual(model.alias, "their-ckpt")
        self.assertEqual(model, self.reg.model(CKPT))  # the same model, written two ways

    def test_object_roles_from_the_models_folder(self) -> None:
        (self.dir / "text_encoders/sub").mkdir(parents=True)
        (self.dir / "text_encoders/sub/my-te.safetensors").write_bytes(b"t")
        self.reg = models.load(ModelsFolder(self.dir))  # the folder is indexed once
        model = self.reg.model({"engine": "qwen", "text_encoder": "my-te", "diffusion": DIFF})
        spec = model.files["text_encoder"]
        self.assertEqual(spec.dest, "text_encoders/sub/my-te.safetensors")
        self.assertEqual(spec.local, str(self.dir / spec.dest))
        with self.assertRaisesRegex(models.ModelsError, "is not an address .* or a file in"):
            self.reg.model({"engine": "qwen", "vae": "nowhere"})

    def test_object_errors(self) -> None:
        for raw, message in (
            ({"diffusion": DIFF}, 'needs an "engine"'),
            ({"engine": "pony", "diffusion": DIFF}, "unknown engine"),
            ({"engine": "zimage", "checkpoint": CKPT}, "does not use"),
            ({"engine": "illustrious", "name": "x"}, "illustrious needs checkpoint"),
            ({"engine": "zimage", "diffusion": DIFF, "files": {}}, "unknown keys"),
            ({"engine": "zimage", "diffusion": DIFF, "name": "zimage"}, "built-in alias"),
            ({"engine": "zimage", "diffusion": DIFF, "name": "a/b"}, "invalid name"),
            ({"engine": "zimage", "diffusion": ""}, "needs an address or a file name"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                self.reg.model(raw)
        self.assertEqual(self.reg.model({"engine": "zimage", "diffusion": DIFF, "_note": "ok"}).alias, "model")

    def test_two_models_with_one_name(self) -> None:
        self.reg.model({"engine": "zimage", "diffusion": DIFF, "name": "mine"})
        self.reg.model({"engine": "zimage", "diffusion": DIFF, "name": "mine"})  # same: fine
        with self.assertRaisesRegex(models.ModelsError, 'two different models are named'):
            self.reg.model({"engine": "krea2", "diffusion": DIFF, "name": "mine"})
        with self.assertRaisesRegex(models.ModelsError, 'built-in alias; write it as an object'):
            self.reg.model("https://x.example/zimage.safetensors")

    def test_mixed_model_list(self) -> None:
        renders = self.run_job(
            model=["zimage", CKPT, {"engine": "krea2", "diffusion": DIFF, "name": "k"}],
            seeds=[1, 2],
        )
        self.assertEqual(
            sorted({(r.model, r.engine) for r in renders}),
            [("k", "krea2"), ("their-ckpt", "illustrious"), ("zimage", "zimage")],
        )
        files = assets.needed(renders, self.reg)
        self.assertEqual(len(files), 3 + 1 + 3)  # krea2 keeps its text encoder and VAE
        with self.assertRaisesRegex(ConfigError, "is for zimage"):
            self.run_job(model=CKPT, loras=["zimage-pixel-art"])

    def test_address_needs_a_file_name(self) -> None:
        for bad in ("https://x.example/", "hf://owner/repo"):
            with self.subTest(bad=bad), self.assertRaises(models.ModelsError):
                self.reg.lora(bad)
        with self.assertRaisesRegex(models.ModelsError, "use a name, an address"):
            self.reg.model(3)

    def test_pulled_address_is_a_local_copy(self) -> None:
        spec = self.reg.lora(LORA).file
        (self.dir / spec.dest).parent.mkdir(parents=True)
        (self.dir / spec.dest).write_bytes(b"x")
        again = models.load(ModelsFolder(self.dir)).lora(LORA).file
        self.assertEqual((again.url, again.local), (spec.url, str(self.dir / spec.dest)))


if __name__ == "__main__":
    unittest.main()
