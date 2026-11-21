"""Name resolution: catalog → models folder (checkpoints/, diffusion_models/<engine>/)."""

import tempfile
import unittest
from pathlib import Path

from jena_image_render import models
from jena_image_render.config import ConfigError, expand
from jena_image_render.folder import ModelsFolder


def touch(root: Path, rel: str, data: bytes = b"w") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class FolderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dir = self.root / "models"
        self.folder = ModelsFolder(self.dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def registry(self) -> models.Registry:
        return models.load(self.folder)

    def test_checkpoint_found_by_stem_name_or_path(self) -> None:
        touch(self.dir, "checkpoints/sdxl/my-anime.safetensors")
        touch(self.dir, "checkpoints/README.md")
        reg = self.registry()
        for name in ("my-anime", "my-anime.safetensors", "sdxl/my-anime"):
            with self.subTest(name=name):
                model = reg.model(name)
                self.assertEqual(model.engine, "illustrious")
                spec = model.files["checkpoint"]
                self.assertEqual(spec.dest, "checkpoints/sdxl/my-anime.safetensors")
                self.assertEqual(spec.name, "sdxl/my-anime.safetensors")
                self.assertEqual(spec.local, str(self.dir / spec.dest))
        with self.assertRaises(models.ModelsError):
            reg.model("README")

    def test_catalog_beats_folder(self) -> None:
        touch(self.dir, "checkpoints/zimage.safetensors")
        touch(self.dir, "diffusion_models/krea2/krea2.safetensors")
        reg = self.registry()
        self.assertEqual(reg.model("zimage").engine, "zimage")
        self.assertEqual(reg.model("zimage"), models.catalog().model("zimage"))
        self.assertEqual(reg.model("krea2"), models.catalog().model("krea2"))

    def test_unknown_name_says_where_to_put_it(self) -> None:
        with self.assertRaisesRegex(models.ModelsError, "known: illustrious, klein-4b") as ctx:
            self.registry().model("nope")
        message = str(ctx.exception)
        self.assertIn("put the file there", message)
        self.assertIn("models/checkpoints/ or ", message)
        self.assertIn("models/diffusion_models/<engine>/", message)
        self.assertIn('write its address in the job: "model": "hf://', message)
        self.assertNotIn("models.json", message)
        self.assertNotIn("storage", message)
        with self.assertRaisesRegex(models.ModelsError, "models/loras/") as ctx:
            self.registry().lora("nope")
        self.assertIn('"loras": ["hf://', str(ctx.exception))

    def test_same_stem_twice_is_ambiguous(self) -> None:
        touch(self.dir, "checkpoints/a/x.safetensors")
        touch(self.dir, "checkpoints/b/x.safetensors")
        reg = self.registry()
        with self.assertRaisesRegex(models.ModelsError, "a/x.safetensors, checkpoints/b"):
            reg.model("x")
        self.assertEqual(reg.model("a/x").files["checkpoint"].dest, "checkpoints/a/x.safetensors")

    def test_lora_engine_from_first_sub_folder(self) -> None:
        touch(self.dir, "loras/zimage/z-style.safetensors")
        touch(self.dir, "loras/styles/any-style.safetensors")
        reg = self.registry()
        z = reg.lora("z-style")
        self.assertEqual((z.engine, z.file.dest, z.trigger), ("zimage", "loras/zimage/z-style.safetensors", ""))
        self.assertEqual(reg.lora("any-style").engine, "")
        self.assertEqual(reg.lora("any-style").file.name, "styles/any-style.safetensors")

    def test_any_engine_lora_skips_the_engine_check(self) -> None:
        touch(self.dir, "loras/any-style.safetensors")
        touch(self.dir, "loras/zimage/z-style.safetensors")
        touch(self.dir, "checkpoints/my-anime.safetensors")
        reg = self.registry()
        for model in ("my-anime", "zimage"):
            (r,) = expand({"version": 1, "model": model, "prompt": "x", "loras": ["any-style"]}, reg)
            self.assertEqual(r.loras, (("any-style", 1.0),))
        with self.assertRaisesRegex(ConfigError, "is for zimage"):
            expand({"version": 1, "model": "my-anime", "prompt": "x", "loras": ["z-style"]}, reg)

    def test_fine_tune_in_an_engine_folder(self) -> None:
        touch(self.dir, "diffusion_models/zimage/my-finetune.safetensors")
        touch(self.dir, "diffusion_models/klein/my-klein.safetensors")
        reg = self.registry()
        model = reg.model("my-finetune")
        self.assertEqual((model.alias, model.engine), ("my-finetune", "zimage"))
        spec = model.files["diffusion"]
        self.assertEqual(spec.dest, "diffusion_models/zimage/my-finetune.safetensors")
        self.assertEqual(spec.name, "zimage/my-finetune.safetensors")
        self.assertEqual(spec.local, str(self.dir / spec.dest))
        official = reg.model("zimage").files
        self.assertEqual(model.files["text_encoder"], official["text_encoder"])
        self.assertEqual(model.files["vae"], official["vae"])
        self.assertEqual(reg.model("zimage/my-finetune.safetensors").files, model.files)
        klein = reg.model("my-klein")
        self.assertEqual(klein.engine, "klein")
        self.assertEqual(klein.files["vae"], reg.model("klein-4b").files["vae"])
        (r,) = expand({"version": 1, "model": "my-finetune", "prompt": "x",
                       "loras": ["zimage-pixel-art"]}, reg)
        self.assertEqual((r.engine, r.output), ("zimage", "batch/my-finetune/job1_0.png"))

    def test_diffusion_model_outside_an_engine_folder(self) -> None:
        touch(self.dir, "diffusion_models/loose.safetensors")
        touch(self.dir, "diffusion_models/tuned/other.safetensors")
        touch(self.dir, "diffusion_models/illustrious/sdxl.safetensors")
        reg = self.registry()
        for name in ("loose", "other", "sdxl"):
            with self.subTest(name=name), self.assertRaises(models.ModelsError) as ctx:
                reg.model(name)
            self.assertIn("engine of diffusion_models/", str(ctx.exception))
            self.assertIn("diffusion_models/<engine>/", str(ctx.exception))
            self.assertIn('object with an "engine"', str(ctx.exception))

    def test_same_stem_as_checkpoint_and_diffusion_model(self) -> None:
        touch(self.dir, "checkpoints/twin.safetensors")
        touch(self.dir, "diffusion_models/zimage/twin.safetensors")
        reg = self.registry()
        with self.assertRaisesRegex(
            models.ModelsError,
            "checkpoints/twin.safetensors and diffusion_models/zimage/twin.safetensors",
        ):
            reg.model("twin")

    def test_catalog_copy_is_known_anywhere_in_its_folder(self) -> None:
        touch(self.dir, "diffusion_models/pulled/z_image_turbo_bf16.safetensors", b"abc")
        spec = self.registry().model("zimage").files["diffusion"]
        self.assertEqual(spec.local, "")  # the catalog's size differs
        data = {"models": {"z": {"engine": "zimage", "files": {
            "diffusion": "https://x.example/z_image_turbo_bf16.safetensors",
            "text_encoder": "https://x.example/t.safetensors",
            "vae": "https://x.example/v.safetensors"}}}}
        spec = models.parse(data, folder=self.folder).model("z").files["diffusion"]
        self.assertEqual(spec.local, str(self.dir / "diffusion_models/pulled/z_image_turbo_bf16.safetensors"))

    def test_url_file_notes_a_local_copy_of_the_same_size(self) -> None:
        touch(self.dir, "vae/sub/v.safetensors", b"abc")
        data = {"models": {"m": {"engine": "illustrious", "files": {"checkpoint": {"source": "https://x.example/v.safetensors", "size": 3}}}}}
        spec = models.parse(data, folder=self.folder).model("m").files["checkpoint"]
        self.assertEqual(spec.local, "")  # different role folder
        data["models"]["m"]["files"]["checkpoint"]["source"] = "https://x.example/c.safetensors"
        touch(self.dir, "checkpoints/c.safetensors", b"abc")
        spec = models.parse(data, folder=self.folder).model("m").files["checkpoint"]
        self.assertEqual((spec.url, spec.local), ("https://x.example/c.safetensors", str(self.dir / "checkpoints/c.safetensors")))
        touch(self.dir, "checkpoints/c.safetensors", b"abcd")
        spec = models.parse(data, folder=self.folder).model("m").files["checkpoint"]
        self.assertEqual(spec.local, "")  # size differs: not the same file


if __name__ == "__main__":
    unittest.main()
