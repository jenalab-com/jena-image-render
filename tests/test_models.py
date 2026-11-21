import copy
import os
import tempfile
import unittest
from pathlib import Path

from helpers import REGISTRY_DATA, SHA_A, registry

from jena_image_render import models
from jena_image_render.folder import ModelsFolder


class ModelsTests(unittest.TestCase):
    def test_sources_become_url_or_local(self) -> None:
        reg = registry(Path("/base"))
        z = reg.model("zimage").files
        self.assertEqual(
            z["diffusion"].url,
            "https://huggingface.co/org/zimage/resolve/rev1/split/z.safetensors",
        )
        self.assertEqual(z["diffusion"].dest, "diffusion_models/z.safetensors")
        self.assertIn("/resolve/main/te/", z["text_encoder"].url)
        self.assertEqual(z["text_encoder"].sha256, SHA_A)
        ckpt = reg.model("ill").files["checkpoint"]
        self.assertEqual(ckpt.local, "/base/ckpt/ill.safetensors")
        self.assertEqual(ckpt.name, "ill.safetensors")

    def test_loras_live_under_their_engine(self) -> None:
        lora = registry().lora("zstyle")
        self.assertEqual(lora.file.dest, "loras/zimage/zs.safetensors")
        self.assertEqual(lora.file.name, "zimage/zs.safetensors")
        self.assertEqual(lora.trigger, "zst")

    def test_engine_needs_its_files(self) -> None:
        data = {"version": 1, "models": {"z": {"engine": "zimage", "files": {"diffusion": "x"}}}}
        with self.assertRaisesRegex(models.ModelsError, "also needs text_encoder, vae"):
            models.parse(data)
        data = {"version": 1, "models": {"i": {"engine": "illustrious", "files": {"checkpoint": "a", "vae": "b"}}}}
        with self.assertRaisesRegex(models.ModelsError, "does not use"):
            models.parse(data)

    def test_unknown_engine(self) -> None:
        data = {"version": 1, "models": {"p": {"engine": "pony", "files": {}}}}
        with self.assertRaisesRegex(ValueError, "unknown engine 'pony'"):
            models.parse(data)

    def test_same_name_different_file_conflicts(self) -> None:
        data = copy.deepcopy(REGISTRY_DATA)
        data["models"]["klein"]["files"]["text_encoder"] = "https://example.com/x/qwen_3_4b.safetensors"
        with self.assertRaisesRegex(models.ModelsError, 'give one of them a "name"'):
            models.parse(data)
        data["models"]["klein"]["files"]["text_encoder"] = {
            "source": "https://example.com/x/qwen_3_4b.safetensors",
            "name": "qwen_3_4b_klein.safetensors",
        }
        models.parse(data)

    def test_bad_names_and_hashes(self) -> None:
        for raw in ({"source": "x", "name": "../evil"}, {"source": "x", "sha256": "123"}):
            data = {"version": 1, "models": {"i": {"engine": "illustrious", "files": {"checkpoint": raw}}}}
            with self.subTest(raw=raw), self.assertRaises(models.ModelsError):
                models.parse(data)

    def test_unknown_alias_lists_known(self) -> None:
        with self.assertRaisesRegex(models.ModelsError, "known: ill, klein, krea2, qwen, zimage"):
            registry().model("sdxl")

    def test_load_is_the_catalog_with_the_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = ModelsFolder(Path(tmp) / "models")
            reg = models.load(folder)
            self.assertEqual(reg, models.catalog())
            self.assertIs(reg.folder, folder)
            self.assertIsNone(reg.remote)
            remote = models.load(folder, remote=lambda top: [])
            self.assertIsNotNone(remote.remote)
        self.assertEqual(reg.model("zimage").engine, "zimage")

    def test_no_models_json_is_read(self) -> None:
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                Path("models.json").write_text("not even json")
                os.environ["JENA_MODELS"] = "models.json"
                reg = models.load(ModelsFolder(Path(tmp) / "models"))
            finally:
                os.environ.pop("JENA_MODELS", None)
                os.chdir(cwd)
        self.assertEqual(set(reg.models), set(models.catalog().models))
        self.assertFalse(hasattr(models, "DEFAULT_FILE"))

    def test_catalog_has_no_base(self) -> None:
        data = {"version": 1, "models": {"x": {"base": "zimage", "files": {}}}}
        with self.assertRaises(ValueError):
            models.parse(data)

if __name__ == "__main__":
    unittest.main()
