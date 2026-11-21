"""Chunked uploads: in-order appends, hash check on the last chunk, safe paths."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from jena_fetch.upload import UploadError, append_chunk, is_verified


class UploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)
        self.body = b"lora-weights" * 1000
        self.sha = hashlib.sha256(self.body).hexdigest()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def send(self, data: bytes, offset: int, dest: str = "loras/q/a.safetensors") -> str:
        return append_chunk(self.models, dest, self.sha, offset, len(self.body), data)

    def test_chunks_assemble_and_verify(self) -> None:
        self.assertEqual(self.send(self.body[:5000], 0), "partial")
        self.assertEqual(self.send(self.body[5000:], 5000), "done")
        target = self.models / "loras/q/a.safetensors"
        self.assertEqual(target.read_bytes(), self.body)
        self.assertTrue(is_verified(self.models, "loras/q/a.safetensors", self.sha))
        self.assertFalse(any(self.models.rglob("*.part")))

    def test_out_of_order_chunk_rejected(self) -> None:
        self.send(self.body[:5000], 0)
        with self.assertRaisesRegex(UploadError, "offset"):
            self.send(self.body[6000:], 6000)

    def test_restart_from_zero_replaces_partial(self) -> None:
        self.send(self.body[:100], 0)
        self.assertEqual(self.send(self.body, 0), "done")

    def test_hash_mismatch_discards_file(self) -> None:
        bad = b"x" * len(self.body)
        with self.assertRaisesRegex(UploadError, "sha256"):
            self.send(bad, 0)
        self.assertFalse((self.models / "loras/q/a.safetensors").exists())
        self.assertFalse(any(self.models.rglob("*.part")))

    def test_rejects_paths_outside_model_folders(self) -> None:
        for dest in ("../x.safetensors", "/tmp/x", "custom_nodes/x.py", "x.safetensors", "loras/../vae/x"):
            with self.subTest(dest=dest), self.assertRaises(ValueError):
                self.send(self.body, 0, dest)

    def test_accepts_model_folders(self) -> None:
        for dest in (
            "checkpoints/pie.safetensors",
            "diffusion_models/krea2.safetensors",
            "text_encoders/qwen.safetensors",
            "vae/vae.safetensors",
            "upscale_models/4x.pth",
            "ultralytics/bbox/face_yolov8m.pt",
        ):
            with self.subTest(dest=dest):
                self.assertEqual(self.send(self.body, 0, dest), "done")
                self.assertTrue((self.models / dest).is_file())

    def test_chunk_past_total_rejected(self) -> None:
        with self.assertRaisesRegex(UploadError, "total"):
            append_chunk(self.models, "loras/a.safetensors", self.sha, 0, 3, b"abcd")


if __name__ == "__main__":
    unittest.main()
