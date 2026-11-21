import hashlib
import tempfile
import unittest
from pathlib import Path

from jena_image_render import assets, pull
from jena_image_render.folder import ModelsFolder
from jena_image_render.models import FileSpec

DATA = b"0123456789" * 100
SHA = hashlib.sha256(DATA).hexdigest()
URL = "https://huggingface.co/org/repo/resolve/abc/vae/ae.safetensors"


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status_code = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def iter_content(self, size):
        for i in range(0, len(self.body), 300):
            yield self.body[i : i + 300]


class FakeOpener:
    """Serves DATA, honouring Range unless told not to."""

    def __init__(self, ranges: bool = True, body: bytes = DATA) -> None:
        self.ranges = ranges
        self.body = body
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers, stream, timeout):
        self.calls.append((url, dict(headers)))
        start = headers.get("Range", "")
        if start and self.ranges:
            offset = int(start.split("=")[1].rstrip("-"))
            if offset >= len(self.body):
                return FakeResponse(416, b"")
            return FakeResponse(206, self.body[offset:])
        return FakeResponse(200, self.body)


class PullTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = ModelsFolder(self.root / "models")
        self.hashes = assets.Hashes(self.root / "sha256.json")
        self.spec = FileSpec("vae/ae.safetensors", url=URL, sha256=SHA, size=len(DATA))
        self.target = self.folder.path(self.spec.dest)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_pull(self, opener, spec=None) -> int:
        return pull.pull([spec or self.spec], self.folder, self.hashes, "hf_t", opener)

    def test_downloads_verifies_and_sends_token_to_hf_only(self) -> None:
        opener = FakeOpener()
        self.assertEqual(self.run_pull(opener), 1)
        self.assertEqual(self.target.read_bytes(), DATA)
        self.assertFalse(self.target.with_name("ae.safetensors.part").exists())
        self.assertEqual(opener.calls[0][1], {"Authorization": "Bearer hf_t"})
        self.assertEqual(pull.headers("https://cdn.example.com/x", "hf_t"), {})
        self.assertEqual(pull.headers("https://evilhuggingface.co/x", "hf_t"), {})

    def test_present_with_right_hash_is_skipped(self) -> None:
        self.run_pull(FakeOpener())
        opener = FakeOpener()
        self.assertEqual(self.run_pull(opener), 0)
        self.assertEqual(opener.calls, [])

    def test_wrong_file_is_replaced(self) -> None:
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"old")
        self.assertEqual(self.run_pull(FakeOpener()), 1)
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_resumes_a_part_file(self) -> None:
        self.target.parent.mkdir(parents=True)
        self.target.with_name("ae.safetensors.part").write_bytes(DATA[:400])
        opener = FakeOpener()
        self.run_pull(opener)
        self.assertEqual(opener.calls[0][1]["Range"], "bytes=400-")
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_server_without_ranges_restarts(self) -> None:
        self.target.parent.mkdir(parents=True)
        self.target.with_name("ae.safetensors.part").write_bytes(b"garbage")
        self.run_pull(FakeOpener(ranges=False))
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_complete_part_is_finished(self) -> None:
        self.target.parent.mkdir(parents=True)
        self.target.with_name("ae.safetensors.part").write_bytes(DATA)
        self.run_pull(FakeOpener())
        self.assertEqual(self.target.read_bytes(), DATA)

    def test_hash_mismatch_removes_the_download(self) -> None:
        with self.assertRaisesRegex(pull.PullError, "sha256 mismatch"):
            self.run_pull(FakeOpener(body=b"tampered"))
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_name("ae.safetensors.part").exists())

    def test_local_only_files_are_not_downloaded(self) -> None:
        opener = FakeOpener()
        spec = FileSpec("loras/x.safetensors", local="/somewhere/x.safetensors")
        self.assertEqual(self.run_pull(opener, spec), 0)
        self.assertEqual(opener.calls, [])


if __name__ == "__main__":
    unittest.main()
