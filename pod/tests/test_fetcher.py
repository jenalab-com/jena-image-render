"""Downloads must stay inside the models folder and never trust partial files."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from jena_fetch.fetcher import Fetcher, FetchItem, resolve_dest


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body, self.status_code = body, status
        self.headers = {"content-length": str(len(body))}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def item(body: bytes, dest: str, url: str = "https://huggingface.co/x") -> FetchItem:
    return FetchItem(url, dest, hashlib.sha256(body).hexdigest(), len(body))


class FetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)
        self.calls: list[tuple[str, dict]] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def opener(self, body: bytes):
        def open_(url: str, headers: dict, stream: bool, timeout):
            self.calls.append((url, headers))
            return FakeResponse(body)

        return open_

    def test_rejects_paths_outside_models(self) -> None:
        for bad in ("../x.safetensors", "/etc/passwd", "loras/../../x", ""):
            with self.subTest(dest=bad), self.assertRaises(ValueError):
                resolve_dest(self.models, bad)

    def test_downloads_verifies_and_renames(self) -> None:
        body = b"weights" * 1000
        fetcher = Fetcher(self.models, opener=self.opener(body))
        fetcher.submit([item(body, "vae/ae.safetensors")])
        state = fetcher.wait(5)["vae/ae.safetensors"]
        self.assertEqual(state["status"], "done")
        self.assertEqual((self.models / "vae/ae.safetensors").read_bytes(), body)
        self.assertFalse(any(self.models.rglob("*.part")))

    def test_hash_mismatch_deletes_file(self) -> None:
        bad = FetchItem("https://huggingface.co/x", "vae/ae.safetensors", "0" * 64, 3)
        fetcher = Fetcher(self.models, opener=self.opener(b"abc"))
        fetcher.submit([bad])
        state = fetcher.wait(5)["vae/ae.safetensors"]
        self.assertEqual(state["status"], "error")
        self.assertIn("sha256", state["error"])
        self.assertFalse((self.models / "vae/ae.safetensors").exists())

    def test_existing_verified_file_is_skipped(self) -> None:
        body = b"x" * 10
        fetcher = Fetcher(self.models, opener=self.opener(body))
        fetcher.submit([item(body, "loras/a.safetensors")])
        fetcher.wait(5)
        fetcher.submit([item(body, "loras/a.safetensors")])
        self.assertEqual(fetcher.wait(5)["loras/a.safetensors"]["status"], "done")
        self.assertEqual(len(self.calls), 1)

    def test_hf_token_only_for_huggingface(self) -> None:
        body = b"z"
        fetcher = Fetcher(self.models, hf_token="hf-secret", opener=self.opener(body))
        hf = "https://huggingface.co/org/gated/resolve/abc/w.safetensors"
        other = "https://example.com/w.safetensors"
        fetcher.submit([item(body, "diffusion_models/a.safetensors", hf), item(body, "vae/b.safetensors", other)])
        fetcher.wait(5)
        headers = {url: h for url, h in self.calls}
        self.assertEqual(headers[hf]["Authorization"], "Bearer hf-secret")
        self.assertNotIn("Authorization", headers[other])

    def test_civitai_token_only_for_civitai(self) -> None:
        body = b"y"
        fetcher = Fetcher(self.models, civitai_token="secret", opener=self.opener(body))
        civitai = "https://civitai.com/api/download/models/1"
        fetcher.submit(
            [
                item(body, "loras/c.safetensors", civitai),
                item(body, "loras/h.safetensors", "https://huggingface.co/h"),
            ]
        )
        fetcher.wait(5)
        headers = {url: h for url, h in self.calls}
        self.assertEqual(headers[civitai]["Authorization"], "Bearer secret")
        self.assertNotIn("Authorization", headers["https://huggingface.co/h"])

    def test_no_hash_downloads_and_records_url(self) -> None:
        body = b"user storage" * 100
        fetcher = Fetcher(self.models, opener=self.opener(body))
        url = "https://storage.example.com/my.safetensors"
        fetcher.submit([FetchItem(url, "loras/zimage/my.safetensors")])
        state = fetcher.wait(5)["loras/zimage/my.safetensors"]
        self.assertEqual(state["status"], "done")
        self.assertEqual(state["total"], len(body))  # from Content-Length
        target = self.models / "loras/zimage/my.safetensors"
        self.assertEqual(target.read_bytes(), body)
        marker = target.with_name(target.name + ".sha256").read_text()
        self.assertEqual(marker, hashlib.sha256(body).hexdigest())

    def test_no_hash_same_url_is_skipped_other_url_refetches(self) -> None:
        body = b"v1"
        fetcher = Fetcher(self.models, opener=self.opener(body))
        dest = "vae/v.safetensors"
        fetcher.submit([FetchItem("https://a.example/v", dest)])
        fetcher.wait(5)
        fetcher.submit([FetchItem("https://a.example/v", dest)])
        fetcher.wait(5)
        self.assertEqual(len(self.calls), 1)
        fetcher.submit([FetchItem("https://b.example/v", dest)])
        fetcher.wait(5)
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
