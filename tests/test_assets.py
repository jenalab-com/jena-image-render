import hashlib
import tempfile
import unittest
from pathlib import Path

from helpers import registry

from jena_image_render import assets
from jena_image_render.config import expand
from jena_image_render.models import FileSpec


class FakeClient:
    def __init__(self, options=None, fetch_states=None) -> None:
        self._options = options or {}
        self.fetched: list[FileSpec] = []
        self.uploaded: list[str] = []
        self.states = fetch_states or []

    def options(self, folder):
        return self._options.get(folder, [])

    def fetch(self, files):
        self.fetched.extend(files)

    def upload(self, item):
        self.uploaded.append(item.dest)

    def fetch_status(self):
        return self.states.pop(0)


def renders(**job):
    data = {"version": 1, **job}
    return expand(data, registry(), "t")


class AssetTests(unittest.TestCase):
    def test_needed_dedupes_shared_files(self) -> None:
        todo = renders(defaults={"model": ["zimage", "klein"], "seeds": [1, 2]}, jobs=[{"prompt": "x"}])
        dests = [f.dest for f in assets.needed(todo, registry())]
        self.assertEqual(len(dests), len(set(dests)))
        self.assertEqual(dests.count("text_encoders/qwen_3_4b.safetensors"), 1)
        self.assertEqual(len(dests), 5)

    def test_pod_names_are_models_json_names(self) -> None:
        (r,) = renders(model="zimage", prompt="x", loras=["zstyle"])
        names = assets.pod_names(r, registry())
        self.assertEqual(names.base, "z.safetensors")
        self.assertEqual(names.loras, ("zimage/zs.safetensors",))

    def test_local_names_match_by_file_name(self) -> None:
        client = FakeClient(
            {
                "checkpoints": ["sdxl/ill.safetensors", "other.safetensors"],
                "loras": ["mine/is.safetensors"],
            }
        )
        (r,) = renders(model="ill", prompt="x", loras=["istyle"])
        names = assets.LocalNames(client).names(r, registry())
        self.assertEqual(names.base, "sdxl/ill.safetensors")
        self.assertEqual(names.loras, ("mine/is.safetensors",))

    def test_local_missing_or_ambiguous(self) -> None:
        spec = FileSpec("checkpoints/ill.safetensors", local="/x/ill.safetensors")
        with self.assertRaisesRegex(assets.AssetError, "not in your ComfyUI models/checkpoints/"):
            assets.LocalNames(FakeClient()).find(spec)
        two = FakeClient({"checkpoints": ["a/ill.safetensors", "b/ill.safetensors"]})
        with self.assertRaisesRegex(assets.AssetError, "ambiguous"):
            assets.LocalNames(two).find(spec)

    def test_local_missing_hints_at_folder_and_pull(self) -> None:
        spec = FileSpec("vae/ae.safetensors", url="https://x/ae.safetensors")
        with self.assertRaises(assets.AssetError) as ctx:
            assets.LocalNames(FakeClient()).find(spec)
        for hint in ("pull JOB.json", "JENA_MODELS_DIR", "extra_model_paths.yaml",
                     "ComfyUI on another computer cannot be filled from here"):
            self.assertIn(hint, str(ctx.exception))
        self.assertEqual(assets.LocalNames(FakeClient()).lookup(spec), "")

    def test_plan_per_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            here = Path(tmp) / "w.safetensors"
            here.write_bytes(b"w")
            url = FileSpec("vae/v", url="https://x/v")
            both = FileSpec("vae/b", url="https://x/b", local=str(here))
            local = FileSpec("loras/w", local=str(here))
            gone = FileSpec("loras/g", local=f"{tmp}/gone")
            files = [url, both, local, gone]

            def how(target, storage=False):
                return [h for h, _ in assets.plan(files, target, storage)]

            self.assertEqual(how("runpod"), ["fetch", "fetch", "upload", "missing"])
            self.assertEqual(how("runpod", True), ["fetch", "fetch", "storage", "missing"])
            self.assertEqual(how(""), how("runpod"))
            self.assertEqual(how("local"), ["local"] * 4)

    def test_url_wins_over_a_local_copy_on_the_pod(self) -> None:
        both = FileSpec("vae/b", url="https://x/b", local="/not/hashed")
        self.assertEqual(assets.prepare_uploads([both], assets.Hashes(Path("/nonexistent/c.json"))), [both])
        client = FakeClient(fetch_states=[{"vae/b": {"status": "done"}}])
        assets.deliver(client, [both], sleep=lambda s: None)
        self.assertEqual((len(client.fetched), client.uploaded), (1, []))

    def test_prepare_uploads_hashes_and_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "w.safetensors"
            file.write_bytes(b"weights")
            sha = hashlib.sha256(b"weights").hexdigest()
            hashes = assets.Hashes(Path(tmp) / "cache.json")
            local = FileSpec("loras/zimage/w.safetensors", local=str(file))
            url = FileSpec("vae/v.safetensors", url="https://x/v")
            ready = assets.prepare_uploads([local, url], hashes)
            self.assertEqual((ready[0].sha256, ready[0].size), (sha, 7))
            self.assertEqual(ready[1], url)
            wrong = FileSpec(local.dest, local=str(file), sha256="b" * 64)
            with self.assertRaisesRegex(assets.AssetError, "differs"):
                assets.prepare_uploads([wrong], hashes)
            with self.assertRaisesRegex(assets.AssetError, "not found"):
                assets.prepare_uploads([FileSpec("vae/n", local=f"{tmp}/none")], hashes)

    def test_deliver_fetches_uploads_then_waits(self) -> None:
        states = [
            {"vae/v": {"status": "downloading", "bytes": 1, "total": 2}},
            {"vae/v": {"status": "done", "bytes": 2, "total": 2}},
        ]
        client = FakeClient(fetch_states=states)
        files = [
            FileSpec("vae/v", url="https://x/v"),
            FileSpec("loras/zimage/b", local="/b", sha256="c" * 64, size=20),
            FileSpec("loras/zimage/a", local="/a", sha256="d" * 64, size=10),
        ]
        assets.deliver(client, files, sleep=lambda s: None)
        self.assertEqual([f.dest for f in client.fetched], ["vae/v"])
        self.assertEqual(client.uploaded, ["loras/zimage/a", "loras/zimage/b"])  # small first
        self.assertEqual(states, [])

    def test_failed_download_raises(self) -> None:
        client = FakeClient(fetch_states=[{"vae/v": {"status": "error", "error": "403"}}])
        with self.assertRaisesRegex(Exception, "403"):
            assets.deliver(client, [FileSpec("vae/v", url="https://x/v")], sleep=lambda s: None)


class UploadingClient(FakeClient):
    def upload(self, item, progress=None):
        self.uploaded.append(item.dest)
        if progress is not None:
            progress(item.size // 2, item.size)
            progress(item.size, item.size)


class DeliverProgressTests(unittest.TestCase):
    def test_progress_for_uploads_and_downloads(self) -> None:
        states = [
            {"vae/v": {"status": "downloading", "bytes": 1, "total": 2}},
            {"vae/v": {"status": "done", "bytes": 2, "total": 2}},
        ]
        client = UploadingClient(fetch_states=states)
        files = [FileSpec("vae/v", url="https://x/v"),
                 FileSpec("loras/zimage/a", local="/a", sha256="d" * 64, size=10)]
        events = []
        assets.deliver(client, files, sleep=lambda s: None, progress=events.append)
        got = [(e["dest"], e["how"], e["bytes"], e["status"]) for e in events]
        self.assertEqual(got, [
            ("loras/zimage/a", "upload", 5, "uploading"),
            ("loras/zimage/a", "upload", 10, "uploading"),
            ("loras/zimage/a", "upload", 10, "done"),
            ("vae/v", "fetch", 1, "downloading"),
            ("vae/v", "fetch", 2, "done"),
        ])


if __name__ == "__main__":
    unittest.main()
