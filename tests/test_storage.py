import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jena_image_render import models, storage
from jena_image_render.config import expand
from jena_image_render.folder import ModelsFolder
from jena_image_render.models import FileSpec

SHA = "e" * 64


class FakeHub:
    """huggingface_hub.HfApi as far as storage uses it."""

    def __init__(self, files=None, fail: Exception | None = None) -> None:
        self.repos: list[tuple] = []
        self.uploads: list[dict] = []
        self.listed: list[tuple] = []
        self.files = files or {}  # path → (sha256, size)
        self.fail = fail

    def create_repo(self, repo_id, **kwargs):
        self.repos.append((repo_id, kwargs))

    def model_info(self, repo_id, **kwargs):
        self.listed.append((repo_id, kwargs))
        if self.fail:
            raise self.fail
        siblings = [
            SimpleNamespace(
                rfilename=path, size=size, lfs={"sha256": sha, "size": size, "pointer_size": 1}
            )
            for path, (sha, size) in self.files.items()
        ]
        siblings.append(SimpleNamespace(rfilename=".gitattributes", size=9, lfs=None))
        return SimpleNamespace(sha="head1", siblings=siblings)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(oid=f"commit{len(self.uploads)}")


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "storage.json"
        self.hub = FakeHub()
        self.tokens: list[str] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self, repo: str = "someone/models", hub=None) -> storage.HFStorage:
        hub = hub or self.hub

        def api(token: str) -> FakeHub:
            self.tokens.append(token)
            return hub

        return storage.HFStorage(repo, "hf_write", api=api, state=self.state)

    def spec(self, sha: str = SHA) -> FileSpec:
        return FileSpec("loras/zimage/my style.safetensors", local="/x/my.safetensors", sha256=sha, size=5)

    def test_first_sync_uploads_and_remembers(self) -> None:
        synced = self.store().sync(self.spec())
        self.assertEqual(
            synced.url,
            "https://huggingface.co/someone/models/resolve/commit1/loras/zimage/my%20style.safetensors",
        )
        self.assertEqual((synced.dest, synced.sha256, synced.size), (self.spec().dest, SHA, 5))
        self.assertEqual(self.hub.repos, [("someone/models", {"private": True, "exist_ok": True})])
        (upload,) = self.hub.uploads
        self.assertEqual(
            (upload["path_or_fileobj"], upload["path_in_repo"], upload["repo_id"]),
            ("/x/my.safetensors", "loras/zimage/my style.safetensors", "someone/models"),
        )
        state = json.loads(self.state.read_text())
        self.assertEqual(
            state["loras/zimage/my style.safetensors"],
            {"repo": "someone/models", "sha256": SHA, "size": 5, "revision": "commit1"},
        )
        self.assertEqual(self.tokens, ["hf_write"])

    def test_same_hash_is_reused_without_the_hub(self) -> None:
        first = self.store().sync(self.spec())
        again = self.store().sync(self.spec())  # a new run reads the state file
        self.assertEqual(again.url, first.url)
        self.assertEqual(len(self.hub.uploads), 1)
        self.assertEqual(len(self.tokens), 1)  # no API was even created

    def test_changed_file_or_other_repo_uploads_again(self) -> None:
        self.store().sync(self.spec())
        changed = self.store().sync(self.spec("f" * 64))
        self.assertIn("/resolve/commit2/", changed.url)
        other = self.store("someone/other").sync(self.spec("f" * 64))
        self.assertIn("someone/other/resolve/commit3/", other.url)

    def test_sync_all_leaves_url_files(self) -> None:
        url = FileSpec("vae/v.safetensors", url="https://x.example/v")
        result = storage.sync_all(self.store(), [url, self.spec()])
        self.assertEqual(result[0], url)
        self.assertTrue(result[1].url)
        self.assertEqual(len(self.hub.uploads), 1)

    def test_only_hashed_local_files(self) -> None:
        with self.assertRaises(storage.StorageError):
            self.store().sync(FileSpec("vae/v", local="/v"))

    def test_open_storage_errors(self) -> None:
        saved = os.environ.pop("HF_TOKEN", None)
        try:
            for value, message in (
                ("", "no storage"),
                ("s3://bucket", "only hf://"),
                ("hf://just-owner", "hf://owner/repo"),
                ("hf://owner/repo", "HF_TOKEN"),
            ):
                with self.subTest(value=value), self.assertRaisesRegex(storage.StorageError, message):
                    storage.open_storage(value, state=self.state)
            os.environ["HF_TOKEN"] = "t"
            self.assertEqual(storage.open_storage("hf://owner/repo/", state=self.state).repo, "owner/repo")
        finally:
            os.environ.pop("HF_TOKEN", None)
            if saved is not None:
                os.environ["HF_TOKEN"] = saved

    def test_repo_is_asked_before_uploading(self) -> None:
        hub = FakeHub({"loras/zimage/my style.safetensors": (SHA, 5)})
        synced = self.store(hub=hub).sync(self.spec())
        self.assertEqual(hub.uploads, [])
        self.assertEqual(hub.repos, [])  # nothing to create either
        self.assertIn("/resolve/head1/loras/zimage/my%20style.safetensors", synced.url)
        self.assertEqual(hub.listed, [("someone/models", {"files_metadata": True})])
        state = json.loads(self.state.read_text())
        self.assertEqual(state[self.spec().dest]["revision"], "head1")
        self.store(hub=hub).sync(self.spec())  # the record now answers
        self.assertEqual(len(hub.listed), 1)

    def test_other_hash_in_the_repo_is_uploaded(self) -> None:
        hub = FakeHub({"loras/zimage/my style.safetensors": ("f" * 64, 5)})
        store = self.store(hub=hub)
        synced = store.sync(self.spec())
        self.assertIn("/resolve/commit1/", synced.url)
        store.sync(FileSpec("vae/other.safetensors", local="/x/o", sha256=SHA, size=1))
        self.assertEqual((len(hub.listed), len(hub.uploads)), (1, 2))  # listed once

    def test_a_repo_that_does_not_exist_yet_is_not_a_warning(self) -> None:
        missing = RuntimeError("404 Client Error\n\nRepository Not Found for url ...")
        missing.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
        store = storage.HFStorage("o/r", "t", api=lambda token: FakeHub(fail=missing),
                                  state=self.state)
        with self.assertNoLogs("jena_image_render.storage", "WARNING"):
            self.assertEqual(store.listing(), {})

    def test_other_listing_failures_warn_in_one_line(self) -> None:
        broken = RuntimeError("401 Unauthorized\n\nlong explanation\nmore lines")
        broken.response = SimpleNamespace(status_code=401)  # type: ignore[attr-defined]
        store = storage.HFStorage("o/r", "t", api=lambda token: FakeHub(fail=broken),
                                  state=self.state)
        with self.assertLogs("jena_image_render.storage", "WARNING") as logs:
            self.assertEqual(store.listing(), {})
        self.assertEqual(len(logs.output), 1)
        self.assertNotIn("long explanation", logs.output[0])

    def test_listing_failure_warns_and_uploads(self) -> None:
        hub = FakeHub(fail=OSError("offline"))
        store = self.store(hub=hub)
        with self.assertLogs("jena_image_render.storage", "WARNING") as logs:
            synced = store.sync(self.spec())
            self.assertEqual(store.files("loras"), [])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("offline", logs.output[0])
        self.assertIn("/resolve/commit1/", synced.url)

    def test_files_by_role_folder(self) -> None:
        hub = FakeHub({
            "checkpoints/sdxl/mine.safetensors": (SHA, 7),
            "loras/zimage/z.safetensors": ("f" * 64, 3),
            "loras/README.md": ("", 1),
        })
        store = self.store(hub=hub)
        (ckpt,) = store.files("checkpoints")
        self.assertEqual(
            ckpt,
            FileSpec(
                "checkpoints/sdxl/mine.safetensors",
                url="https://huggingface.co/someone/models/resolve/head1/checkpoints/sdxl/mine.safetensors",
                sha256=SHA,
                size=7,
            ),
        )
        self.assertEqual([f.dest for f in store.files("loras")], ["loras/zimage/z.safetensors"])
        self.assertEqual(store.files("vae"), [])
        self.assertEqual(len(hub.listed), 1)

    def lookup(self, hub: FakeHub, tmp: str) -> tuple[models.Registry, list]:
        opened = []

        def opener() -> storage.HFStorage:
            opened.append(1)
            return self.store(hub=hub)

        lookup = storage.Lookup(opener)
        return models.load(ModelsFolder(Path(tmp) / "models"), lookup), opened

    def test_names_found_in_storage(self) -> None:
        hub = FakeHub({
            "checkpoints/mine.safetensors": (SHA, 7),
            "diffusion_models/krea2/tuned.safetensors": (SHA, 8),
            "diffusion_models/loose.safetensors": (SHA, 8),
            "loras/zimage/z-style.safetensors": ("f" * 64, 3),
            "loras/any.safetensors": ("f" * 64, 4),
        })
        reg, opened = self.lookup(hub, self.tmp.name)
        self.assertEqual(reg.model("zimage").engine, "zimage")
        self.assertEqual((opened, hub.listed), ([], []))  # lazy: nothing asked yet
        mine = reg.model("mine").files["checkpoint"]
        self.assertEqual((mine.url.split("/resolve/")[1], mine.sha256, mine.size, mine.local),
                         ("head1/checkpoints/mine.safetensors", SHA, 7, ""))
        tuned = reg.model("tuned")
        self.assertEqual(tuned.engine, "krea2")
        self.assertEqual(tuned.files["vae"], reg.model("krea2").files["vae"])
        with self.assertRaisesRegex(models.ModelsError, "engine of diffusion_models/loose"):
            reg.model("loose")
        self.assertEqual(reg.lora("z-style").engine, "zimage")
        self.assertEqual(reg.lora("any").engine, "")
        (r,) = expand({"version": 1, "model": "tuned", "prompt": "x", "loras": ["any"]}, reg)
        self.assertEqual(r.engine, "krea2")
        with self.assertRaisesRegex(models.ModelsError, "or in your storage"):
            reg.model("nowhere")
        self.assertEqual((len(opened), len(hub.listed)), (1, 1))  # once per run

    def test_models_folder_beats_storage(self) -> None:
        local = Path(self.tmp.name) / "models/checkpoints/mine.safetensors"
        local.parent.mkdir(parents=True)
        local.write_bytes(b"x")
        hub = FakeHub({"checkpoints/mine.safetensors": (SHA, 7)})
        reg, opened = self.lookup(hub, self.tmp.name)
        self.assertEqual(reg.model("mine").files["checkpoint"].local, str(local))
        self.assertEqual(opened, [])

    def test_storage_failure_is_one_warning_then_not_found(self) -> None:
        reg, _ = self.lookup(FakeHub(fail=PermissionError("401 bad token")), self.tmp.name)
        with self.assertLogs("jena_image_render", "WARNING") as logs:
            for _ in range(2):
                with self.assertRaisesRegex(models.ModelsError, "is not a known alias"):
                    reg.model("mine")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("401 bad token", logs.output[0])

        def broken() -> storage.HFStorage:
            raise storage.StorageError("JENA_STORAGE needs HF_TOKEN")

        reg = models.load(ModelsFolder(Path(self.tmp.name)), storage.Lookup(broken))
        with self.assertLogs("jena_image_render", "WARNING") as logs:
            with self.assertRaises(models.ModelsError):
                reg.lora("mine")
            with self.assertRaises(models.ModelsError):
                reg.model("mine")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Storage not searched: JENA_STORAGE needs HF_TOKEN", logs.output[0])

    def test_missing_huggingface_hub_says_how_to_install(self) -> None:
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            store = storage.HFStorage("o/r", "t", state=self.state)
            with self.assertRaisesRegex(storage.StorageError, r"pip install huggingface_hub"):
                with self.assertLogs("jena_image_render.storage", "WARNING"):
                    store.sync(self.spec())
            self.assertEqual(store.files("loras"), [])  # the failed listing is kept
        finally:
            if saved is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = saved


if __name__ == "__main__":
    unittest.main()
